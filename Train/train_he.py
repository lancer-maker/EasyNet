#!/usr/bin/env python
"""
HE 半监督训练入口：有监督多任务 + Teacher-Student 自训练
优化点：
- EMA 每步更新
- 梯度裁剪
- 集成验证 mIoU
- 伪标签全图有效（无 ignore_index 填充）
"""
import torch
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import yaml
from tqdm import tqdm
from torch.amp import autocast
from datetime import datetime
import torchvision.utils as vutils

# 确保可以导入项目内模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.he_model import HEModel
from losses.HE_losses import he_losses
from schedulers.HE_scheduler import HEScheduler
from schedulers.ema import EMAUpdater
from utils.pseudo_label_he import PseudoLabeler
from utils.metrics import compute_miou
from datasets.dataloader import get_dataloader

# 类别颜色定义（0-5，255忽略）
CLASS_COLORS = torch.tensor([
    [0, 0, 0],          # 0 背景（黑色）
    [128, 128, 128],    # 1 正常组织（灰色）
    [0, 255, 255],      # 2 STAS 实体（青色）
    [0, 255, 0],        # 3 STAS 微乳头（亮绿）
    [255, 255, 0],      # 4 STAS 单细胞（黄色）
    [255, 0, 0],        # 5 肿瘤主体（红色）
], dtype=torch.uint8)

def label_to_color(label_tensor, colors=CLASS_COLORS, ignore_color=[128,128,128]):
    """
    将 (B, H, W) 类别索引图映射为 (B, 3, H, W) RGB 彩色图，归一化到 [0,1]。
    忽略区域 (255) 会被染成 ignore_color（默认灰色）。
    """
    B, H, W = label_tensor.shape
    # 初始化为忽略颜色（灰色）
    color_img = torch.full((B, 3, H, W), fill_value=128, dtype=torch.uint8, device=label_tensor.device)
    for cls_idx in range(len(colors)):
        mask = (label_tensor == cls_idx)
        for c in range(3):
            color_img[:, c, :, :][mask] = colors[cls_idx, c]
    return color_img.float() / 255.0

def train_one_epoch(model, teacher, loader, optimizer, scheduler, pseudo_labeler,
                    ema_updater, labeled_count, device, epoch, total_epochs,
                    grad_clip=1.0):# grad_clip梯度裁剪
    """
    labeled_count: 每个 batch 中有标签样本的数量（来自 ConcatDataset，前半部分）
    """
    model.train()
    teacher.eval()

    total_loss = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs}")
    for batch_idx, batch in enumerate(pbar):
        branch_a = batch['branch_a'].to(device)            # (B, 3, H, W)
        branch_b_gray = batch['branch_b_gray'].to(device)  # (B, 1, H, W)
        original_rgb = batch['original_rgb'].to(device)
        semantic = batch['semantic'].to(device)            # (B, H, W)
        boundary = batch['boundary'].to(device)            # (B, 1, H, W)

        # 拆分有标签 / 无标签
        labeled_mask = (semantic != 255).any(dim=1).any(dim=1)
        unlabeled_mask = ~labeled_mask      # ~取反,等价于 torch.logical_not(labeled_mask)
# ============================================AMP=================================================
        weights, self_training = scheduler.get_weights()
        # -------- Step 1：仅前向传播在 autocast 内（FP16 加速）--------
        with autocast(device_type='cuda', enabled = False):              # False: 禁用 AMP，可能是显卡不支持，先禁用
            if labeled_mask.sum() > 0:
                outputs_labeled = model(
                    branch_a[labeled_mask], branch_b_gray[labeled_mask]
                )
                # labels_labeled 也在此创建，但损失在外面算
                labels_labeled = {
                    'semantic': semantic[labeled_mask],
                    'boundary': boundary[labeled_mask],
                    'original_rgb': original_rgb[labeled_mask]
                }
            else:
                outputs_labeled = None
                labels_labeled = None

            if unlabeled_mask.sum() > 0:
                outputs_unlabeled = model(
                    branch_a[unlabeled_mask], branch_b_gray[unlabeled_mask]
                )
                if self_training:
                    with torch.no_grad():
                        teacher_out = teacher(
                            branch_a[unlabeled_mask], branch_b_gray[unlabeled_mask]
                        )
                else:
                    teacher_out = None
            else:
                outputs_unlabeled = None
                teacher_out = None

        # -------- Step 2：损失计算在 autocast 外部（FP32 精度，防止 NaN）--------
        if labeled_mask.sum() > 0:
            loss_labeled, _ = he_losses(outputs_labeled, labels_labeled, weights)
        else:
            loss_labeled = 0.0

        if unlabeled_mask.sum() > 0:
            # 着色损失
            loss_color, _ = he_losses(
                {'colorization': outputs_unlabeled['colorization']},
                {'original_rgb': original_rgb[unlabeled_mask]},
                {'colorization': weights['colorization']}
            )
            loss_total_unlabeled = loss_color

            # 自训练伪标签损失
            if self_training:
                pseudo = pseudo_labeler(teacher_out, outputs_unlabeled)
                labels_pseudo = {
                    'semantic': pseudo['semantic'],
                    'boundary': pseudo['boundary']
                }
                loss_pseudo, _ = he_losses(
                    {
                        'semantic': outputs_unlabeled['semantic'],
                        'boundary': outputs_unlabeled['boundary']
                    },
                    labels_pseudo,
                    {'semantic': weights['semantic'], 'boundary': weights['boundary']}
                )
                loss_total_unlabeled = loss_color + loss_pseudo
        else:
            loss_total_unlabeled = 0.0

        loss = loss_labeled + loss_total_unlabeled
# =================中期部分数据结果提取==================
        if epoch % 5 == 0 and batch_idx == 0:   # 每10个epoch的第一个batch
            # 使用有标签样本中的第一张图（如果有标签样本存在）
            if labeled_mask.sum() > 0:
                idx = 0  # 取有标签子集的第一个样本
                img = branch_a[labeled_mask][idx:idx+1]  # (1,3,H,W)
                # 从 outputs_labeled 中取对应样本的语义预测
                pred_idx = outputs_labeled['semantic'][idx:idx+1].argmax(dim = 1)  # (1,H,W)
                pred = pred_idx.float().unsqueeze(1) / 5.0   # 归一化到 [0,1]，便于可视化归一化
                pred_gray = pred.repeat(1,3,1,1)  # (1,3,H,W)

                pred_color = label_to_color(pred_idx)  # (1,3,H,W)


                raw_gt = semantic[labeled_mask][idx:idx+1]
                gt = label_to_color(raw_gt) # 原掩码对应颜色区域
                # 保存拼接图
                vutils.save_image(
                    torch.cat([img, pred_gray, pred_color, gt ], dim = 0),
                    f'checkpoints/vis_epoch{epoch}.png',
                    nrow=4
                )
# =====================================
        optimizer.zero_grad()
       
        loss.backward()
        # scaler.scale(loss).backward()                                           # 1. 缩放损失值，并计算梯度

        # 梯度裁剪
        # scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)  # 2. 在梯度裁剪前，必须先将梯度 unscale 还原

        optimizer.step()
        # scaler.step(optimizer)                                                  # # 3. 优化器更新与 Scaler 状态更新
        # scaler.update()

        # EMA 每步更新 Teacher
        ema_updater.update(teacher, model)

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return total_loss / len(loader)

def validate(model, loader, device, num_classes=6):
    """计算验证集 mIoU，忽略 ignore_index=255"""
    model.eval()
    iou_sum = 0.0
    count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            branch_a = batch['branch_a'].to(device)
            branch_b_gray = batch['branch_b_gray'].to(device)
            semantic = batch['semantic'].to(device)       # (B, H, W)

            out = model(branch_a, branch_b_gray)
            pred = out['semantic'].argmax(dim=1)          # (B, H, W)

            # compute_miou 期望 numpy，忽略 255
            miou = compute_miou(
                pred.cpu().numpy(),
                semantic.cpu().numpy(),
                num_classes=num_classes,
                ignore_index=255
            )
            iou_sum += miou
            count += 1
    return iou_sum / max(count, 1)

# from torch.amp import GradScaler
def main(config_path):
    # 加载配置
    with open(config_path, 'r', encoding = 'utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据加载
    train_loader = get_dataloader(config, split='train')
    val_loader = get_dataloader(config, split='val')

    # ConcatDataset 时，labeled_count = len(第一个子数据集)
    labeled_count = len(train_loader.dataset.datasets[0])

    # 创建模型
    model = HEModel(he_classes=config['he_classes'], base=config['base_channels']).to(device)
    teacher = HEModel(he_classes=config['he_classes'], base=config['base_channels']).to(device)
    teacher.load_state_dict(model.state_dict())
    for p in teacher.parameters():
        p.requires_grad_(False)

    # 优化器
    # 确保 lr 为浮点数，避免从配置中读取到字符串导致类型错误
    optimizer = optim.Adam(model.parameters(), lr=float(config['lr']))

    # 调度器
    scheduler = HEScheduler(
        initial_weights=config['task_weights_initial'],
        progress_thresh=config['progress_thresh'],
        sigma_ratio=config['sigma_ratio'],
        total_epochs=config['num_epochs']
    )

    # EMA 更新器
    ema_updater = EMAUpdater(decay=config['ema_decay'])

    # 伪标签生成器
    pseudo_labeler = PseudoLabeler(
        conf_thresh=config['self_training_conf_thresh'],
        teacher_weak_thresh=config['teacher_weak_thresh']
    )

    # scaler = GradScaler(
    #     init_scale=2.**13,    # 8192，默认是 2**16=65536，降低初始缩放
    #     growth_interval=2000, # 增加 step 间隔，减少频繁上调
    #     backoff_factor=0.5,
    #     growth_factor=2.0
    # )

    best_miou = 0.0
    checkpoint_dir = Path('./checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir = Path('./checkpoints/logs')
    log_dir.mkdir(exist_ok=True)

    # ----------日志保存----------
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_csv = log_dir / f'training_log_{run_id}.csv'
    with open(log_csv, 'w') as f:
        f.write('epoch,train_loss,val_miou\n')

    for epoch in range(config['num_epochs']):
        scheduler.step(epoch)

        # 训练
        train_loss = train_one_epoch(
            model, teacher, train_loader, optimizer, scheduler, pseudo_labeler,
            ema_updater, labeled_count, device, epoch, config['num_epochs'],
            grad_clip=1.0   # 可配置
        )

        # 验证
        val_miou = validate(model, val_loader, device, num_classes=config['he_classes'])
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val mIoU={val_miou:.4f}")
            
        with open(log_csv, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.4f},{val_miou:.4f}\n")
        # 保存最佳模型
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), checkpoint_dir / 'best_he_model.pth')
            print(f"  -> New best model saved (mIoU={best_miou:.4f})")
            # ---- 将本 epoch 指标追加写入 CSV ----


    print("Training finished. Best mIoU:", best_miou)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='Train/configs/he.yaml')
    args = parser.parse_args()
    main(args.config)