"""
最小 HE 训练测试：仅使用有标签数据，验证模型前向/反向、损失计算、梯度更新。
不启用 Teacher/EMA/自训练，单纯测试多任务有监督训练。
"""

import torch
import torch.optim as optim
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))  # 确保 train/ 在 sys.path 中

from models.he_model import HEModel
from losses.HE_losses import he_losses
from datasets.dataloader import HEDataset, HEValTransform
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------- 配置 ----------
CONFIG = {
    'he_classes': 6,
    'base_channels': 64,
    'lr': 1e-4,
    'batch_size': 2,
    'size': 512,
    'task_weights': {
        'semantic': 1.0,
        'boundary': 0.5,
        'colorization': 0.8
    }
}

# ---------- 数据集 ----------
# 修改为你的真实 labeled 路径
LABELED_ROOT = r'D:\\files\\HE\\test\\labeled'
val_transform = HEValTransform(size=CONFIG['size'])
dataset = HEDataset(
    root=LABELED_ROOT,
    labeled=True,
    transform=val_transform,
    image_dir='images',
    mask_dir='masks',
    boundary_dir=None
)
loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)

# ---------- 模型、优化器 ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HEModel(he_classes=CONFIG['he_classes'], base=CONFIG['base_channels']).to(device)
optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])

print(f"Device: {device}")
print(f"Training samples: {len(dataset)}")

# ---------- 运行几个 batch ----------
model.train()
epochs_to_test = 2
for epoch in range(epochs_to_test):
    epoch_loss = 0.0
    
    epoch_bar = tqdm(range(1), desc=f"Epoch {epoch+1}/{epochs_to_test}", position=0, leave=True, unit="epoch")
    epoch_bar.update(0)  # 初始化进度条
    
    batch_bar = tqdm(loader, desc=f"  Batch", position=1, leave=True, unit="batch")
    for batch_idx, batch in enumerate(batch_bar):        # 仅使用有标签样本（本测试数据集全是 labeled）
        branch_a = batch['branch_a'].to(device)          # (B,3,H,W)
        branch_b_gray = batch['branch_b_gray'].to(device)  # (B,1,H,W)
        original_rgb = batch['original_rgb'].to(device)
        semantic = batch['semantic'].to(device)
        boundary = batch['boundary'].to(device)

        optimizer.zero_grad()

        # 前向
        outputs = model(branch_a, branch_b_gray)

        # 准备 labels 字典
        labels = {
            'semantic': semantic,
            'boundary': boundary,
            'original_rgb': original_rgb
        }

        # 计算损失
        loss_total, loss_dict = he_losses(outputs, labels, CONFIG['task_weights'])

        # 反向
        loss_total.backward()
        optimizer.step()

        epoch_loss += loss_total.item()

        # if batch_idx == 0:
        #     print(f"Epoch {epoch}, Batch {batch_idx}: total_loss={loss_total.item():.4f}")
        #     print(f"  Sub losses: { {k: f'{v:.4f}' for k, v in loss_dict.items()} }")
        
        # if batch_idx % 50 == 0:
        #     print(f"Epoch {epoch}, Batch {batch_idx}: loss={loss_total.item():.4f}")
        batch_bar.set_postfix(loss=f"{loss_total.item():.4f}")
    
    epoch_bar.update(1)
    epoch_bar.set_postfix(avg_loss=f"{epoch_loss/len(loader):.4f}")
    epoch_bar.close()   # 外层进度条更新

    print(f"Epoch {epoch+1} 平均损失: {epoch_loss / len(loader):.4f}")

    # 简单检查梯度范数
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.norm().item() ** 2
    print(f"Gradient norm: {total_norm ** 0.5:.4f}")

print("HE supervised training test PASSED.")