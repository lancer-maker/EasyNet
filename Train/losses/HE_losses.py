import torch
import torch.nn.functional as F

def dice_loss_multi(pred, target, num_classes, smooth=1e-5, ignore_index=255):
    """
    pred: (B, C, H, W) logits
    target: (B, H, W) class indices
    num_classes: 类别数
    """
    pred = F.softmax(pred, dim=1)  # (B, C, H, W),沿一维归一化
    
    # ---------- 强制清洗所有非法标签 ----------
    # 所有不是 [0, num_classes-1] 且不是 ignore_index 的值，统统视为 ignore_index
    valid_mask = (target >= 0) & (target < num_classes) | (target == ignore_index)
    # 将非法值替换为 ignore_index
    target = torch.where(valid_mask, target, torch.full_like(target, ignore_index))
    # ------------------------------------------

    # 创建安全的 target：将 ignore_index 临时替换为 0，避免 one_hot 越界
    safe_target = torch.where(target == ignore_index, torch.zeros_like(target), target)
    target_one_hot = F.one_hot(safe_target, num_classes).permute(0, 3, 1, 2).float()
    mask = (target != ignore_index).unsqueeze(1).float()  # (B,1,H,W),mask中为1的像素参与计算，为0的像素被忽略
    pred = pred * mask                                    # 语义分割输出
    target_one_hot = target_one_hot * mask

    dice = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    valid_classes = 0

    for c in range(num_classes):
        p_c = pred[:, c, :, :].contiguous().view(-1)           # c类别的预测概率
        t_c = target_one_hot[:, c, :, :].contiguous().view(-1) # c类别的真实标签（0或1）
        intersection = (p_c * t_c).sum()                       # 只保留真实属于第 c 类的像素的预测概率，其他像素归零

        union = p_c.sum() + t_c.sum()                          # 
        if union > 0:
            dice += 1 - (2. * intersection + smooth) / (union + smooth)
            valid_classes += 1
        '''
        假阳性：真实标签中没有第 c 类（t_c=0），但模型预测了第 c 类（p_c>0），dice=1,此时将迫使模型降虚假的预测概率
        '''
    # return dice / num_classes # 只在类别真实存在或者网络有预测时计算，避免NaN
    return dice / max(valid_classes, 1)


def he_losses(outputs, labels, weights):
    """
    outputs: dict with 'semantic', 'boundary', 'colorization'
    labels: dict with 'semantic' (B,H,W class indices),
                       'boundary' (B,1,H,W binary),
                       'original_rgb' (B,3,H,W [0,1])
                       'is_labeled' (B,) bool
    weights: dict with keys 'semantic', 'boundary', 'colorization'
    """
    loss_total = torch.tensor(0.0, device=outputs['colorization'].device)
    loss_dict = {}
    
    # 获取监督标签，如果存在 'is_labeled' (Shape: [B])，则过滤出有标签的样本
    is_labeled = labels.get('is_labeled', None)
    
    # 提取有标签的样本掩码
    labeled_mask = (is_labeled == 1) if is_labeled is not None else torch.ones(outputs['semantic'].shape[0], dtype=torch.bool)
    has_labeled_samples = labeled_mask.sum() > 0
    
    # --- 1. 多类语义分割 ---
    if 'semantic' in outputs and 'semantic' in labels and has_labeled_samples:
        logits = outputs['semantic'][labeled_mask]               # (B, C, H, W)
        target = labels['semantic'][labeled_mask]                # (B, H, W) 类别索引, long
        num_classes = logits.size(1)

        # 防御：全 255 的 Batch 跳过 CE 和 Dice
        if (target != 255).sum() > 0:
            ce_loss = F.cross_entropy(logits, target, ignore_index=255)
            dice = dice_loss_multi(logits, target, num_classes)
            loss_sem = ce_loss + dice
            w = weights.get('semantic', 1.0)
            loss_total += w * loss_sem
            loss_dict['semantic_ce'] = ce_loss.item()
            loss_dict['semantic_dice'] = dice.item()

    # --- 2. 边界增强 ---
    if 'boundary' in outputs and 'boundary' in labels and has_labeled_samples:
        pred_logits = outputs['boundary'][labeled_mask]                 # (B, 1, H, W) 未 sigmoid
        target_bnd = labels['boundary'][labeled_mask]                # (B, 1, H, W) 二值
        target_sem = labels['semantic'][labeled_mask]  # 取出对应的语义标签

        # 只有 semantic != 255 的区域才是有效的边界训练区
        valid = (target_sem != 255).unsqueeze(1).float()
        if valid.sum() > 0:
            # 加权 BCE（边界像素少，提高正样本权重）
            pos_weight = torch.tensor([10.0], device=pred_logits.device)  # 可配置
            bce_map = F.binary_cross_entropy_with_logits(
                pred_logits, target_bnd, pos_weight=pos_weight, reduction='none'
                )
            # mean:默认 所有像素平均，返回一个标量
            # sum:所有像素求和，同上
            # none:不归一化，返回每个像素的损失，返回一个 (B, 1, H, W) 的张量
            # 仅在 valid 区域求均值
            bce_loss = (bce_map * valid).sum() / valid.sum()

            w = weights.get('boundary', 0.5)
            loss_total += w * bce_loss
            loss_dict['boundary_bce'] = bce_loss.item()

    # --- 3. 自监督着色 ---
    if 'colorization' in outputs and 'original_rgb' in labels:
        pred = torch.sigmoid(outputs['colorization'])
        target = labels['original_rgb']
        fg_mask = (target.mean(dim=1, keepdim=True) < 0.95).float()
        
        if fg_mask.sum() > 0:
            diff = torch.abs(pred - target) * fg_mask
            l1_loss = diff.sum() / (fg_mask.sum() * 3 + 1e-3)
            w = weights.get('colorization', 0.8)
            loss_total += w * l1_loss
            loss_dict['colorization_l1'] = l1_loss.item()

    if loss_total.item() == 0.0:
        loss_total.requires_grad = True

    return loss_total, loss_dict