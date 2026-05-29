import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1e-5):
    """Dice 损失，pred 和 target 均为 (B,1,H,W)"""
    pred = pred.contiguous().view(-1)
    # contiguous()-确保张量连续，view(-1)展平为一维向量
    # pred.reshape(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def fluor_losses(outputs, labels, weights):
    """
    outputs: 模型返回的字典
    labels: dict 包含 'mask', 'dist', 'boundary', 'clean' 等
    weights: 当前动态权重 {'semantic': w1, 'dist': w2, 'boundary': w3, 'denoising': w4}
    """
    loss_total = 0.0
    loss_dict = {}

    if 'semantic' in outputs and 'mask' in labels:
        pred = outputs['semantic']
        target = labels['mask']
        bce = F.binary_cross_entropy(pred, target)
        dice = dice_loss(pred, target)
        loss_sem = bce + dice
        loss_total += weights['semantic'] * loss_sem
        loss_dict['semantic'] = loss_sem.item()

    if 'dist' in outputs and 'dist' in labels:
        pred = outputs['dist']
        target = labels['dist']
        # 只计算有细胞区域的损失
        mask = labels['mask'].bool()
        loss_dist = F.mse_loss(pred[mask], target[mask])
        loss_total += weights['dist'] * loss_dist
        loss_dict['dist'] = loss_dist.item()

    if 'boundary' in outputs and 'boundary' in labels:
        pred = outputs['boundary']
        target = labels['boundary']
        # 边界像素少，加权
        pos_weight = torch.tensor(10.0, device=pred.device)
        loss_bnd = F.binary_cross_entropy(pred, target, pos_weight=pos_weight)
        loss_total += weights['boundary'] * loss_bnd
        loss_dict['boundary'] = loss_bnd.item()

    if 'denoised' in outputs and 'clean' in labels:
        pred = outputs['denoised']
        target = labels['clean']
        loss_den = F.l1_loss(pred, target)
        loss_total += weights['denoising'] * loss_den
        loss_dict['denoising'] = loss_den.item()

    return loss_total, loss_dict