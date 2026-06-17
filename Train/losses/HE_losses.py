import torch
import torch.nn.functional as F

def dice_loss_multi(pred, target, num_classes, smooth=1e-5, ignore_index=255):
    """
    pred: (B, C, H, W) logits
    target: (B, H, W) class indices
    num_classes: 类别数
    """
    pred = F.softmax(pred, dim=1)  # (B, C, H, W),沿一维归一化
    # 转成 one-hot 形式方便计算
    target_one_hot = F.one_hot(target, num_classes).permute(0, 3, 1, 2).float()  # (B, C, H, W)
    # one_hot(target, num_classes)-->target(B,H,W)-->target_one_hot(B,H,W,C/num_classes)-->permute(B,C,H,W)
    if ignore_index is not None:
        '''
        生成掩码，忽略指定类别的像素（如背景），不参与损失计算。
        '''
        mask = (target != ignore_index).unsqueeze(1).float()  # (B,1,H,W),mask中为1的像素参与计算，为0的像素被忽略
        pred = pred * mask                                    # 语义分割输出
        target_one_hot = target_one_hot * mask

    dice = 0.0
    for c in range(num_classes):
        p_c = pred[:, c, :, :].contiguous().view(-1)           # c类别的预测概率
        t_c = target_one_hot[:, c, :, :].contiguous().view(-1) # c类别的真实标签（0或1）
        intersection = (p_c * t_c).sum()                       # 只保留真实属于第 c 类的像素的预测概率，其他像素归零

        union = p_c.sum() + t_c.sum()                          # 
        if union > 0:
            dice += 1 - (2. * intersection + smooth) / (union + smooth)
        '''
        假阳性：真实标签中没有第 c 类（t_c=0），但模型预测了第 c 类（p_c>0），dice=1,此时将迫使模型降虚假的预测概率
        '''
    return dice / num_classes


def he_losses(outputs, labels, weights):
    """
    outputs: dict with 'semantic', 'boundary', 'colorization'
    labels: dict with 'semantic' (B,H,W class indices),
                       'boundary' (B,1,H,W binary),
                       'original_rgb' (B,3,H,W [0,1])
    weights: dict with keys 'semantic', 'boundary', 'colorization'
    """
    loss_total = 0.0
    loss_dict = {}

    # --- 1. 多类语义分割 ---
    if 'semantic' in outputs and 'semantic' in labels:
        logits = outputs['semantic']               # (B, C, H, W)
        target = labels['semantic']                # (B, H, W) 类别索引, long
        num_classes = logits.size(1)

        # 交叉熵
        ce_loss = F.cross_entropy(logits, target, ignore_index=255)  # 可配置忽略背景外的其他类
        # Dice
        dice = dice_loss_multi(logits, target, num_classes)
        loss_sem = ce_loss + dice

        w = weights.get('semantic', 1.0)
        loss_total = loss_total + w * loss_sem
        loss_dict['semantic_ce'] = ce_loss.item()
        loss_dict['semantic_dice'] = dice.item()

    # --- 2. 边界增强 ---
    if 'boundary' in outputs and 'boundary' in labels:
        pred_logits = outputs['boundary']                 # (B, 1, H, W) 未 sigmoid
        target = labels['boundary']                # (B, 1, H, W) 二值

        # 加权 BCE（边界像素少，提高正样本权重）
        pos_weight = torch.tensor([10.0], device=pred_logits.device)  # 可配置
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_logits, target, pos_weight=pos_weight
            )

        w = weights.get('boundary', 0.5)
        loss_total = loss_total + w * bce_loss
        loss_dict['boundary_bce'] = bce_loss.item()

    # --- 3. 自监督着色 ---
    if 'colorization' in outputs and 'original_rgb' in labels:
        pred = outputs['colorization']             # (B, 3, H, W) [0,1]
        target = labels['original_rgb']            # (B, 3, H, W) [0,1]

        # 只在组织区域计算损失，避免背景白底占比过大
        # 组织区域可来自语义标签（排除背景类）或目标的边界掩码等。
        # 这里使用颜色目标中非白色的区域粗略当作前景：
        # 如果 R,G,B 都接近 1，则是白底，不计入损失
        if 'semantic' in labels:
            # 更准确：用语义标签的前景掩码
            fg_mask = (labels['semantic'] != 0) & (labels['semantic'] != 255)  # 假设背景类为0
            fg_mask = fg_mask.unsqueeze(1).float()  # (B,1,H,W)
        else:
            # 回退：用 RGB 的平均值判断非白区
            fg_mask = (target.mean(dim=1, keepdim=True) < 0.95).float() # 预处理时>235时跳过

        diff = torch.abs(pred - target) * fg_mask
        l1_loss = diff.sum() / (fg_mask.sum() * 3 + 1e-6)  # 归一化

        w = weights.get('colorization', 0.8)
        loss_total = loss_total + w * l1_loss
        loss_dict['colorization_l1'] = l1_loss.item()

    return loss_total, loss_dict