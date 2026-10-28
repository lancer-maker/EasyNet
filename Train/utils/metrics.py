"""
评估指标：mIoU，用于验证集监控。
"""
import numpy as np


def compute_miou(pred, target, num_classes, ignore_index=255):
    """
    pred: (B, H, W) 预测类别
    target: (B, H, W) 真实标签
    num_classes: 类别总数（含背景）
    ignore_index: 忽略的标签值（通常为 255 表示未标注区域，注意背景类 0 不会被忽略）
    """
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)

        # 忽略 target 中的 ignore_index 区域
        valid = (target != ignore_index)
        intersection = np.logical_and(pred_cls, target_cls) & valid
        union = np.logical_or(pred_cls, target_cls) & valid

        intersection_sum = intersection.sum()
        union_sum = union.sum()
        if union_sum == 0:
            ious.append(float('nan'))   # 若该类未出现，不计入平均
        else:
            ious.append(intersection_sum / union_sum)

    return np.nanmean(ious)  # 忽略 nan 计算平均