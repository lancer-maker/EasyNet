"""
伪标签生成器：基于 Teacher-Student 交叉置信度过滤，将无标签数据转为有监督信号。
"""
import torch
import torch.nn.functional as F
from datasets.mask_utils import generate_boundary_from_semantic


class PseudoLabeler:
    """
    使用方法：
        labeler = PseudoLabeler(conf_thresh=0.9, teacher_weak_thresh=0.4)
        pseudo_dict = labeler(teacher_out, student_out)
        # pseudo_dict['semantic'] (B, H, W)  long, 忽略区域=255
        # pseudo_dict['boundary'] (B, 1, H, W) float
    """

    def __init__(self, conf_thresh=0.9, teacher_weak_thresh=0.4):
        self.conf_thresh = conf_thresh
        self.teacher_weak_thresh = teacher_weak_thresh

    @torch.no_grad()
    def __call__(self, teacher_outputs, student_outputs):
        """
        teacher_outputs: dict with 'semantic' logits (B, C, H, W)
        student_outputs: dict with 'semantic' logits (B, C, H, W)
        返回:
            dict: {'semantic': pseudo_label (B,H,W) long, 'boundary': (B,1,H,W) float}
        """
        # 概率化
        t_prob = F.softmax(teacher_outputs['semantic'], dim=1)  # (B, C, H, W)
        s_prob = F.softmax(student_outputs['semantic'], dim=1)

        # 每像素最大置信度与对应类别
        s_conf, s_pred = s_prob.max(dim=1)       # (B, H, W) s_conf-每像素最高预测概率，s_pred-每像素预测类别索引
        t_conf, t_pred = t_prob.max(dim=1)

        # 1. 默认采用 Teacher
        pseudo_semantic = t_pred.clone()

        # 2. 交叉置信度条件覆盖
        overwrite_mask = (s_conf > self.conf_thresh) & (t_conf < self.teacher_weak_thresh)
        pseudo_semantic = torch.where(overwrite_mask, s_pred, pseudo_semantic)

        # 3. 抑制背景：凡是预测为 0 (Background) 的，一律设为 255 (Ignore)
        pseudo_semantic[pseudo_semantic == 0] = 255

        # 4. 边界从伪语义生成(暂用 CPU，日后瓶颈再优化)
        pseudo_boundary = torch.empty(
            (pseudo_semantic.size(0), 1, *pseudo_semantic.shape[1:]),
            dtype=torch.float32, 
            device=pseudo_semantic.device
        )
        for i in range(pseudo_semantic.size(0)):
            # 转回 numpy 处理（边界生成在 CPU 更简单）
            sem_np = pseudo_semantic[i].cpu().numpy()
            bound_np = generate_boundary_from_semantic(sem_np)
            pseudo_boundary[i, 0] = torch.from_numpy(bound_np).float().to(pseudo_semantic.device)

        return {'semantic': pseudo_semantic, 'boundary': pseudo_boundary}