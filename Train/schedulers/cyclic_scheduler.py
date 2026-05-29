# 通用的自监督任务权重调度器
import math

class CyclicScheduler:
    def __init__(self, num_epochs, progress_thresh=0.5, sigma_ratio=0.3,
                 base_weights=None):
        self.total = num_epochs
        self.thresh = progress_thresh # progress_thresh前为自监督权重衰减阶段，后50%为极低权重阶段
        self.sigma = sigma_ratio * num_epochs
        # 衰减公式为weights = w0 * exp(-0.5 * (progress / sigma)^2)
        self.base = base_weights or {}
        # 如果base_weights存在且非空，则使用它；否则使用空字典
        self.epoch = 0

    def get_task_weights(self, epoch):
        self.epoch = epoch
        progress = epoch / self.total
        weights = {}
        # 基础权重
        for k, v in self.base.items():
            weights[k] = v
        # weights = self.base.copy()

        if progress < self.thresh:
            # 高斯衰减自监督任务权重
            if 'denoising' in weights:
                w0 = self.base['denoising']
                weights['denoising'] = w0 * math.exp(-0.5 * (progress / self.sigma) ** 2)
            if 'colorization' in weights:
                w0 = self.base['colorization']
                weights['colorization'] = w0 * math.exp(-0.5 * (progress / self.sigma) ** 2)
        else:
            # 后50%极低权重
            if 'denoising' in weights:
                weights['denoising'] = 1e-4
            if 'colorization' in weights:
                weights['colorization'] = 1e-4
        return weights

    def should_use_teacher(self, epoch):
        return epoch < self.total * self.thresh

    def is_self_training_phase(self, epoch):
        return epoch >= self.total * self.thresh