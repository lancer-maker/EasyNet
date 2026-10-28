"""
HE 半监督动态任务调度器
"""
import math


class HEScheduler:
    def __init__(self, initial_weights, progress_thresh=0.5, sigma_ratio=0.4, total_epochs=200):
        self.initial = initial_weights.copy()
        self.progress_thresh = progress_thresh
        self.sigma = sigma_ratio * total_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def step(self, epoch):
        self.current_epoch = epoch

    def get_weights(self):
        progress = self.current_epoch / max(1, self.total_epochs)
        
        # 计算高斯衰减因子
        if progress < self.progress_thresh:
            x = self.current_epoch
            self_training = False
        else:
            # 超过阈值后，将 x 锁定在临界点，保持权重不再跳变
            x = int(self.total_epochs * self.progress_thresh)
            self_training = True
            
        decay = math.exp(- (x ** 2) / (2 * self.sigma ** 2))
        
        weights = {
            'semantic': self.initial['semantic'],
            'boundary': self.initial['boundary'],
            'colorization': self.initial['colorization'] * decay
        }
        return weights, self_training