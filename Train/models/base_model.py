# 接口，强制继承的模型实现forward逻辑
import torch.nn as nn

class BaseMTLModel(nn.Module):
    """不实现 forward，仅声明接口，子类必须重写"""
    def __init__(self):
        super().__init__()
    def forward(self, *args, **kwargs):
        raise NotImplementedError