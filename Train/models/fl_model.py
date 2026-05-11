import torch.nn as nn
import torch
from .base_model import BaseMTLModel
from .backbones import UNetEncoder, UNetDecoder

class FluorescenceModel(BaseMTLModel):
    def __init__(self, in_channels=1, base=64):
        super().__init__()
        self.encoder = UNetEncoder(in_channels=in_channels, base=base)
        self.decoder = UNetDecoder(base=base)

        # 四个独立输出头
        self.head_semantic = nn.Conv2d(in_channels = base, out_channels = 1, kernel_size = 1)   # sigmoid
        self.head_dist = nn.Conv2d(base, 1, 1)       # linear
        self.head_boundary = nn.Conv2d(base, 1, 1)   # sigmoid
        self.head_denoise = nn.Conv2d(base, 1, 1)    # sigmoid

    def forward(self, x):
        # x: (B,1,H,W)
        feats = self.encoder(x)
        dec_out = self.decoder(*feats)   # (B, base, H, W)

        semantic = torch.sigmoid(self.head_semantic(dec_out))
        dist = self.head_dist(dec_out)               # 回归，不加激活
        boundary = torch.sigmoid(self.head_boundary(dec_out))
        denoised = torch.sigmoid(self.head_denoise(dec_out))

        return {
            'semantic': semantic,
            'dist': dist,
            'boundary': boundary,
            'denoised': denoised
        }
# 把模型 forward 的四个输出封装成一个 Python 字典并返回，便于训练/推理时按任务访问和计算损失。