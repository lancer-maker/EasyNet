import torch
import torch.nn as nn
from .base_model import BaseMTLModel
from .backbones import UNetEncoder, UNetDecoder

class HEModel(BaseMTLModel):
    def __init__(self, he_classes=6, base=64):
        super().__init__()
        # 共享编码器（三通道，灰度分支将1通道复制为3）
        self.encoder = UNetEncoder(in_channels=3, base=base)

        # 分支 A 解码器（语义+边界）
        self.decoder_sem = UNetDecoder(base=base)
        self.head_semantic = nn.Conv2d(base, he_classes, 1)   # 多类，不加softmax（损失内含）
        self.head_boundary = nn.Conv2d(base, 1, 1)            # sigmoid

        # 分支 B 解码器（着色）
        self.decoder_color = UNetDecoder(base=base)
        self.head_color = nn.Conv2d(base, 3, 1)               # 输出 RGB

    def forward(self, branch_a, branch_b_gray):
        """
        branch_a: (B, 3, H, W)  经过ColorJitter的RGB
        branch_b_gray: (B, 1, H, W) 灰度图
        """
        B = branch_a.size(0)
        # 灰度分支复制为3通道，与branch_a在batch维度拼接
        branch_b_3ch = branch_b_gray.repeat(1, 3, 1, 1)
        # 多分支特征融合：在并行网络（如双路输入、多尺度特征）中，将不同分支的输出沿 batch 维度堆叠，然后一起送入后续层。
        combined = torch.cat([branch_a, branch_b_3ch], dim=0)  # (2B, 3, H, W)

        # 共享编码器一次前向
        x1, x2, x3, x4 = self.encoder(combined)
        # 一次性送入后，前B个样本来自 branch_a，后B个样本来自 branch_b_gray。后续解码器根据索引分开处理。

        # 拆分特征（各层都需要拆）
        # 接受一个张量f（2B, C, H, W），返回前B个样本和后B个样本
        def split_feat(f):
            return f[:B], f[B:]
        x1_a, x1_b = split_feat(x1)
        x2_a, x2_b = split_feat(x2)
        x3_a, x3_b = split_feat(x3)
        x4_a, x4_b = split_feat(x4)

        # 分支 A 上采样 + 输出
        feats_a = (x1_a, x2_a, x3_a, x4_a)
        dec_a = self.decoder_sem(*feats_a)
        semantic = self.head_semantic(dec_a)          # (B, C, H, W)
        boundary = torch.sigmoid(self.head_boundary(dec_a))

        # 分支 B 上采样 + 输出
        feats_b = (x1_b, x2_b, x3_b, x4_b)
        dec_b = self.decoder_color(*feats_b)
        color_out = self.head_color(dec_b)  # 原始logits，损失内含激活

        # 统一数据交换接口
        return {
            'semantic': semantic,       # logits (B, C, H, W)
            'boundary': boundary,       # logits (B, 1, H, W)
            'colorization': color_out   # sigmoid激活后的数值(B, 3, H, W)，值域 [0,1]
        }