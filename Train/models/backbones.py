# U-Net作为主要特征网络
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """基础双层卷积块"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNetEncoder(nn.Module):
    """共享编码器（下采样 3 次）"""
    def __init__(self, in_channels=3, base=64): # base是基础通道数，后续倍增~out_channels=512
        super().__init__()
        self.inc = DoubleConv(in_channels, base)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base, base*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*2, base*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*4, base*8))

    def forward(self, x):
        x1 = self.inc(x)      # (B, base, H, W)
        x2 = self.down1(x1)   # (B, base*2, H/2, W/2)
        x3 = self.down2(x2)   # (B, base*4, H/4, W/4)
        x4 = self.down3(x3)   # (B, base*8, H/8, W/8)
        return x1, x2, x3, x4

class UNetDecoder(nn.Module):
    """通用解码器（上采样 3 次 + skip connections）"""
    def __init__(self, base=64):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec1 = DoubleConv(base*8, base*4)

        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = DoubleConv(base*4, base*2)
        
        self.up3 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec3 = DoubleConv(base*2, base)

    def forward(self, x1, x2, x3, x4):
        d1 = self.dec1(torch.cat([self.up1(x4), x3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), x2], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), x1], dim=1))
        return d3   # (B, base, H, W)