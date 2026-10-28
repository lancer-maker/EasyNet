# 数据增强与预处理（全 numpy float32 [0,1] 域操作）
# 输入约定：
#   - 图像：float32 (H,W,3) 或 (H,W) ，值域 [0,1]
#   - 掩码：uint8 或 int (H,W)
# 所有变换保证空间一致性，并自动裁剪/填充到固定尺寸。
# HE 双尺度 + 荧光多任务 统一数据增强与预处理
import random
import numpy as np
import cv2
from .mask_utils import generate_boundary_from_semantic

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *arrays):
        for t in self.transforms:
            arrays = t(*arrays)
        return arrays
    
# =====================================================================
# 二、 H&E 双尺度特化增强 (Dual-Scale + Dual-Branch)
# =====================================================================
class HEGeometricTransform:
    """
    双尺度空间严格同步增强：
    对 Local (512x512) 与 Global (512x512) 施加完全相同的正交几何拓扑变换，
    不使用任意比例缩放裁剪，确保物理对应点在像素矩阵中绝对重合。
    """
    def __init__(self, p_flip: float = 0.5, p_rot90: float = 0.5):
        self.p_rot90 = p_rot90
        self.p_flip = p_flip

    def __call__(self, local_img, global_img, mask_sem=None, mask_boundary=None):
        # 1.水平镜像
        if random.random() < self.p_flip:
            local_img = np.ascontiguousarray(local_img[:, ::-1])
            global_img = np.ascontiguousarray(global_img[:, ::-1])
            if mask_sem is not None:
                mask_sem = np.ascontiguousarray(mask_sem[:, ::-1])
            if mask_boundary is not None:
                mask_boundary = np.ascontiguousarray(mask_boundary[:, ::-1])

        # 2.垂直镜像
        if random.random() < self.p_flip:
            local_img = np.ascontiguousarray(local_img[::-1, :])
            global_img = np.ascontiguousarray(global_img[::-1, :])
            if mask_sem is not None:
                mask_sem = np.ascontiguousarray(mask_sem[::-1, :])
            if mask_boundary is not None:
                mask_boundary = np.ascontiguousarray(mask_boundary[::-1, :])

        # 3.90度正交旋转
        if random.random() < self.p_rot90:
            k = random.choice([1, 2, 3])  # 随机选择逆时针旋转90(1)、180(2)、270(3)度
            local_img = np.ascontiguousarray(np.rot90(local_img, k))
            global_img = np.ascontiguousarray(np.rot90(global_img, k))
            if mask_sem is not None:
                mask_sem = np.ascontiguousarray(np.rot90(mask_sem, k))
            if mask_boundary is not None:
                mask_boundary = np.ascontiguousarray(np.rot90(mask_boundary, k))

        return local_img, global_img, mask_sem, mask_boundary
    
class HEColorAugmentation:
    """
    仅施加于 Local RGB - RGB分支的极致颜色增强, (H,W,3) float32 [0,1]
    基于标准 HSV 色彩空间
    """
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08, p_blur=0.2):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p_blur = p_blur            # config中有对应数值，在train中再调用config，保持代码简洁

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img_u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV).astype(np.float32)

        # 色调 (H) 与饱和度 (S) 扰动
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-self.hue, self.hue) * 180.0) % 180.0
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + random.uniform(-self.saturation, self.saturation)), 0.0, 255.0)

        # 明度 (V) 与对比度扰动
        v = hsv[:, :, 2] / 255.0
        v = np.clip(v * (1.0 + random.uniform(-self.brightness, self.brightness)), 0.0, 1.0)
        mean_v = np.mean(v)
        v = np.clip(mean_v + (1.0 + random.uniform(-self.contrast, self.contrast)) * (v - mean_v), 0.0, 1.0)
        hsv[:, :, 2] = v * 255.0

        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

        if random.random() < self.p_blur:
            ksize = random.choice([3, 5])
            rgb = cv2.GaussianBlur(rgb, (ksize, ksize), sigmaX=1.0)

        return np.clip(rgb, 0.0, 1.0)


class HETrainTransform:
    """
    HE 双尺度训练组合变换器
    接收: local_img (512,512,3), global_img (512,512,3), mask_sem, mask_boundary
    返回:
        local_branch_a:     Local 强色彩增强 RGB (用于结构语义学习)
        local_branch_b_gray:Local 纯灰度图 (用于自监督着色任务输入)
        orig_local_rgb:     Local 原始 RGB (自监督着色 Ground Truth)
        global_branch:      Global 原始 RGB (宏观上下文输入)
        mask_sem:           语义掩码
        mask_boundary:      边界掩码
    """
    def __init__(self, color_jitter_params=None):
        self.geo_transform = HEGeometricTransform()
        self.color_aug = HEColorAugmentation(**(color_jitter_params or {}))

    def __call__(self, local_img, global_img, mask_sem=None, mask_boundary=None):
        # 1. 双尺度空间同屏几何变换
        local_img, global_img, mask_sem, mask_boundary = self.geo_transform(
            local_img, global_img, mask_sem, mask_boundary
            )
        # 2. Local 双分支生成
        orig_local_rgb = local_img.copy()
        local_branch_a = self.color_aug(local_img)
        local_branch_b_gray = (
            0.299 * local_img[:, :, 0] + 
            0.587 * local_img[:, :, 1] + 
            0.114 * local_img[:, :, 2]
        )# 三色通道转换灰度公式

        # 3. Global 分支保持真实物理采样色彩
        global_branch = global_img.copy()

        return local_branch_a, local_branch_b_gray, orig_local_rgb, global_branch, mask_sem, mask_boundary

class HEValTransform:
    """HE 双尺度验证变换器（无随机增强，仅做数据类型装配）"""
    def __call__(self, local_img, global_img, mask_sem=None, mask_boundary=None):
        orig_local_rgb = local_img.copy()
        local_branch_b_gray = (
            0.299 * orig_local_rgb[:, :, 0] +
            0.587 * orig_local_rgb[:, :, 1] +
            0.114 * orig_local_rgb[:, :, 2]
        )
        return local_img, local_branch_b_gray, orig_local_rgb, global_img, mask_sem, mask_boundary

class FluorescenceTransform:
    """
    荧光增强：噪声、亮度/对比度，输入 (H,W) 或 (H,W,C) float32 [0,1]
    返回 tensor 形状 (C,H,W)  float32
    """
    def __init__(self, size=512, noise_std=0.05, brightness=0.2, contrast=0.2):
        self.size = size
        self.noise_std = noise_std
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img):
        # 统一尺寸
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

        # 确保至少二维 (H,W) → (H,W,1)
        if img.ndim == 2:
            img = img[..., np.newaxis]

        # 亮度/对比度
        if random.random() < 0.8:
            alpha = 1.0 + random.uniform(-self.contrast, self.contrast)
            beta = random.uniform(-self.brightness, self.brightness)
            img = np.clip(alpha * img + beta, 0, 1)

        # 高斯噪声
        if random.random() < 0.8:
            noise = np.random.randn(*img.shape).astype(np.float32) * self.noise_std
            img = np.clip(img + noise, 0, 1)

        return img.transpose(2, 0, 1)   # (C, H, W)

