'''
HE 部分已验证
'''
# 数据增强与预处理（全 numpy float32 [0,1] 域操作）
# 输入约定：
#   - 图像：float32 (H,W,3) 或 (H,W) ，值域 [0,1]
#   - 掩码：uint8 或 int (H,W)
# 所有变换保证空间一致性，并自动裁剪/填充到固定尺寸。


import random
import numpy as np
import cv2


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *arrays):
        for t in self.transforms:
            arrays = t(*arrays)
        return arrays


class HEGeometricTransform:
    """几何增强：旋转、翻转、缩放裁剪（保持 float32）"""
    def __init__(self, size=512, scale=(0.8, 1.2), rotation=15, p_flip=0.5):
        self.size = size
        self.scale = scale
        self.rotation = rotation
        self.p_flip = p_flip

    def __call__(self, img, mask_sem=None, mask_boundary=None):
        h, w = img.shape[:2]

        # 随机缩放
        scale = random.uniform(*self.scale)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if mask_sem is not None:
            mask_sem = cv2.resize(mask_sem, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        if mask_boundary is not None:
            mask_boundary = cv2.resize(mask_boundary, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # 随机裁剪或填充到固定尺寸
        img, mask_sem, mask_boundary = self._crop_or_pad(img, mask_sem, mask_boundary, self.size)

        # 随机旋转
        angle = random.uniform(-self.rotation, self.rotation)
        img = self._rotate_image(img, angle, cv2.INTER_LINEAR)
        if mask_sem is not None:
            mask_sem = self._rotate_image(mask_sem, angle, cv2.INTER_NEAREST)
        if mask_boundary is not None:
            mask_boundary = self._rotate_image(mask_boundary, angle, cv2.INTER_NEAREST)

        # 随机水平翻转
        if random.random() < self.p_flip:
            img = np.ascontiguousarray(img[:, ::-1])
            if mask_sem is not None:
                mask_sem = np.ascontiguousarray(mask_sem[:, ::-1])
            if mask_boundary is not None:
                mask_boundary = np.ascontiguousarray(mask_boundary[:, ::-1])

        return img, mask_sem, mask_boundary

    def _crop_or_pad(self, img, mask_sem, mask_boundary, size):
        h, w = img.shape[:2]
        # 如果小于目标尺寸，对称填充 0
        if h < size or w < size:
            pad_h = max(0, size - h)
            pad_w = max(0, size - w)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w,
                                     borderType=cv2.BORDER_CONSTANT, value=0.0)
            if mask_sem is not None:
                mask_sem = cv2.copyMakeBorder(mask_sem, 0, pad_h, 0, pad_w,
                                              borderType=cv2.BORDER_CONSTANT, value=0)
            if mask_boundary is not None:
                mask_boundary = cv2.copyMakeBorder(mask_boundary, 0, pad_h, 0, pad_w,
                                                   borderType=cv2.BORDER_CONSTANT, value=0)
            h, w = img.shape[:2]

        # 随机裁剪
        y = random.randint(0, max(0, h - size))
        x = random.randint(0, max(0, w - size))
        img = img[y:y+size, x:x+size]
        if mask_sem is not None:
            mask_sem = mask_sem[y:y+size, x:x+size]
        if mask_boundary is not None:
            mask_boundary = mask_boundary[y:y+size, x:x+size]
        return img, mask_sem, mask_boundary

    @staticmethod
    def _rotate_image(img, angle, interp):
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)         # 计算以图像中心为旋转中心、旋转 angle 度、不缩放的仿射变换矩阵。
        return cv2.warpAffine(img, M, (w, h), flags=interp,
                              borderMode=cv2.BORDER_REFLECT_101)    # 仿射变换，输出尺寸与原图一致 (w, h)，插值方法由参数 interp 指定，边界填充模式为 cv2.BORDER_REFLECT_101（反射填充，避免边界伪影）


class HEColorAugmentation:
    """
    极致颜色增强，仅作用于分支A的 RGB 图 (H,W,3) float32 [0,1]
    """
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p_blur=0.2):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p_blur = p_blur            # config中有对应数值，在train中再调用config，保持代码简洁

    def __call__(self, img):
        img = img.copy()
        # 亮度
        factor = 1.0 + random.uniform(-self.brightness, self.brightness)
        img = np.clip(img * factor, 0, 1)

        # 对比度
        factor = 1.0 + random.uniform(-self.contrast, self.contrast)
        gray = np.mean(img, axis=-1, keepdims=True)
        img = np.clip(gray + factor * (img - gray), 0, 1)

        # 饱和度
        factor = 1.0 + random.uniform(-self.saturation, self.saturation)
        gray = np.mean(img, axis=-1, keepdims=True)
        img = np.clip(gray + factor * (img - gray), 0, 1)

        # 色调（在 RGB 上近似旋转）
        factor = random.uniform(-self.hue, self.hue)
        img = self._hue_rotate(img, factor)

        # 高斯模糊（直接对 float32 图像）
        if random.random() < self.p_blur:
            ksize = random.choice([3, 5])
            sigma = random.uniform(0.5, 1.5)
            img = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

        return np.clip(img, 0, 1)

    @staticmethod
    def _hue_rotate(img, delta):
        """在 HSV 空间旋转色调（输入输出 float32 [0,1]）"""
        # 转为 uint8 供 OpenCV 处理
        img_u8 = (img * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + delta * 180) % 180
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
        return rgb


class HETrainTransform:
    """
    HE 训练完整变换（内部保持 float32 [0,1]）
    返回：
        image_aug:     (H,W,3) float32 [0,1] 分支A
        image_gray:    (H,W) float32 [0,1]   分支B
        original_rgb:  (H,W,3) float32 [0,1] 着色目标
        mask_sem:      (H,W) uint8 (可选)
        mask_boundary: (H,W) uint8 (可选)
    """
    def __init__(self, size=512, color_jitter_params=None):
        self.geo_transform = HEGeometricTransform(size=size)
        self.color_aug = HEColorAugmentation(**(color_jitter_params or {}))

    def __call__(self, img, mask_sem=None, mask_boundary=None):
        # 1. 几何增强（统一变换）
        img, mask_sem, mask_boundary = self.geo_transform(img, mask_sem, mask_boundary)

        # 2. 保留原始色彩
        original_rgb = img.copy()

        # 3. 分支A：颜色增强
        image_aug = self.color_aug(img)

        # 4. 分支B：灰度（直接浮点计算，无量化损失）
        image_gray = 0.299 * original_rgb[:, :, 0] + 0.587 * original_rgb[:, :, 1] + 0.114 * original_rgb[:, :, 2]

        return image_aug, image_gray, original_rgb, mask_sem, mask_boundary


class HEValTransform:
    """HE 验证变换（无增强，仅统一尺寸）"""
    def __init__(self, size=512):
        self.size = size

    def __call__(self, img, mask_sem=None, mask_boundary=None):
        h, w = img.shape[:2]
        scale = self.size / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if mask_sem is not None:
            mask_sem = cv2.resize(mask_sem, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        if mask_boundary is not None:
            mask_boundary = cv2.resize(mask_boundary, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # 中心裁剪
        y = (new_h - self.size) // 2
        x = (new_w - self.size) // 2
        img = img[y:y+self.size, x:x+self.size]
        if mask_sem is not None:
            mask_sem = mask_sem[y:y+self.size, x:x+self.size]
        if mask_boundary is not None:
            mask_boundary = mask_boundary[y:y+self.size, x:x+self.size]

        image_gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        return img, image_gray, img.copy(), mask_sem, mask_boundary


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


# 边界掩码生成（语义->边界）
def generate_boundary_from_semantic(semantic_mask, dilation_radius=2):
    """
    semantic_mask: (H,W) uint8 或 int，0=背景
    returns: (H,W) uint8 (0/1)
    """
    foreground = (semantic_mask > 0).astype(np.uint8)
    if foreground.sum() == 0:
        return foreground
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (dilation_radius * 2 + 1, dilation_radius * 2 + 1))
    dilated = cv2.dilate(foreground, kernel, iterations=1)
    eroded = cv2.erode(foreground, kernel, iterations=1)
    return dilated - eroded