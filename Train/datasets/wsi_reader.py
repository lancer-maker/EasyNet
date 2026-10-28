# TiffSlide 安全多尺度读取、尺寸强约束、越界白底填充
import tiffslide
import numpy as np

class WSIReader:
    """封装 WSI 底层 IO，消灭浮点数除法截断 bug，使用 Alpha 通道处理边缘安全填充"""
    def __init__(self, svs_path):
        self.slide = tiffslide.TiffSlide(str(svs_path))
        self.level_downsamples = self.slide.level_downsamples

    def read_region_safe(self, cx_L0: int, cy_L0: int, level: int, output_size: int = 512) -> np.ndarray:
        """
        以 Level-0 主坐标 (cx_L0, cy_L0) 为物理中心提取图像。
        强制保障输出形状为严格的 (output_size, output_size, 3) float32，范围 [0, 1]。
        """
        downsample = self.level_downsamples[level]
        n = output_size * downsample
        
        # 计算 Level-0 上的左上角起始坐标
        start_x_L0 = int(cx_L0 - n / 2.0)
        start_y_L0 = int(cy_L0 - n / 2.0)

        # 无论坐标是否跨界，直接向底层 API 请求确切的 output_size 像素，规避 511 截断错误
        region = self.slide.read_region((start_x_L0, start_y_L0), level, (output_size, output_size))
        region_np = np.array(region)

        # 处理透明通道 (OpenSlide/TiffSlide 越界区域 A=0)
        if region_np.shape[2] == 4:
            alpha = region_np[:, :, 3]
            rgb = region_np[:, :, :3]
            white_bg = np.full_like(rgb, 255)
            # Alpha > 0 保留组织，Alpha == 0 (超出WSI边界) 填入白色 (255,255,255)
            mask = (alpha > 0)[:, :, np.newaxis]
            output = np.where(mask, rgb, white_bg)
        else:
            output = region_np[:, :, :3]

        return output.astype(np.float32) / 255.0