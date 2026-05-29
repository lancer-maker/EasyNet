# 动态识别模态，归一化，滑动切块，空白跳过
# 文件位置：project_root/deploy/preprocess.py
import numpy as np
import tifffile
from typing import Tuple, Iterator
import warnings
from loadwis import VirtualWIS
import cv2



def lazy_load_wsi(filepath: str) -> np.ndarray:
    try:
        """防 OOM 惰性加载"""
        print(f"[*] tifffile.memmap 惰性映射加载: {filepath}")
        return tifffile.memmap(filepath)
# 单通道：二维数组（H，W）
# 多通道：三维数组（H，W，C），C=3 或 4（可能包含 alpha 通道）
    
    except ValueError:
        print(f'[!] tifffile.memmap 加载失败，尝试 TiffFile对象读取模式: {filepath}')
        tif = tifffile.TiffFile(filepath)
        page = tif.pages[0]
        print(f'[*] 图像维度: {page.shape}, 数据类型: {page.dtype}, 通道数: {page.samplesperpixel}')
        return VirtualWIS(tif)
    
    except (tifffile.TiffFileError, KeyError) as e:
        print(f"[!] 不是标准 TIFF 文件: {e}")
        print(f"[*] 回退到 OpenCV 加载: {filepath}")
        
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)  
        if img is None:
            raise FileNotFoundError(f"OpenCV 无法读取文件: {filepath}")
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f'[*] 图像维度: {img.shape}, 数据类型: {img.dtype}, 通道数: {img.shape[2]}')
        return img
# 在 lazy_load_wsi 返回 VirtualWSI 之后，手动测试



def detect_modality(image_mmap: np.ndarray) -> str:
    # TODO: 依据通道数与直方图特征判断 "fluorescence" 或 "he"
    print(f'[-]自动检测模态')
    eight_bit_threshold = 210
    sixteen_bit_threshold = 5400
    correlation_threshold = 0.65
    ration_bright_threshold = 0.05
    '''
    规则1:通道数
    如果数据的元素是2,直接输出 "fluorescence"。
    规则2:直方图特征,用于判断c>=3
    '''
    try:
        if image_mmap.ndim == 2 or (image_mmap.ndim == 3 and image_mmap.shape[-1] == 1 ):
            print(f'[+]判定: FLUORESCENCE (灰度)')
            return "fluorescence"
        if image_mmap.ndim == 3 and image_mmap.shape[-1] >= 3:
            # 多通道分析
            '''
            全局极大步长降采样,提取1000x1000缩略图,只取前三个通道（丢弃透明通道）
            并将图像展平为(N, 3)便于统计
            聚类算法通常要求输入为 (样本数, 特征/通道数) 的二维数组
            '''
            h, w = image_mmap.shape[0], image_mmap.shape[1] # image_mmap.shape[:2]
            step_y, step_x = max(h // 1000, 1), max(w // 1000, 1)
            thumbnail = image_mmap[::step_y, ::step_x, :3].astype(np.float32)
            flat = thumbnail.reshape(-1, 3)# 将（H, W, 3）展平为（N, 3）

            off_diag_mean = 0.0
            '''
            通道相关性与白底判定做双重验证,增加防NaN崩溃机制
            1.通道相关性判定：方差极小（纯色）--报错
            '''
            if np.all(np.std(flat, axis = 0) < 1e-3):# 从axis0开始，判断每个通道的方差是否都小于1e-3
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore") # 防止浮点噪声忽略错误
                    corr_matrix = np.corrcoef(flat, rowvar=False)
                    # np.corrcoef计算皮尔逊相关系数矩阵--用于衡量多个变量之间的线性相关度
                    # rowvar=False表示每列代表一个变量（通道），每行代表一个样本/观测值（像素点）--RGB三个通道各自采集1000x1000缩略图的像素点作为样本
                    # rowvar=True表示每行代表一个变量（通道），每列代表一个样本/观测值（像素点）
                    if not np.isnan(corr_matrix).any(): # 如果相关矩阵中任何都不是NaN才计算
                        # 计算方式：取三个非对角元素计算其平均值作为相关性指标
                        off_diag_mean = (
                            corr_matrix[0, 1] + corr_matrix[0, 2] + corr_matrix[1, 2]
                        )/3.0
            '''
            2.白底判定
            '''
            intensity = np.mean(flat, axis = -1) # 计算每个通道的平均强度
            max_val = np.max(intensity) # 取三个通道平均强度的最大值作为整体亮度指标
            bright_threshold = eight_bit_threshold if max_val <= 255 else sixteen_bit_threshold # 亮度阈值设定，适应8位和16位图像
            # 计算像素比例
            ratio_bright = np.sum(
                intensity > bright_threshold
            ) / (intensity.size + 1e-7) # 1e-7:防止除零错误
            # np.sum()内部是判断，也就是布尔值。True=1
            print(f'[-]通道相关性指标: {off_diag_mean:.4f}, 白底像素比例: {ratio_bright:.4f}')

            '''
            3.综合决策
            '''
            if off_diag_mean > correlation_threshold or ratio_bright > ration_bright_threshold:
                print("[+] 判定结果: H&E (符合多通道高相关或明场特征)")
                return "he"
            else:
                print("[+] 判定结果: FLUORESCENCE (符合通道独立或暗场特征)")
                return "fluorescence"
    except Exception as e:
        print(f'[!] 模态检测异常: {e}, 请检查后重新尝试')

def normalize_fluorescence(patch: np.ndarray) -> np.ndarray:
    """
    荧光:最后输出为单通道的浮点数组，值域 [0, 1]
    1.方差最大通道提取
    2.百分位截断归一化
    """
    if patch.ndim == 3 and patch.shape[-1] in [3, 4]:
        variances = np.var(patch, axis=(0, 1))
        patch = patch[:, :, np.argmax(variances)]
        
    p_low, p_high = np.percentile(patch, (1.0, 99.0))
    if p_high - p_low < 1e-5:
        return np.zeros_like(patch, dtype=np.float32)
        
    patch_norm = np.clip(patch, p_low, p_high) # 数据压缩到p_low, p_high
    return ((patch_norm - p_low) / (p_high - p_low)).astype(np.float32)

def normalize_he(patch: np.ndarray) -> np.ndarray:
    """
    HE:除以255用以归一化
    """
    if patch.shape[-1] == 4: # 如果有alpha通道，丢弃
        patch = patch[:, :, :3]
    patch_norm = patch / 255.0
    return patch_norm.astype(np.float32)

class PatchExtractor:
    def __init__(
            self, 
            patch_size: int = 512, 
            stride: int = 256,
            coarse_stride: int = 1024, 
            single_margin:int = 2048
            ):
        self.patch_size = patch_size
        self.stride = stride
        self.coarse_stride = coarse_stride
        self.single_margin = single_margin

    def _estimate_background(self, image_mmap: np.ndarray, modality) -> Iterator[Tuple[int, int, np.ndarray, bool]]:
        # Iterator[Tuple[int, int, np.ndarray, bool]]: 返回一个迭代器，每次迭代返回一个元组，元组包含四个元素：y坐标，x坐标，patch图像，是否为空patch(True表示空白)
        """背景估计"""

        h, w = image_mmap.shape[:2]
        sample_size = min(256, h, w)
        corners = [
            (0, 0),
            (0, w - sample_size),
            (h - sample_size, 0),
            (h - sample_size, w - sample_size),
        ]
        if modality == 'fluorescence':
            means, vars_ = [], []
            for y, x in corners:
                region = image_mmap[y:y+sample_size, x:x+sample_size].astype(np.float32)
                means.append(np.mean(region))
                vars_.append(np.var(region))
            self.bg_mean = np.median(means)
            self.bg_std = np.sqrt(np.median(vars_))
            print(f"[*] 荧光背景估计: 均值={self.bg_mean:.1f}, 标准差={self.bg_std:.1f}")
        elif modality == 'he':
            self.bg_mean, self.bg_std = 255.0, 0.0 # HE默认白底背景统计
            print(f"[*] HE 模态不使用背景估计，直接推理")
        else:
            print(f"[!] 未知模态: {modality}")

    def _is_background(self, patch: np.ndarray, modality) -> bool:
        """基于背景统计的自适应空白判定"""
        if patch.size == 0:
            return True
        # 降采样加速
        sub = patch[::8, ::8].astype(np.float32)  # 降采样加速
        mean_val = np.mean(sub)
        var_val = np.var(sub)
        # 背景条件：均值接近背景均值，且方差极低
        if modality == 'fluorescence':
            # 荧光：低均值+低方差
            return (mean_val < self.bg_mean + 2 * self.bg_std) and (var_val < 10.0)

    def _find_first_signal_block(self, image_mmap: np.ndarray, modality: str) -> tuple:
        """
        基于先验，从中心向外粗搜索，返回第一个有效信号块的坐标 (y, x)。
        若未找到返回 None。
        """
        h, w = image_mmap.shape[:2]
        cy, cx = h // 2, w // 2
        # 搜索顺序：中心，然后上下左右四个方向
        offsets = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
                   (-1, -1), (-1, 1), (1, -1), (1, 1)]
        max_attempts = 20  # 最多搜索20个粗块，足够覆盖常用视场
        for dy, dx in offsets:
            y_start = cy + dy * self.coarse_stride - self.patch_size // 2
            x_start = cx + dx * self.coarse_stride - self.patch_size // 2
            y_start = max(0, min(y_start, h - self.patch_size))
            x_start = max(0, min(x_start, w - self.patch_size))
            block = image_mmap[y_start:y_start+self.patch_size, x_start:x_start+self.patch_size]
            
            if not self._is_background(block, modality):
                print(f"[+] 在 ({y_start}, {x_start}) 找到第一个信号块")
                return (y_start, x_start)
            max_attempts -= 1
            if max_attempts <= 0:
                break
        print("[!] 未找到任何信号块，将回退到全图遍历")
        return None

    def _find_roi_by_growth(self, image_mmap, sig_block_y, sig_block_x, modality: str):
        """
        从信号块中心向四个方向扫描，估计组织实际边界，返回 (y_min, y_max, x_min, x_max)。
        若任意方向扫描失败，回退到全图。
        """
        h, w = image_mmap.shape[:2]
        ps = self.patch_size
        half = ps // 2
        center_y = sig_block_y + half
        center_x = sig_block_x + half

        # 方向定义：dy, dx，与扫描轴
        directions = [
            ('up',    -1,  0, 0),   # 向上：y 减小
            ('down',   1,  0, 0),
            ('left',   0, -1, 1),
            ('right',  0,  1, 1),
        ]
        bounds = {}

        for dir_name, dy, dx, axis in directions:
            cur_y, cur_x = center_y, center_x
            last_valid_y, last_valid_x = cur_y, cur_x
            blank_count = 0
            max_steps = max(h, w) // self.stride  # 最大搜索步数
            step_count = 0

            while 0 <= cur_y < h and 0 <= cur_x < w and step_count < max_steps:
                # 通过移动步长 self.stride 来探测
                if dir_name in ('up', 'down'):
                    cur_y += dy * self.stride
                else:
                    cur_x += dx * self.stride

                # 边界检查
                if cur_y < 0 or cur_y + ps > h or cur_x < 0 or cur_x + ps > w:
                    break

                block = image_mmap[cur_y:cur_y+ps, cur_x:cur_x+ps]
                
                if not self._is_background(block, modality):
                    last_valid_y, last_valid_x = cur_y + half, cur_x + half
                    blank_count = 0
                else:
                    blank_count += 1
                    if blank_count >= 3:  # 连续3次背景，认为到达边界
                        break
                step_count += 1
            else:
                # 到达最大步数但未下决心，使用最后一次有效位置
                pass

            bounds[dir_name] = (last_valid_y, last_valid_x)

        # 提取边界坐标
        y_min = bounds['up'][0] - half
        y_max = bounds['down'][0] + half + ps
        x_min = bounds['left'][1] - half
        x_max = bounds['right'][1] + half + ps

        # 钳位到图像内部
        y_min = max(0, y_min)
        y_max = min(h, y_max)
        x_min = max(0, x_min)
        x_max = min(w, x_max)

        # 简单检查合法性
        if y_min >= y_max or x_min >= x_max:
            print("[!] ROI 计算失败，回退全图")
            return 0, h, 0, w

        return y_min, y_max, x_min, x_max
    def extract(self, image_mmap: np.ndarray, modality: str) -> Iterator[Tuple[int, int, np.ndarray, bool]]:
        h, w = image_mmap.shape[:2]
        # 1. FL估计背景
        if modality == 'fluorescence':
            self._estimate_background(image_mmap, modality)
        else:
            self.bg_mean, self.bg_std = None, None # H&E默认白底背景统计

        # 2. 粗搜索信号块（中心优先）
        signal_pos = self._find_first_signal_block(image_mmap, modality)
        
        if signal_pos is not None:
            roi = self._find_roi_by_growth(image_mmap, *signal_pos, modality)
            print(f"[*] 精细提取区域: y:[{roi[0]}, {roi[1]}], x:[{roi[2]}, {roi[3]}]")
        else:
            roi = (0, h, 0, w)

        y_min, y_max, x_min, x_max = roi

        # 3. 仅在ROI内滑动窗口
        for y in range(y_min, y_max - self.patch_size + 1, self.stride):
            for x in range(x_min, x_max - self.patch_size + 1, self.stride):
                patch = image_mmap[y:y+self.patch_size, x:x+self.patch_size]
                if modality == 'he':
                    is_blank = False # 后续思考由于HE模式在使用的时候几乎不存在空白切片，所以直接跳过空白判定以提升效率
                else:
                    is_blank = self._is_background(patch, modality)

                if is_blank:
                    if modality == "he":
                        dummy = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.float32)
                    else:
                        dummy = np.zeros((self.patch_size, self.patch_size, 1), dtype=np.float32)
                    yield (y, x, dummy, True)
                else:
                    if modality == "fluorescence":
                        normed = normalize_fluorescence(patch)
                        if normed.ndim == 2:
                            normed = normed[..., np.newaxis] # normed[:, :, np.newaxis] 在数组后面加一个维度，变成单通道格式 (H, W, 1（N）)
                        yield (y, x, normed, False)
                    elif modality == "he":
                        normed = normalize_he(patch)
                        yield (y, x, normed, False)
                
if __name__ == "__main__":
    # 测试代码
    file_path = "D:\\files\\HE\\SICAPv2\\images\\\\16B0001851_Block_Region_1_0_0_xini_6803_yini_59786.jpg"
    image_mmap = lazy_load_wsi(file_path)
    modality = detect_modality(image_mmap)
    print(f"检测到的模态: {modality}")

     # ===== 验证 VirtualWSI 切片读取是否正常 =====
    print("-" * 40)
    print("[*] 执行切片读取验证...")
    h, w = image_mmap.shape[:2]
    cy, cx = h // 2, w // 2
    size = 512
    try:
        test_patch = image_mmap[cy:cy+size, cx:cx+size]
        print(f"   中心区域 ({cy}, {cx}) 形状: {test_patch.shape}, dtype: {test_patch.dtype}")
        print(f"   最小值: {test_patch.min()}, 最大值: {test_patch.max()}, 均值: {test_patch.mean():.2f}")
        # 可选：保存为图像以便人工检查
        # import cv2
        # cv2.imwrite("test_center_patch.png", (test_patch // 256).astype(np.uint8))
    except Exception as e:
        print(f"[!] 切片读取测试失败: {e}")
    print("-" * 40)
    # ===== 测试结束 =====   

    extractor = PatchExtractor(patch_size=512, stride=256)
    for y, x, patch, is_blank in extractor.extract(image_mmap, modality):
        print(f"Patch at ({y}, {x}), Blank: {is_blank}, Patch shape: {patch.shape}")
