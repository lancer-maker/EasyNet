import numpy as np
import tifffile

class VirtualWIS:
    '''
    使用tiffFile模拟numpy操作以实现惰性加载的虚拟WSI对象，兼容detect_modality和PatchExtractor的输入要求
    '''
    def __init__(self, tif: tifffile.TiffFile, level: int = 0):
        # 虚拟定义
        self.tif = tif
        self.page = tif.pages[level]          # 默认读取最高分辨率层
        self.shape = self.page.shape
        self.dtype = self.page.dtype
        self._page = self.page                # 缓存页面引用
        self.ndim = len(self.shape) # 维度信息，兼容单通道和多通道
    def __getitem__(self, key) -> np.ndarray:
        if not isinstance(key, tuple):
            key = (key, slice(None))
        try:    
            return self.page[key]
        except TypeError:
            return self.page.asarray()[key]
    
    def __array__(self):
        """禁止隐式转换为完整数组，防止 OOM"""
        raise MemoryError("VirtualWSI 不支持转换为完整 numpy 数组，请使用切片访问。")

    def close(self):
        self.tif.close() # 关闭文件释放资源