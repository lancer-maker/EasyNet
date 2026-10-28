'''
HE部分已验证
'''
# 统一数据加载器（支持 8/16-bit 混合、任意常见图像格式）
# 所有图像在读取后立即归一化到 float32 [0,1]，形状 (H,W,C) 或 (H,W)
import pandas as pd
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from .transforms import (
    HETrainTransform, HEValTransform, FluorescenceTransform,
)
from .wsi_reader import WSIReader
from .geojson_utils import GeoJSONManager
from .mask_utils import generate_boundary_from_semantic


def _normalize_image_by_dtype(img: np.ndarray) -> np.ndarray:
    """根据位深度动态归一化到 [0,1] float32"""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    else:
        # 假设已经是 float 且范围合理
        return img.astype(np.float32)


class WSIHEDataset(Dataset):
    """
    双尺度 WSI 训练数据集：支持有标签与无标签半监督数据的安全隔离。
    统一绝对基准：Level-0 像素坐标。
    """
    def __init__(self, svs_dir, patch_list, geojson_dir=None, transform=None, size=512, max_cache_size=2):
        """
        patch_list: 包含采样点信息的字典列表 [{"svs_name": "x.svs", "cx": 1000, "cy": 1000}, ...]
        max_cache_size: 单个 worker 允许同时缓存的最大 WSI 数量。
        """
        self.svs_dir = Path(svs_dir)
        self.geojson_dir = Path(geojson_dir) if geojson_dir else None
        self.patch_list = patch_list
        self.transform = transform
        self.size = size

        # LRU 缓存控制
        self.max_cache_size = max_cache_size
        self.wsi_cache = {}
        self.geo_cache = {}
        self.cache_keys = []  # 记录使用顺序，队尾为最新使用

    def _get_managers(self, svs_name: str):
        if svs_name not in self.wsi_cache:
            # 1. 触发 LRU 淘汰机制
            if len(self.cache_keys) >= self.max_cache_size:
                oldest_svs = self.cache_keys.pop(0)  # 弹出最旧的
                old_reader = self.wsi_cache.pop(oldest_svs)
                self.geo_cache.pop(oldest_svs)
                # 显式关闭文件句柄，防止 OOM 与 Errno 24
                if hasattr(old_reader.slide, 'close'):
                    old_reader.slide.close()

            # 2. 加载新的 WSI 读取器
            self.wsi_cache[svs_name] = WSIReader(self.svs_dir / svs_name)
            
            # 3. 加载新的 GeoJSON 空间索引 (若存在)
            self.geo_cache[svs_name] = None
            if self.geojson_dir:
                json_path = self.geojson_dir / (Path(svs_name).stem + ".geojson")
                if json_path.exists():
                    self.geo_cache[svs_name] = GeoJSONManager(json_path)
            
            self.cache_keys.append(svs_name)
        else:
            # 更新使用顺序 (移动到队尾)
            self.cache_keys.remove(svs_name)
            self.cache_keys.append(svs_name)

        return self.wsi_cache[svs_name], self.geo_cache[svs_name]

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, idx):
        info = self.patch_list[idx]
        svs_name = info['svs_name']
        cx, cy = int(info['cx']), int(info['cy'])

        wsi_reader, geo_manager = self._get_managers(svs_name)

        # 1. 双尺度提取 (保证返回 float32 [0,1], 形状精确为 512x512x3)
        local_img = wsi_reader.read_region_safe(cx, cy, level=0, output_size=self.size)
        global_img = wsi_reader.read_region_safe(cx, cy, level=2, output_size=self.size)

        # 2. 标签生成（严格区分有监督与无标签半监督数据）
        if geo_manager is not None:
            mask_sem = geo_manager.rasterize_patch(cx, cy, patch_size=self.size)
        else:
            # 无标签数据：绝对初始化为 255 (Ignore)
            mask_sem = np.full((self.size, self.size), 255, dtype=np.uint8)

        mask_boundary = generate_boundary_from_semantic(mask_sem)

        # 3. 数据变换
        if self.transform:
            branch_a, branch_b_gray, orig_local, global_rgb, mask_sem, mask_boundary = self.transform(
                local_img, global_img, mask_sem, mask_boundary
            )
        else:
            branch_a = local_img
            branch_b_gray = 0.299 * local_img[:, :, 0] + 0.587 * local_img[:, :, 1] + 0.114 * local_img[:, :, 2]
            orig_local = local_img.copy()
            global_rgb = global_img

        return {
            'branch_a': torch.from_numpy(branch_a.transpose(2, 0, 1)).float(),           # (3, H, W)
            'branch_b_gray': torch.from_numpy(branch_b_gray).unsqueeze(0).float(),       # (1, H, W)
            'original_rgb': torch.from_numpy(orig_local.transpose(2, 0, 1)).float(),     # (3, H, W)
            'global_rgb': torch.from_numpy(global_rgb.transpose(2, 0, 1)).float(),       # (3, H, W)
            'semantic': torch.from_numpy(mask_sem).long(),                               # (H, W) 取值 0~5 及 255
            'boundary': torch.from_numpy(mask_boundary).unsqueeze(0).float(),            # (1, H, W) 取值 0, 1
            'is_labeled': torch.tensor(1 if geo_manager is not None else 0, dtype=torch.uint8),
            'filename': f"{svs_name}_X{cx}_Y{cy}"
        }

def he_collate_fn(batch):
    """支持有标签与无标签混合堆叠的 Collate 函数"""
    # 过滤可能因为读取失败返回 None 的样本
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return {}

    collated = {
        'branch_a': torch.stack([s['branch_a'] for s in batch]),
        'branch_b_gray': torch.stack([s['branch_b_gray'] for s in batch]),
        'original_rgb': torch.stack([s['original_rgb'] for s in batch]),
        'global_rgb': torch.stack([s['global_rgb'] for s in batch]),
        'semantic': torch.stack([s['semantic'] for s in batch]),
        'boundary': torch.stack([s['boundary'] for s in batch]),
        'is_labeled': torch.stack([s['is_labeled'] for s in batch]),
        'filename': [s['filename'] for s in batch]
    }
    return collated

class FluorescenceDataset(Dataset):
    """荧光弱监督数据集：root/class_name/*.ext, 输出归一化图像和干净目标"""
    '''
    关于csv适配暂未完成
    '''
    def __init__(self, root, transform=None, image_ext=None,csv_file=None):
        self.root = Path(root)
        self.transform = transform        
        if csv_file:
            x = pd.read_csv(csv_file)
            self.samples = [
                (Path(root) / row['filename'], 
                 row['label']                       # row：这一行--row['filename']：csv文件中filename行的值，row['label']：csv文件中label行的值
                 )
                for _, row in x.iterrows()          # _--接受iterrows返回的第一个值并且忽略它，row--接受iterrows返回的第二个值，即每行数据
            ]
        else:
            if image_ext is None:
                image_ext = ('.png', '.tif', '.tiff', '.jpg', '.jpeg')
            if isinstance(image_ext, str):
                image_ext = (image_ext,)
            self.image_ext = tuple(ext.lower() for ext in image_ext)

            # 收集所有类别文件夹
            self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

            self.samples = []   # 存储 (完整路径, 类别索引)
            for cls in self.classes:
                cls_dir = self.root / cls
                for f in cls_dir.iterdir():
                    if f.is_file() and f.suffix.lower() in self.image_ext:
                        self.samples.append((f, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # 荧光以灰度读取，保留原始动态范围
        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(f"Cannot read: {img_path}")
        image = _normalize_image_by_dtype(image)   # (H,W) float32

        clean_image = image.copy()

        if self.transform:
            transformed = self.transform(image)     # 返回 (C,H,W) tensor
        else:
            transformed = image[np.newaxis, ...]    # (1,H,W)

        clean_tensor = torch.from_numpy(clean_image).unsqueeze(0)  # (1,H,W)

        return {
            'image': torch.from_numpy(transformed),
            'class_label': label,
            'clean_image': clean_tensor,
            'filename': img_path.name
        }
    

def get_dataloader(config, split='train'):
    modality = config['modality'].lower()
    data_cfg = config['data']

    if modality == 'he':
        # --- HE 双尺度 WSI 模式 ---
        # 依赖配置文件提供 csv_path (包含 svs_name, cx, cy)，以及 svs_dir, geojson_dir
        csv_key = 'train_csv' if split == 'train' else 'val_csv'
        if csv_key not in data_cfg:
            raise ValueError(f"Missing '{csv_key}' in HE config for WSI sampling.")
            
        df = pd.read_csv(data_cfg[csv_key])
        patch_list = df.to_dict('records') # 转为 [{'svs_name':'..', 'cx':.., 'cy':..}, ...]

        if split == 'train':
            transform = HETrainTransform(color_jitter_params=data_cfg.get('color_jitter', {}))
            shuffle = True
        else:
            transform = HEValTransform()
            shuffle = False

        dataset = WSIHEDataset(
            svs_dir=data_cfg['svs_dir'],
            patch_list=patch_list,
            geojson_dir=data_cfg.get('geojson_dir', None),
            transform=transform,
            size=data_cfg.get('size', 512)
        )
        
        return DataLoader(
            dataset,
            batch_size=data_cfg.get('batch_size', 8),
            shuffle=shuffle,
            num_workers=data_cfg.get('num_workers', 4),
            pin_memory=True,
            collate_fn=he_collate_fn
        )

    elif modality == 'fluorescence_weak':
        transform = FluorescenceTransform(
            size=data_cfg.get('size', 512),
            noise_std=data_cfg.get('noise_std', 0.05),
            brightness=data_cfg.get('brightness', 0.2),
            contrast=data_cfg.get('contrast', 0.2)
        )
        dataset = FluorescenceDataset(
            root=data_cfg['train_path'],
            transform=transform,
            image_ext=data_cfg.get('image_ext', None)
        )
        batch_size = data_cfg.get('batch_size', 8)
        shuffle = (split == 'train')
        # if split == 'train':
        #     shuffle = True
        # else:
        #     shuffle = False
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=data_cfg.get('num_workers', 4),
            pin_memory=True,
            collate_fn=he_collate_fn
        )
        return dataloader
    else:
        raise ValueError(f"Unknown modality: {modality}")