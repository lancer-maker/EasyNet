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
    generate_boundary_from_semantic
)


def _normalize_image_by_dtype(img: np.ndarray) -> np.ndarray:
    """根据位深度动态归一化到 [0,1] float32"""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    else:
        # 假设已经是 float 且范围合理
        return img.astype(np.float32)


def _load_and_normalize(path: Path, color: bool = True):
    """
    读取图像并归一化到 float32 [0,1]。
    color=True  → RGB (H,W,3)
    color=False → 灰度 (H,W)
    """
    flag = cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(str(path), flag)
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    if color:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return _normalize_image_by_dtype(img)


class HEDataset(Dataset):
    """HE 数据集（有标签/无标签），支持混合位深度和多格式"""
    def __init__(self, root, labeled=True, transform=None,
                 image_dir='images', mask_dir='masks', boundary_dir=None,
                 image_ext=None, mask_ext=None):
        self.root = Path(root)
        self.labeled = labeled
        self.transform = transform
        self.image_dir = self.root / image_dir
        self.mask_dir = self.root / mask_dir if mask_dir else None
        self.boundary_dir = self.root / boundary_dir if boundary_dir else None

        # 扩展名处理：支持字符串或元组，统一为小写元组
        if image_ext is None:
            image_ext = ('.png', '.tif', '.tiff', '.jpg', '.jpeg')
        if mask_ext is None:
            mask_ext = ('.png', '.tif', '.tiff')
        if isinstance(image_ext, str):
            image_ext = (image_ext,)
        if isinstance(mask_ext, str):
            mask_ext = (mask_ext,)
        self.image_ext = tuple(ext.lower() for ext in image_ext)
        self.mask_ext = tuple(ext.lower() for ext in mask_ext)

        # 收集图像文件名（仅文件名，不含路径）
        if not self.image_dir.exists():
            raise ValueError(f"Image directory not found: {self.image_dir}")
        image_files = []
        for f in self.image_dir.iterdir():
            if f.is_file() and f.suffix.lower() in self.image_ext:
                image_files.append(f.name)
        self.images = sorted(image_files)

        if labeled:
            if not self.mask_dir or not self.mask_dir.exists():
                raise ValueError(f"Mask directory not found: {self.mask_dir}")
            # 确保图像有对应掩码
            matched = []
            for img_name in self.images:
                base = Path(img_name).stem                      # Path(img_name)：用 pathlib.Path 把字符串 img_name 转成路径对象。.stem：返回该路径最后一部分（文件名）去掉最后一个后缀后的字符串
                found = False                                   # eg. img_name = 'img_001.tif' → base = 'img_001'
                for ext in self.mask_ext:
                    if (self.mask_dir / (base + ext)).exists(): # 此处为多扩展名支持，检查 mask_dir 下是否存在与图像同名但扩展名为 mask_ext 中任一的文件
                        found = True
                        break
                if found:
                    matched.append(img_name)
            self.images = matched

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img_name = self.images[idx]
            img_path = self.image_dir / img_name
            image = _load_and_normalize(img_path, color=True)   # (H,W,3) float32

            mask_sem = None
            mask_boundary = None

        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            return None

        if self.labeled:
            # 寻找匹配的掩码
            base = Path(img_name).stem
            mask_path = None
            for ext in self.mask_ext:
                candidate = self.mask_dir / (base + ext)
                if candidate.exists():
                    mask_path = candidate
                    break
            if mask_path is None:
                raise FileNotFoundError(f"Mask not found for {img_name} in {self.mask_dir}")

            mask_sem = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)         # 以灰度模式从磁盘读取 mask 文件并返回为 NumPy 数组
            if mask_sem is None:
                raise IOError(f"Cannot read mask: {mask_path}")
            # 掩码保留原始整数类别

            # 边界掩码
            if self.boundary_dir:
                boundary_path = self.boundary_dir / (base + mask_path.suffix)   # Path.suffix 返回路径最后一部分的最后一个后缀（包含前导点），例如返回 '.txt'
                if boundary_path.exists():
                    mask_boundary = cv2.imread(str(boundary_path), cv2.IMREAD_GRAYSCALE)
                    if mask_boundary is not None:
                        mask_boundary = (mask_boundary > 127).astype(np.uint8)  # 由于后续损失函数使用的BCEWithLogitsLoss ，需要将边界掩码转换为二值化的 uint8 格式（0 或 1），以便正确计算边界损失 
            if mask_boundary is None:                                           # 可能是cv2读取失败或 boundary_dir 不存在，此时从语义掩码生成边界
                mask_boundary = generate_boundary_from_semantic(mask_sem)

        # 应用变换：统一数据格式，确保无论是否配置了数据增强（self.transform），后续流程都能拿到模型所需的 (image_aug, image_gray, original_rgb, mask_sem, mask_boundary) 五个输出
        if self.transform:
            image_aug, image_gray, original_rgb, mask_sem, mask_boundary = self.transform(
                image, mask_sem, mask_boundary
            )
        else:
            image_aug = image
            image_gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]   # 灰度化公式（0.299·R + 0.587·G + 0.114·B）
            original_rgb = image.copy()

        # 转为 tensor
        branch_a = torch.from_numpy(image_aug.transpose(2, 0, 1))  # (3,H,W)
        branch_b_gray = torch.from_numpy(image_gray).unsqueeze(0)   # (1,H,W)
        original_rgb_t = torch.from_numpy(original_rgb.transpose(2, 0, 1))

        sample = {
            'branch_a': branch_a,
            'branch_b_gray': branch_b_gray,
            'original_rgb': original_rgb_t,
            'filename': img_name
        }

        if self.labeled:
            sample['semantic'] = torch.from_numpy(mask_sem).long()
            sample['boundary'] = torch.from_numpy(mask_boundary).float().unsqueeze(0)

        return sample


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
    
def he_collate_fn(batch):
    """
    处理 HE 混合 batch：缺失的 semantic 用 ignore_index(255) 填充，
    缺失的 boundary 用 0 填充。
    """
    collated = {}

    # 基础图像字段（一定有）
    for key in ['branch_a', 'branch_b_gray', 'original_rgb']:
        collated[key] = torch.stack([s[key] for s in batch])

    # 文件名保留为列表
    collated['filename'] = [s['filename'] for s in batch]

    # 语义掩码：缺失时填充 255 (ignore_index)
    semantic_list = []
    for s in batch:
        if 'semantic' in s:
            semantic_list.append(s['semantic'])
        else:
            # 根据 branch_a 的空间尺寸创建全 255 张量
            _, H, W = s['branch_a'].shape
            semantic_list.append(torch.full((H, W), 255, dtype=torch.long))
    collated['semantic'] = torch.stack(semantic_list)

    # 边界掩码：缺失时填充 0
    boundary_list = []
    for s in batch:
        if 'boundary' in s:
            boundary_list.append(s['boundary'])
        else:
            _, H, W = s['branch_a'].shape
            boundary_list.append(torch.zeros((1, H, W), dtype=torch.float))
    collated['boundary'] = torch.stack(boundary_list)

    return collated

def get_dataloader(config, split='train'):
    modality = config['modality'].lower()
    data_cfg = config['data']

    if modality == 'he':
        if split == 'train':
            train_transform = HETrainTransform(
                size=data_cfg.get('size', 512),
                color_jitter_params=data_cfg.get('color_jitter', {})
            )
            labeled_dataset = HEDataset(
                root=data_cfg['labeled_path'],
                labeled=True,
                transform=train_transform,
                image_dir=data_cfg.get('image_dir', 'images'),
                mask_dir=data_cfg.get('mask_dir', 'masks'),
                boundary_dir=data_cfg.get('boundary_dir', None),
                image_ext=data_cfg.get('image_ext', None),
                mask_ext=data_cfg.get('mask_ext', None)
            )
            unlabeled_dataset = HEDataset(
                root=data_cfg['unlabeled_path'],
                labeled=False,
                transform=train_transform,
                image_dir=data_cfg.get('image_dir', 'images'),
                image_ext=data_cfg.get('image_ext', None)
            )
            dataset = ConcatDataset([labeled_dataset, unlabeled_dataset])
            batch_size = data_cfg.get('batch_size', 8)
            shuffle = True
        else:   # val
            val_transform = HEValTransform(size=data_cfg.get('size', 512))
            dataset = HEDataset(
                root=data_cfg['val_path'],
                labeled=True,
                transform=val_transform,
                image_dir=data_cfg.get('image_dir', 'images'),
                mask_dir=data_cfg.get('mask_dir', 'masks'),
                boundary_dir=data_cfg.get('boundary_dir', None),
                image_ext=data_cfg.get('image_ext', None),
                mask_ext=data_cfg.get('mask_ext', None)
            )
            batch_size = data_cfg.get('batch_size', 4)
            shuffle = False

        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
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
    else:
        raise ValueError(f"Unknown modality: {modality}")
    return dataloader