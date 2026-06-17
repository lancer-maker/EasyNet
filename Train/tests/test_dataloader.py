import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.dataloader import get_dataloader
import tempfile
import shutil
import numpy as np
import cv2
import torch

def create_dummy_he_data(root: Path):
        """创建 HE 假数据：有标签和无标签"""
        # 有标签部分
        labeled_img_dir = root / "labeled" / "images"
        labeled_mask_dir = root / "labeled" / "masks"
        labeled_img_dir.mkdir(parents=True, exist_ok=True)
        labeled_mask_dir.mkdir(parents=True, exist_ok=True)

        # 无标签部分
        unlabeled_img_dir = root / "unlabeled" / "images"
        unlabeled_img_dir.mkdir(parents=True, exist_ok=True)

        # 生成 5 张有标签图像（512x512 RGB），对应掩码（类别 0,1,2,3）
        for i in range(5):
            img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
            cv2.imwrite(str(labeled_img_dir / f"img_{i}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            mask = np.random.randint(0, 4, (512, 512), dtype=np.uint8)
            cv2.imwrite(str(labeled_mask_dir / f"img_{i}.png"), mask)

        # 生成 10 张无标签图像
        for i in range(10):
            img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
            cv2.imwrite(str(unlabeled_img_dir / f"unlabeled_{i}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        return labeled_img_dir.parent, unlabeled_img_dir.parent

    # ========== HE 测试 ==========
print("=" * 50)
print("Testing HE dataloader...")

with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        labeled_root, unlabeled_root = create_dummy_he_data(tmp)
        config = {
            'modality': 'he',
            'data': {
                'labeled_path': r'D:/files//HE/test/labeled',
                'unlabeled_path':r'D:\\files\\HE\\test\\unlabeled',
                'val_path': r'D:/files//HE/test/labeled',  # 验证集暂时复用有标签
                'batch_size': 2,
                'num_workers': 0,
                'size': 512,
                'image_dir': 'images',
                'mask_dir': 'masks',
                'boundary_dir': None,
                'image_ext': ['.png', '.tif', '.tiff', '.jpg', '.jpeg'],
                'mask_ext': ['.png', '.tif', '.tiff', '.jpg', '.jpeg'],
                'he_classes': 2             # 暂时只有 2 个类别（0 和 1），以测试边界生成和损失计算
            }
        }
        
        try:
            train_loader = get_dataloader(config, split='train')
            val_loader = get_dataloader(config, split='val')
            print(f"Train batches: {len(train_loader)}")
            batch = next(iter(train_loader))
            print("Keys in train batch:", batch.keys())
            print("Branch A shape:", batch['branch_a'].shape)     # (B,3,H,W)
            print("Branch B gray shape:", batch['branch_b_gray'].shape)
            print("Original RGB shape:", batch['original_rgb'].shape)
            # 有标签样本应有 semantic 和 boundary，无标签样本可能没有，ConcatDataset 混合后无统一键
# ==========================================================================            
            batch = next(iter(train_loader))
            print("Keys:", batch.keys())
            for key in ['branch_a', 'branch_b_gray', 'original_rgb', 'semantic', 'boundary']:
                if key in batch:
                    val = batch[key]
                    if isinstance(val, list):
                        print(f"{key}: list of length {len(val)}, "
                            f"first element type: {type(val[0]).__name__ if val[0] is not None else 'None'}")
                    elif isinstance(val, torch.Tensor):
                        print(f"{key} shape: {val.shape}")
                    else:
                        print(f"{key}: {type(val).__name__}")
# ===========================================================================
            if 'semantic' in batch:
                print("Semantic mask shape:", batch['semantic'].shape)
                print("Boundary mask shape:", batch['boundary'].shape)
            else:
                print("Note: batch may contain unlabeled samples (no 'semantic' key)")
            print("HE dataloader test PASSED.\n")
        except Exception as e:
            print(f"HE test FAILED: {e}\n")
        
