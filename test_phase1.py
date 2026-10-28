import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# 导入您重构后的数据层模块
from Train.datasets.transforms import HEValTransform
from Train.datasets.dataloader import WSIHEDataset, he_collate_fn
from torch.utils.data import DataLoader

def run_visual_validation():
    print("=== 开始 Phase 1 终极可视化与半监督验证 ===")

    # =====================================================================
    # 1. 目录配置 (请根据您的真实路径修改)
    # =====================================================================
    SVS_DIR = "D:\\files\\HE\\train\\raw\\1\\svs"              # 真实的 SVS 存放目录
    GEOJSON_DIR = "D:\\files\\HE\\train\\raw\\1\\geojson"          # 真实的 GeoJSON 存放目录

    # 我们挑两个真实存在的 SVS 文件名（假设您的真实文件名为 slide1.svs 和 slide2.svs）
    # 为了测试无标签，我们故意给 slide2 捏造一个假的 JSON 目录，让它找不到 JSON
    # 坐标请选择靠近肿瘤区域的真实位置，避免全白
    labeled_svs = "a-202500006-1.svs"     # 替换为真实的 SVS 文件名
    unlabeled_svs = "a-202500006-2.svs"   # 替换为真实的 SVS 文件名
    
    cx, cy = 64562.65, 70901.78      # 替换为真实的、有组织的中心点坐标

    mock_patch_list = [
        {"svs_name": labeled_svs, "cx": cx, "cy": cy},       # 样本 0：有标签测试
        {"svs_name": unlabeled_svs, "cx": cx, "cy": cy}      # 样本 1：无标签测试
    ]

    # 使用 ValTransform (无颜色抖动，方便肉眼比对原图)
    transform = HEValTransform()

    # =====================================================================
    # 2. 故意制造一个“空目录”来测试半监督 Unlabeled 分支
    # =====================================================================
    fake_empty_dir = Path("fake_empty_geojson_dir")
    fake_empty_dir.mkdir(exist_ok=True)

    # 样本 0 使用真实的 JSON 目录，样本 1 动态篡改（在 Dataset 层面不支持按样本指定 JSON dir）
    # 所以我们这里直接初始化 Dataset，但由于 slide2 的 json 可能不存在于真实目录中，
    # 如果您的 slide2 本来就有 JSON，建议您可以先把它移出目录一下，或者改个名字。
    
    dataset = WSIHEDataset(
        svs_dir=SVS_DIR,
        geojson_dir=GEOJSON_DIR, # 确保里面只有 labeled_svs 的 json，没有 unlabeled_svs 的 json
        patch_list=mock_patch_list,
        transform=transform,
        size=512,
        max_cache_size=2
    )

    loader = DataLoader(dataset, batch_size=2, collate_fn=he_collate_fn)
    batch = next(iter(loader))

    # =====================================================================
    # 3. 后台张量统计验证
    # =====================================================================
    print("\n[张量逻辑核验]")
    print(f"Is Labeled Flag: {batch['is_labeled']} (预期: [1, 0])")
    
    for i in range(2):
        mask_unique = torch.unique(batch['semantic'][i]).tolist()
        label_status = "有标签" if batch['is_labeled'][i] == 1 else "无标签"
        print(f"样本 {i} ({label_status}) 的语义掩码包含类别: {mask_unique}")
        if batch['is_labeled'][i] == 0:
            assert mask_unique == [255], f"致命错误：无标签数据的掩码不是纯 255！当前为 {mask_unique}"

    # =====================================================================
    # 4. 可视化绘图验证 (Dual-Scale 对齐 & Mask 覆盖)
    # =====================================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Phase 1: Dual-Scale & Semi-supervised Validation", fontsize=16)

    for i in range(2):
        # 取出图像并反归一化，从 (C,H,W) 转为 (H,W,C)
        local_img = batch['original_rgb'][i].permute(1, 2, 0).numpy()
        global_img = batch['global_rgb'][i].permute(1, 2, 0).numpy()
        sem_mask = batch['semantic'][i].numpy()
        is_labeled = batch['is_labeled'][i].item()

        # 图 1: Global Scale + 红色 Local 框
        ax_global = axes[i, 0]
        ax_global.imshow(global_img)
        title_g = f"Sample {i} (Global 10x)\n{mock_patch_list[i]['svs_name']}"
        ax_global.set_title(title_g)
        ax_global.axis('off')

        # 计算 Local 在 Global 中的相对位置 (Global的物理宽高是 Local 的 4倍)
        # 所以 512 物理像素在 Global 中占据 512/4 = 128 像素
        # 中心点坐标在 512 的中心 (256, 256)
        rect_size = 512 // 4
        rect_x = 256 - rect_size // 2
        rect_y = 256 - rect_size // 2
        rect = patches.Rectangle((rect_x, rect_y), rect_size, rect_size, linewidth=2, edgecolor='r', facecolor='none')
        ax_global.add_patch(rect)

        # 图 2: Local Scale
        ax_local = axes[i, 1]
        ax_local.imshow(local_img)
        ax_local.set_title("Local 40x\n(Should match red box perfectly)")
        ax_local.axis('off')

        # 图 3: Semantic Mask
        ax_mask = axes[i, 2]
        # 用特定的 vmin vmax 保证颜色映射固定
        cmap = plt.cm.get_cmap('tab10', 6) # 0-5 使用颜色
        cmap.set_over('gray')              # 255 使用灰色显示
        
        cax = ax_mask.imshow(sem_mask, cmap=cmap, vmin=0, vmax=5)
        status_text = "Labeled (Classes 0-5)" if is_labeled == 1 else "Unlabeled (All 255 / Gray)"
        ax_mask.set_title(f"Semantic Mask\n{status_text}")
        ax_mask.axis('off')
        
        if is_labeled == 1:
            fig.colorbar(cax, ax=ax_mask, ticks=range(6), fraction=0.046, pad=0.04)

    plt.tight_layout()
    # 建议保存为高分辨率图片仔细放大核对
    plt.savefig("phase1_alignment_check.svg",format = "svg", dpi=300)
    print("\n✅ 可视化结果已保存至 'phase1_alignment_check.svg'")
    print("请打开图片，用肉眼核实: \n1. Local 图是否与 Global 图中的红框内容严丝合缝。\n2. 样本 0 的 Mask 是否精准贴合组织轮廓。\n3. 样本 1 的 Mask 是否全灰（无标签）。")

if __name__ == "__main__":
    run_visual_validation()