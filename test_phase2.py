"""
Phase 2 端到端集成测试脚本：
使用真实的 SVS 和 GeoJSON 文件，串联 Phase 1 (DataLoader) 和 Phase 2 (HEModel + Losses)，
验证全链路的前向传播、损失计算和梯度更新。
"""

import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

# 导入 Phase 1 的数据层组件
from Train.datasets.transforms import HETrainTransform
from Train.datasets.dataloader import WSIHEDataset, he_collate_fn
from torch.utils.data import DataLoader

# 导入 Phase 2 的模型与损失函数
from Train.models.he_model import HEModel
from Train.losses.HE_losses import he_losses

def test_end_to_end():
    print("=== 开始 Phase 2 真实数据端到端集成测试 ===\n")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前设备: {device}")

    # =====================================================================
    # 1. 配置真实文件路径
    # =====================================================================
    SVS_DIR = "D:\\files\\HE\\train\\raw\\1\\svs"              # 真实的 SVS 存放目录
    GEOJSON_DIR = "D:\\files\\HE\\train\\raw\\1\\geojson"          # 真实的 GeoJSON 存放目录

    # 这里需要您填入一个真实的 SVS 文件名，以及该 SVS 上的两个真实中心坐标
    # 建议坐标取在肿瘤/组织的中心，不要取在全白玻璃处
    real_svs_name = "a-202500006-1.svs"  
    cx1, cy1 = 64562.65, 70901.78        
    cx2, cy2 = 10000, 10000        

    mock_patch_list = [
        {"svs_name": real_svs_name, "cx": cx1, "cy": cy1},
    ]

    # =====================================================================
    # 2. 初始化真实数据流 (Phase 1)
    # =====================================================================
    print(">>> 正在初始化数据管道...")
    transform = HETrainTransform(color_jitter_params={'brightness':0.3, 'contrast':0.3})
    dataset = WSIHEDataset(
        svs_dir=SVS_DIR,
        geojson_dir=GEOJSON_DIR,
        patch_list=mock_patch_list,
        transform=transform,
        size=512,
        max_cache_size=2
    )
    
    loader = DataLoader(dataset, batch_size=2, collate_fn=he_collate_fn)
    batch = next(iter(loader))

    if not batch:
        print("FATAL: Batch为空，请检查文件路径或坐标是否有效。")
        return

    # 将所有 Tensor 移动到计算设备 (GPU)
    branch_a      = batch['branch_a'].to(device)
    branch_b_gray = batch['branch_b_gray'].to(device)
    global_rgb    = batch['global_rgb'].to(device)
    original_rgb  = batch['original_rgb'].to(device)
    semantic      = batch['semantic'].to(device)
    boundary      = batch['boundary'].to(device)
    is_labeled    = batch['is_labeled'].to(device)

    print(f"数据加载成功! Batch Size: {branch_a.size(0)}")

    # =====================================================================
    # 3. 初始化双尺度模型与优化器 (Phase 2)
    # =====================================================================
    print("\n>>> 正在加载 HEModel 模型...")
    # he_classes 强制设为 6，匹配我们定死的 0~5 的分类体系
    model = HEModel(he_classes=6, base=64, global_base=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 启用 AMP 混合精度 (模拟真实训练环境)
    scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None

    # =====================================================================
    # 4. 执行前向传播
    # =====================================================================
    print("\n>>> 正在执行前向传播 (Forward)...")
    model.train()
    optimizer.zero_grad()

    # AMP 上下文
    with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled = False):
        outputs = model(branch_a, branch_b_gray, global_rgb)
        
        # 核验输出维度
        B, H, W = branch_a.size(0), branch_a.size(2), branch_a.size(3)
        assert outputs['semantic'].shape == (B, 6, H, W), "Semantic 维度错误！"
        assert outputs['boundary'].shape == (B, 1, H, W), "Boundary 维度错误！"
        assert outputs['colorization'].shape == (B, 3, H, W), "Colorization 维度错误！"
        print("前向传播完成，维度核验通过。")

        # =====================================================================
        # 5. 执行损失计算
        # =====================================================================
        print("\n>>> 正在执行损失计算 (Loss Calculation)...")
        labels = {
            'semantic': semantic,
            'boundary': boundary,
            'original_rgb': original_rgb,
            'is_labeled': is_labeled
        }
        task_weights = {'semantic': 1.0, 'boundary': 0.5, 'colorization': 0.8}
        
        loss_total, loss_dict = he_losses(outputs, labels, task_weights)
        print(f"总 Loss: {loss_total.item():.4f}")
        print(f"子任务分解: { {k: f'{v:.4f}' for k, v in loss_dict.items()} }")

    # =====================================================================
    # 6. 执行反向传播与梯度更新
    # =====================================================================
    print("\n>>> 正在执行反向传播 (Backward)...")
    if scaler:
        scaler.scale(loss_total).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss_total.backward()
        optimizer.step()

    # 验证 Local 和 Global 编码器是否都接收到了梯度
    local_grad_norm = model.encoder.inc.conv[0].weight.grad.norm().item()
    global_grad_norm = model.global_encoder.inc.conv[0].weight.grad.norm().item()
    
    print(f"Local Encoder 梯度范数: {local_grad_norm:.4f}")
    print(f"Global Encoder 梯度范数: {global_grad_norm:.4f}")
    
    assert local_grad_norm > 0, "致命错误：Local 分支梯度断裂！"
    assert global_grad_norm > 0, "致命错误：Global 分支梯度断裂！"

    print("\n✅ 恭喜！基于真实 SVS/GeoJSON 的端到端链路测试全部通过！完美闭环！")

if __name__ == "__main__":
    test_end_to_end()