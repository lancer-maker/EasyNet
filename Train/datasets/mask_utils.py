# 类别映射(5类+1Ignore)、绘制优先级、边界生成
import cv2
import numpy as np

CLASS_MAPPING = {
    "Background": 0,           # 显微镜白底 / 玻璃空白区 / 未标注基质/大部分自动跳过
    "Normal": 1,               # 正常肺组织
    "Tumor": 2,                # 肺癌主病灶 (浸润性成分)
    "STAS-micropapilla": 3,    # STAS 微乳头型
    "STAS-solid nests": 4,     # STAS 实性巢型
    "STAS-single cell": 5,     # STAS 单细胞型
    "Ignore*": 255             # 伪影、折叠、气道内杂质等不参与梯度计算区域
}

# 局部绘制覆盖优先级 (数值越大越晚绘制，强制覆盖下层)
DRAW_PRIORITY = {
    255: 100, # Ignore 绝对置顶覆盖
    5: 50,    # STAS 细分亚型置于最前
    4: 50,
    3: 50,
    2: 20,    # Tumor 覆盖正常组织
    1: 10,    # Normal
    0: 0      # Background 默认垫底
}

# =====================================================================
# 一、 边界掩码工具 (与 6 类别体系完全对齐，强制屏蔽 Ignore=255)
# =====================================================================
def generate_boundary_from_semantic(semantic_mask: np.ndarray, dilation_radius: int = 2) -> np.ndarray:
    """
    从语义掩码中提取前景边界 (有效前景类别为 1~5，0为背景，255为Ignore)。
    dilation_radius：膨胀 / 腐蚀的半径，默认 2，控制边界带的宽度。
    """
    foreground = ((semantic_mask > 0) & (semantic_mask < 255)).astype(np.uint8)
    if foreground.sum() == 0:
        return foreground

    # 卷积核采用椭圆形结构元素，膨胀和腐蚀操作提取边界
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius * 2 + 1, dilation_radius * 2 + 1))

    # 对前景foreground做膨胀操作，膨胀次数为1，得到膨胀后的图像dilated
    dilated = cv2.dilate(foreground, kernel, iterations=1)
    # 对前景foreground做腐蚀操作，腐蚀次数为1，得到腐蚀后的图像eroded
    eroded = cv2.erode(foreground, kernel, iterations=1)
    boundary = dilated - eroded

    # 屏蔽渗透进 255 的边缘伪影
    boundary[semantic_mask == 255] = 0
    return boundary

def map_raw_name_to_class_id(raw_name: str) -> int:
    """
    将 GeoJSON 中可能带有前缀或后缀的原始名称，智能映射到标准的 Class ID。
    如果实在匹配不到，默认返回 255 (Ignore)。
    """
    if not raw_name or not isinstance(raw_name, str):
        return 255
        
    name_lower = raw_name.strip().lower()

    # 1. 映射 Normal (包含血管、淋巴、出血、支气管、纤维组织等)
    normal_keywords = ['vessel', 'lymphocyte', 'hemorrhage', 'bronchus', 'fibrous_tissue', 'normal']
    if any(k in name_lower for k in normal_keywords):
        return CLASS_MAPPING["Normal"]

    # 2. 映射 STAS 单细胞
    if 'stas_single_cancer_cells' in name_lower or 'single cell' in name_lower:
        return CLASS_MAPPING["STAS-single cell"]

    # 3. 映射 STAS 微乳头
    if 'stas_micropapillary_clusters' in name_lower or 'micropapilla' in name_lower:
        return CLASS_MAPPING["STAS-micropapilla"]

    # 4. 映射 STAS 实体巢
    if 'stas_solid_nests' in name_lower or 'solid nests' in name_lower:
        return CLASS_MAPPING["STAS-solid nests"]

    # 5. 映射 Tumor 主体
    if 'tumor' in name_lower:
        return CLASS_MAPPING["Tumor"]

    # 6. 映射 Ignore (伪影、碳末沉着等)
    ignore_keywords = ['artifact', 'anthracotic_pigment', 'ignore']
    if any(k in name_lower for k in ignore_keywords):
        return CLASS_MAPPING["Ignore*"]

    # 7. 找不到匹配项时的兜底安全策略：设为 255
    print(f"Warning: Unknown label '{raw_name}' encountered! Mapped to Ignore (255).")
    return 255