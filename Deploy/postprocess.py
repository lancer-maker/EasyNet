# 荧光(分水岭+绿圈) / HE(Argmax+危险热力图)
'''
暂未测试
'''

import numpy as np
import cv2
from skimage.morphology import h_minima
from skimage.segmentation import watershed
from skimage.measure import label
from skimage.segmentation import find_boundaries

def postprocess_fluorescence(prob_map: np.ndarray): # 输入一张二维概率图
    """荧光：h-minima极值抑制 + 标记分水岭"""
    inv_dist = -prob_map # 概率图取反，高概率转为低深度
    # 找寻细胞核心作为种子点
    markers = label(h_minima(inv_dist, h=0.1)) # h=0.1 表示只有“深度”至少为 0.1 的局部极小值才被保留
    # 沿边界分割粘连细胞
    instance_map = watershed(inv_dist, markers, mask=(prob_map > 0.5))
    # watershed：以 inv_dist 作为地形高度图，以 markers 作为初始种子点，模拟水从种子点向外“注水”，不同种子点的水相遇的地方形成分水岭
    
    cell_count = len(np.unique(instance_map)) - 1
    return cell_count, instance_map

def generate_fluo_overlay(orig_img: np.ndarray, inst_map: np.ndarray):
    """UI交付：生成带绿色细胞轮廓的原图"""
    overlay = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2RGB) if orig_img.ndim == 2 else orig_img.copy()
    contours = cv2.Canny(inst_map.astype(np.uint8), 1, 1)
    overlay[contours > 0] = [0, 255, 0] # 纯绿边缘
    return overlay

def postprocess_he(prob_map: np.ndarray):
    """
    【HE专属】Argmax 分类提取
    假设网络输出特征图 prob_map 形状为 (C, H, W)，C为类别数
    """
    # 沿着通道维度求最大值的索引，得到每个像素的类别分类
    class_map = np.argmax(prob_map, axis=0)
    return class_map

def generate_he_overlay(orig_img: np.ndarray, class_map: np.ndarray):
    """
    【HE专属】交付 UI 的【红黄橙高危热力 Overlay 叠加图】
    """
    # 确保原图是 3 通道 RGB
    if orig_img.ndim == 2:
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2RGB)
        
    # 创建一个纯黑的热力图画布
    heatmap = np.zeros_like(orig_img, dtype=np.uint8)
    
    # 定义危险颜色映射 (BGR格式，因为 OpenCV 默认BGR)
    # 假设类别: 0-背景(无色), 1-正常肺泡(微绿), 2-实性巢(橙色), 3-微乳头STAS(深红高危)
    COLOR_MAP = {
        1: [144, 238, 144], # 浅绿色 (正常组织)
        2: [0, 165, 255],   # 橙色 (中危)
        3: [0, 0, 255]      # 红色 (高危)
    }
    
    # 根据类别填充热力图画布
    for class_idx, color in COLOR_MAP.items():
        heatmap[class_map == class_idx] = color
        
    # Alpha Blending (透明度叠加)
    # 将热力图以 0.4 的透明度叠加到原图上，方便医生同时看清底层细胞和 AI 判定区域
    alpha = 0.4
    overlay = cv2.addWeighted(heatmap, alpha, orig_img, 1 - alpha, 0)
    
    # 将预测背景区域 (class_map == 0) 保留为原图，不叠加任何颜色
    mask_bg = (class_map == 0)
    overlay[mask_bg] = orig_img[mask_bg]
    
    return overlay

# 追加模块，用于处理单个块的叠加，避免全图操作
def generate_fluo_overlay(orig_img: np.ndarray, inst_map: np.ndarray):
    """全图版：交付绿色细胞轮廓 Overlay"""
    overlay = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2RGB) if orig_img.ndim == 2 else orig_img.copy()
    boundaries = find_boundaries(inst_map, mode='inner')
    overlay[boundaries] = [0, 255, 0]
    return overlay

def generate_fluo_overlay_patch(vis_patch: np.ndarray, inst_patch: np.ndarray):
    """分块版：防溢出的细胞边界提取"""
    overlay = vis_patch.copy()
    boundaries = find_boundaries(inst_patch, mode='inner')
    overlay[boundaries] = [0, 255, 0]
    return overlay

def generate_he_overlay_patch(vis_patch: np.ndarray, class_patch: np.ndarray):
    """
    对单个 patch 生成红黄橙热力叠加图。
    vis_patch: uint8 BGR 图像
    class_patch: 类别索引图 (H, W)
    """
    # 颜色映射 (BGR)
    COLOR_MAP = {
        1: [144, 238, 144],  # 浅绿（正常）
        2: [0, 165, 255],    # 橙色
        3: [0, 0, 255]       # 红色
    }
    heatmap = np.zeros_like(vis_patch, dtype=np.uint8)
    for cls_idx, color in COLOR_MAP.items():
        heatmap[class_patch == cls_idx] = color

    alpha = 0.4
    overlay = cv2.addWeighted(heatmap, alpha, vis_patch, 1 - alpha, 0)
    # 背景保留原图
    mask = (class_patch == 0)
    overlay[mask] = vis_patch[mask]
    return overlay