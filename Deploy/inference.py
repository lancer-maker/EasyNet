# 初始化onnxruntime并引用二维高斯权重矩阵，执行滑动切块总循环与边缘无缝融合
import numpy as np
import cv2
import onnxruntime as ort
from preprocess import lazy_load_wsi, detect_modality, PatchExtractor, normalize_fluorescence, normalize_he
from postprocess import postprocess_fluorescence, postprocess_he

class InferencePipeline:
    def __init__(self, fluo_onnx: str, he_onnx: str = None):
        # 强制 CPU 运行，兼容普通医生电脑
        self.sess_fluo = ort.InferenceSession(fluo_onnx, providers=['CPUExecutionProvider'])
        self.sess_he = ort.InferenceSession(he_onnx, providers=['CPUExecutionProvider']) if he_onnx else None
        self.extractor = PatchExtractor(patch_size=512, step=256)

        # 预计算二维高斯权重 (512, 512)
        x = np.linspace(-1, 1, 512)
        xx, yy = np.meshgrid(x, x)
        self.gauss_weight = np.exp(-(xx**2 + yy**2) / 0.5).astype(np.float32)

    def run(self, wsi_path: str):
        """执行完整推理，返回 (cell_count, overlay_image) 或 (overlay_image)"""
        mmap = lazy_load_wsi(wsi_path)
        modality = detect_modality(mmap)

        h, w = mmap.shape[:2]

        # ----- 初始化概率图融合画布 -----
        if modality == "fluorescence":
            canvas = np.zeros((h, w), dtype=np.float32)
            weight_canvas = np.zeros((h, w), dtype=np.float32)
        else:   # HE 需要多通道画布，通道数在首次推理时确定
            canvas = None
            weight_canvas = np.zeros((h, w), dtype=np.float32)

        patch_size = self.extractor.patch_size

        # ----- 滑动切块与推理 -----
        for y, x, patch, is_blank in self.extractor.extract(mmap, modality):
            if is_blank:
                continue

            # 正确转换为 ONNX 输入形状
            # patch 形状: (512, 512, 1) 荧光 或 (512, 512, 3) HE
            if patch.ndim == 2:
                patch = patch[..., np.newaxis]  # 单通道扩展为 (512, 512, 1)
            tensor = patch.transpose(2, 0, 1)[np.newaxis, ...]  # -> (1, C, H, W)

            if modality == "fluorescence":
                input_name = self.sess_fluo.get_inputs()[0].name
                pred = self.sess_fluo.run(None, {input_name: tensor})[0]
                # 预期输出 (1, 1, patch_size, patch_size)，取二维数据
                pred = pred[0, 0]   # shape (512, 512)
            else:
                if self.sess_he is None:
                    raise RuntimeError("HE ONNX 模型未提供")
                input_name = self.sess_he.get_inputs()[0].name
                pred = self.sess_he.run(None, {input_name: tensor})[0]
                # 预期输出 (1, C, patch_size, patch_size)，取多通道
                pred = pred[0]      # shape (C, 512, 512)
                if canvas is None:
                    num_classes = pred.shape[0]
                    canvas = np.zeros((num_classes, h, w), dtype=np.float32)

            # ----- 高斯融合 -----
            if modality == "fluorescence":
                canvas[y:y+patch_size, x:x+patch_size] += pred * self.gauss_weight
                weight_canvas[y:y+patch_size, x:x+patch_size] += self.gauss_weight
            else:
                # 广播权重到所有通道
                canvas[:, y:y+patch_size, x:x+patch_size] += pred * self.gauss_weight[np.newaxis, :, :]
                weight_canvas[y:y+patch_size, x:x+patch_size] += self.gauss_weight

        # 生成无缝概率大图
        if modality == "fluorescence":
            final_map = canvas / (weight_canvas + 1e-7)
        else:
            final_map = canvas / (weight_canvas[np.newaxis, :, :] + 1e-7)  # (C, H, W)

        # ----- 后处理（计数/分类）-----
        if modality == "fluorescence":
            cell_count, inst_map = postprocess_fluorescence(final_map)
        else:
            class_map = postprocess_he(final_map)   # (H, W)

        # ----- 生成叠加图（分块构建，防 OOM）-----
        overlay = self._build_overlay(mmap, modality, inst_map if modality == "fluorescence" else class_map)

        if modality == "fluorescence":
            return cell_count, overlay
        else:
            return overlay

    def _build_overlay(self, image_mmap, modality, processed_map):
        """分块生成最终叠加图，避免一次性加载全图"""
        h, w = image_mmap.shape[:2]
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        # 使用与推理相同的滑动窗口，保证块对齐
        patch_size = self.extractor.patch_size
        step = self.extractor.step

        for y in range(0, h - patch_size + 1, step):
            for x in range(0, w - patch_size + 1, step):
                # 只处理非空白区域（可根据 processed_map 判断是否有内容）
                if modality == "fluorescence":
                    # 若实例图中该块无细胞，可直接填充原图（不画轮廓）
                    inst_block = processed_map[y:y+patch_size, x:x+patch_size]
                    if np.max(inst_block) == 0:   # 无细胞，仅复制原图归一化块
                        orig_patch = image_mmap[y:y+patch_size, x:x+patch_size]
                        vis_patch = self._normalize_patch_for_display(orig_patch, modality)
                        overlay[y:y+patch_size, x:x+patch_size] = vis_patch
                    else:
                        orig_patch = image_mmap[y:y+patch_size, x:x+patch_size]
                        vis_patch = self._normalize_patch_for_display(orig_patch, modality)
                        # 调用后处理模块的单块叠加函数
                        from postprocess import generate_fluo_overlay_patch
                        overlay_patch = generate_fluo_overlay_patch(vis_patch, inst_block)
                        overlay[y:y+patch_size, x:x+patch_size] = overlay_patch
                else:   # HE
                    class_block = processed_map[y:y+patch_size, x:x+patch_size]
                    orig_patch = image_mmap[y:y+patch_size, x:x+patch_size]
                    vis_patch = self._normalize_patch_for_display(orig_patch, modality)
                    from postprocess import generate_he_overlay_patch
                    overlay_patch = generate_he_overlay_patch(vis_patch, class_block)
                    overlay[y:y+patch_size, x:x+patch_size] = overlay_patch

        return overlay

    def _normalize_patch_for_display(self, patch, modality):
        """将原始 patch 归一化为 uint8 彩色图，用于叠加显示"""
        if modality == "fluorescence":
            # 复用预处理中的归一化逻辑，但拉伸到 0-255
            normed = normalize_fluorescence(patch)   # float32 (512,512) 或 (512,512,1)
            if normed.ndim == 2:
                normed = normed[..., np.newaxis]
            vis = (normed * 255).astype(np.uint8)
            if vis.shape[-1] == 1:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2RGB)
            else:
                vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # OpenCV 用 BGR
            return vis
        else:
            normed = normalize_he(patch)   # float32 (512,512,3)
            vis = (normed * 255).astype(np.uint8)
            return cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)