# Shapely 2.x STRtree 空间索引与局部光栅化
import json
import numpy as np
import cv2
from pathlib import Path
from shapely.geometry import shape, box
from shapely.strtree import STRtree
from .mask_utils import CLASS_MAPPING, DRAW_PRIORITY,map_raw_name_to_class_id

class GeoJSONManager:
    # """负责解析标注数据并构建局部光栅化空间索引 (依赖 Shapely >= 2.0)"""
    def __init__(self, json_path: str):
        self.polygons = []
        self.labels = []
        self.priorities = []

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for feature in data.get('features', []):
                geom = shape(feature.get('geometry', {}))
                if not geom.is_valid:
                    geom = geom.buffer(0) # 拓扑修复自交多边形
                    print(f"Warning: Invalid geometry fixed for feature {feature.get('id', 'unknown')}")

                # 解析类别名称，未识别字段一律退回到 Ignore*
                raw_name = feature.get('properties', {}).get('classification', {}).get('name', '')
                class_id = map_raw_name_to_class_id(raw_name)
                priorities = DRAW_PRIORITY.get(class_id, 0)  # 默认优先级为 0

                self.polygons.append(geom)
                self.labels.append(class_id)
                self.priorities.append(priorities)

        self.tree = STRtree(self.polygons)  # 构建空间索引

#      """根据 Level-0 中心点裁切相交多边形，按优先级栅格化为 (patch_size, patch_size) 的 uint8 掩码"""
    def rasterize_patch(self, cx: int, cy: int, patch_size: int = 512) -> np.ndarray:
        half = patch_size // 2
        min_x, min_y = cx - half, cy - half
        max_x, max_y = cx + half, cy + half
        bbox = box(min_x, min_y, max_x, max_y)

        mask = np.zeros(
            (patch_size, patch_size), dtype=np.uint8
        )
        intersect_indices = self.tree.query(bbox)  # 所有和当前这个矩形 bbox 有交集的几何对象返回它们在 self.polygons 里的索引位置
        if len(intersect_indices) == 0:
            return mask  # 没有交集，直接返回全零掩码

        # 按优先级升序排列，使高优先级多边形后画以覆盖低优先级
        candidates = []
        for idx in intersect_indices:
            candidates.append(
                {
                'geom': self.polygons[idx],
                'label': self.labels[idx],
                'priority': self.priorities[idx]
                }
            )
        candidates.sort(key=lambda x: x['priority'])

        for item in candidates:
            intersection = item['geom'].intersection(bbox) # 将多边形与bbox的相交部分保存到 intersection
            if intersection.is_empty:
                continue

            # 光栅化当前多边形
            geom_list = [intersection] if intersection.geom_type == 'Polygon' else getattr(intersection, 'geoms', [])
            for g in geom_list:
                if g.geom_type != 'Polygon':
                    continue

                # 将坐标系平移至patch局部[0，patch_size]
                ext_coords = np.array(g.exterior.coords) - np.array([min_x, min_y])
                ext_pts = np.round(ext_coords).astype(np.int32)
                cv2.fillPoly(mask, [ext_pts], item['label'])
        return mask