# -*- coding: utf-8 -*-
"""small_look.py —— 导出「可安全截取」的小型 GeoJSON + 对应 SVS 元数据。

目标
----
给定一张 SVS(WSI) 与它的 GeoJSON 标注，导出一份：

1. **可以安全截取的小型 GeoJSON**（默认体积上限 50MB）；
2. **与选中区域对应的 SVS 元数据**（OpenSlide properties / level dimensions /
   downsample / MPP 等）。

两者合计体积控制在 ``--max-mb``（默认 50MB）以内。

「安全截取」的定义
------------------
1. 区域是一个矩形 ``(x, y, width, height)``，坐标单位为 **level-0 像素**。
2. 只保留 **完全落在矩形内** 的标注；与矩形边界相交（或落在矩形外）的标注一律剔除，
   并逐条记录到元数据的 ``dropped`` 里（含原因、分类名、bbox）。
   这样后续把 SVS 裁到同一矩形时，标注与图像严格对齐，不会出现「半个多边形」。
3. 元数据给出该区域在每一层的 **尺寸 / downsample / 起始坐标**，可直接用于裁剪或金字塔读取。
4. 所有输出文件都是 **先写临时文件再原子替换**（``os.replace``），不会留下半截文件。

用法
----
    # 直接使用文件末尾的 DEFAULT_* 路径（全片区域）
    python small_look.py

    # 自动挑选「装得下且 < 50MB」的最大区域
    python small_look.py --mode auto --max-mb 50

    # 手动指定 level-0 像素区域 x,y,w,h
    python small_look.py --mode manual --region 0,0,20000,20000

    # 输出到别处
    python small_look.py --outdir "D:\\some\\dir" --prefix demo
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import openslide
from shapely.geometry import box as shapely_box
from shapely.geometry import shape as shapely_shape
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid

# ---------------------------------------------------------------- 默认参数
DEFAULT_SVS = r"E:\TCGA_STAS+\WSI\TCGA-05-4402-01Z-00-DX1.c653ddc2-88c1-45ac-88e7-4e512b8e8d53.svs"
DEFAULT_GEOJSON = r"E:\TCGA_STAS+\geojson_format\TCGA-05-4402-01Z-00-DX1.c653ddc2-88c1-45ac-88e7-4e512b8e8d53.geojson"
DEFAULT_OUTDIR = r"D:\files\HE\train\raw\1\Temporary"
DEFAULT_MAX_MB = 50.0

# 输出 GeoJSON 使用紧凑分隔符以压缩体积
_JSON_KW = dict(ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------- 小工具
def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """先写同目录临时文件，fsync 后原子替换，避免产生半截文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp_path), str(path))
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _dump_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, **_JSON_KW).encode("utf-8")


def _count_points(geometry: Optional[Dict[str, Any]]) -> int:
    """统计坐标点个数（Polygon / MultiPolygon / LineString / Point 皆可）。"""
    if not geometry:
        return 0
    n = 0
    stack: List[Any] = [geometry.get("coordinates")]
    while stack:
        node = stack.pop()
        if isinstance(node, (list, tuple)):
            if node and isinstance(node[0], (int, float)):
                n += 1
            else:
                stack.extend(node)
    return n


def _class_name(props: Any) -> Optional[str]:
    if not isinstance(props, dict):
        return None
    cls = props.get("classification")
    if isinstance(cls, dict):
        return cls.get("name")
    if isinstance(cls, str):
        return cls
    return props.get("classification_name") or props.get("name")


def _bbox_to_list(geom: BaseGeometry) -> List[float]:
    return [float(v) for v in geom.bounds]


# ---------------------------------------------------------------- SVS 元数据
def read_svs_metadata(svs_path: Path) -> Dict[str, Any]:
    """读取完整 OpenSlide properties + level 信息 + MPP/物镜倍数等派生字段。"""
    slide = openslide.OpenSlide(str(svs_path))
    try:
        props = {str(k): str(v) for k, v in slide.properties.items()}
        level_count = int(slide.level_count)
        level_dims = [[int(w), int(h)] for w, h in slide.level_dimensions]
        level_ds = [float(d) for d in slide.level_downsamples]

        levels: List[Dict[str, Any]] = []
        for i in range(level_count):
            tw = props.get("openslide.level[%d].tile-width" % i)
            th = props.get("openslide.level[%d].tile-height" % i)
            levels.append(
                {
                    "level": i,
                    "dimensions": level_dims[i],
                    "downsample": level_ds[i],
                    "tile_width": int(tw) if tw and tw.isdigit() else None,
                    "tile_height": int(th) if th and th.isdigit() else None,
                }
            )

        mpp_x = props.get("openslide.mpp-x") or props.get("aperio.MPP")
        mpp_y = props.get("openslide.mpp-y") or props.get("aperio.MPP")
        objective = props.get("openslide.objective-power") or props.get("aperio.AppMag")

        meta: Dict[str, Any] = {
            "source_svs": str(svs_path),
            "svs_size_bytes": svs_path.stat().st_size,
            "vendor": slide.detect_format(str(svs_path)) or "unknown",
            "slide_dimensions_l0": level_dims[0],  # [width, height]，level-0 像素
            "level_count": level_count,
            "level_dimensions": level_dims,
            "level_downsamples": level_ds,
            "levels": levels,
            "mpp": {
                "x": float(mpp_x) if mpp_x else None,
                "y": float(mpp_y) if mpp_y else None,
                "unit": "micrometer_per_pixel" if mpp_x else None,
            },
            "objective_power": float(objective) if objective else None,
            "openslide_properties": props,
        }
    finally:
        slide.close()
    return meta


def _region_level_info(meta: Dict[str, Any], region: Dict[str, float]) -> List[Dict[str, Any]]:
    """把 level-0 区域换算到每一层：起始坐标 + 该区域在对应层的宽高。"""
    out: List[Dict[str, Any]] = []
    x, y, w, h = region["x"], region["y"], region["width"], region["height"]
    for lv in meta["levels"]:
        ds = lv["downsample"]
        lw, lh = lv["dimensions"]
        rx = int(math.floor(x / ds))
        ry = int(math.floor(y / ds))
        rw = min(lw - rx, int(math.ceil(w / ds)))
        rh = min(lh - ry, int(math.ceil(h / ds)))
        out.append(
            {
                "level": lv["level"],
                "downsample": ds,
                "level_dimensions": lv["dimensions"],
                "region_location_in_level": [max(rx, 0), max(ry, 0)],
                "region_dimensions_in_level": [max(rw, 0), max(rh, 0)],
                "tile_width": lv["tile_width"],
                "tile_height": lv["tile_height"],
            }
        )
    return out


# ---------------------------------------------------------------- 区域求解
def _complement_intervals(
    intervals: Iterable[Tuple[float, float]], lo: float, hi: float
) -> List[Tuple[float, float]]:
    """求 ``[lo, hi]`` 中不被任何开区间 (a, b) 覆盖的部分。

    落在这些闭区间里的坐标可以「安全切割」：不会穿过任何多边形的 bbox。
    """
    merged: List[List[float]] = []
    for a, b in sorted((min(a, b), max(a, b)) for a, b in intervals):
        a, b = max(a, lo), min(b, hi)
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])

    out: List[Tuple[float, float]] = []
    cur = lo
    for a, b in merged:
        if a > cur:
            out.append((cur, a))
        cur = max(cur, b)
    if cur < hi:
        out.append((cur, hi))
    return out


def _snap_cut(comp: Sequence[Tuple[float, float]], value: float, side: str) -> float:
    """把 ``value`` 吸附到最近的安全切割线上（side='lo' 向左，'hi' 向右）。"""
    for a, b in comp:
        if a <= value <= b:
            return a if side == "lo" else b
    if side == "lo":
        cands = [b for _, b in comp if b <= value]
        return max(cands) if cands else value
    cands = [a for a, _ in comp if a >= value]
    return min(cands) if cands else value


def _num(v: float) -> float:
    """整数值用 int 表示，便于元数据可读（如 59812 而非 59812.0）。"""
    fv = float(v)
    return int(fv) if fv.is_integer() else fv


def _region_dict(x0: float, y0: float, x1: float, y1: float, mode: str) -> Dict[str, Any]:
    return {
        "x": _num(x0),
        "y": _num(y0),
        "width": _num(x1 - x0),
        "height": _num(y1 - y0),
        "mode": mode,
        "units": "level0_pixels",
    }


def choose_region(
    mode: str,
    manual_region: Optional[Sequence[float]],
    slide_w: int,
    slide_h: int,
    infos: List[Dict[str, Any]],
    budget_bytes: int,
) -> Optional[Dict[str, float]]:
    """确定裁剪区域。

    - ``manual``: 直接用 ``--region x,y,w,h``。
    - ``full``  : 整张 slide。
    - ``auto``  : 在「安全切割线」组成的矩形集合中，贪心挑选体积不超过预算、
                  且包含标注最多的区域（best-effort）。

    ``infos`` 中每条需含 ``bbox`` / ``geom`` / ``bytes`` / ``area``。
    """
    if mode == "manual":
        if manual_region is None:
            raise ValueError("--mode manual 需要同时提供 --region x,y,w,h")
        x, y, w, h = manual_region
        return _region_dict(x, y, x + w, y + h, "manual")

    if mode == "full":
        return _region_dict(0, 0, slide_w, slide_h, "full")

    if not infos:
        return None

    comp_x = _complement_intervals(((f["bbox"][0], f["bbox"][2]) for f in infos), 0, slide_w)
    comp_y = _complement_intervals(((f["bbox"][1], f["bbox"][3]) for f in infos), 0, slide_h)

    # 先按面积从大到小贪心装包，再用「吸附后的真实体积」反复收缩。
    selected = sorted(infos, key=lambda f: (-f["area"], f["bytes"]))
    packed: List[Dict[str, Any]] = []
    used = 0
    for f in selected:
        if used + f["bytes"] <= budget_bytes:
            packed.append(f)
            used += f["bytes"]

    region_box: Optional[BaseGeometry] = None
    for _ in range(len(packed) + 2):
        if not packed:
            return None
        x0 = _snap_cut(comp_x, min(f["bbox"][0] for f in packed), "lo")
        x1 = _snap_cut(comp_x, max(f["bbox"][2] for f in packed), "hi")
        y0 = _snap_cut(comp_y, min(f["bbox"][1] for f in packed), "lo")
        y1 = _snap_cut(comp_y, max(f["bbox"][3] for f in packed), "hi")
        region_box = shapely_box(x0, y0, x1, y1)

        kept_bytes = 0
        for f in infos:
            bx = f["bbox"]
            if bx[0] >= x0 and bx[2] <= x1 and bx[1] >= y0 and bx[3] <= y1:
                if region_box.covers(f["geom"]):
                    kept_bytes += f["bytes"]
        if kept_bytes <= budget_bytes:
            return _region_dict(x0, y0, x1, y1, "auto")
        # 超预算：丢掉当前包内体积最大的一个，重新吸附
        packed.sort(key=lambda f: -f["bytes"])
        packed.pop(0)

    return None if region_box is None else None


# ---------------------------------------------------------------- GeoJSON 处理
def build_feature_infos(
    features: List[Dict[str, Any]], slide_box: BaseGeometry
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """预处理每个 feature：解析几何、修正非法几何、剔除退化/越界要素。"""
    infos: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []

    for idx, feat in enumerate(features):
        geom_dict = feat.get("geometry")
        base = {
            "index": idx,
            "classification": _class_name(feat.get("properties")),
        }
        if not geom_dict or not geom_dict.get("type"):
            dropped.append({**base, "reason": "missing_geometry"})
            continue

        try:
            geom = shapely_shape(geom_dict)
        except Exception as exc:  # noqa: BLE001
            dropped.append({**base, "reason": "geometry_parse_error", "detail": str(exc)})
            continue

        if not geom.is_valid:
            try:
                fixed = make_valid(geom)
                if fixed.is_valid and not fixed.is_empty:
                    geom = fixed
                else:
                    dropped.append(
                        {**base, "reason": "invalid_geometry", "bbox": _bbox_to_list(geom)}
                    )
                    continue
            except Exception:  # noqa: BLE001
                dropped.append({**base, "reason": "invalid_geometry", "bbox": _bbox_to_list(geom)})
                continue

        if geom.is_empty or geom.area <= 0:
            dropped.append({**base, "reason": "degenerate_geometry", "bbox": _bbox_to_list(geom)})
            continue

        if not slide_box.intersects(geom):
            dropped.append({**base, "reason": "outside_slide_bounds", "bbox": _bbox_to_list(geom)})
            continue

        infos.append(
            {
                **base,
                "feature": feat,  # 保留原始字典，输出时不做坐标 round-trip，保持整数像素
                "geom": geom,
                "bbox": _bbox_to_list(geom),
                "area": float(geom.area),
                "points": _count_points(geom_dict),
                "bytes": len(_dump_json_bytes(feat)),
            }
        )
    return infos, dropped


def classify_features(
    infos: List[Dict[str, Any]], region_box: BaseGeometry
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """只保留完全落在区域内的标注；跨边界 / 在区域外的剔除并给出原因。"""
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for info in infos:
        bx = info["bbox"]
        inside_bbox = (
            bx[0] >= region_box.bounds[0]
            and bx[2] <= region_box.bounds[2]
            and bx[1] >= region_box.bounds[1]
            and bx[3] <= region_box.bounds[3]
        )
        if inside_bbox and region_box.covers(info["geom"]):
            kept.append(info)
        elif info["geom"].intersects(region_box):
            dropped.append(
                {
                    "index": info["index"],
                    "classification": info["classification"],
                    "reason": "crosses_region_boundary",
                    "bbox": bx,
                    "points": info["points"],
                }
            )
        else:
            dropped.append(
                {
                    "index": info["index"],
                    "classification": info["classification"],
                    "reason": "outside_region",
                    "bbox": bx,
                    "points": info["points"],
                }
            )
    return kept, dropped


def build_geojson(kept: List[Dict[str, Any]], region: Dict[str, float]) -> Dict[str, Any]:
    """输出标准 FeatureCollection（保留原始坐标，含 bbox 外延字段）。"""
    x0, y0 = region["x"], region["y"]
    x1, y1 = x0 + region["width"], y0 + region["height"]
    return {
        "type": "FeatureCollection",
        "bbox": [x0, y0, x1, y1],
        "features": [info["feature"] for info in sorted(kept, key=lambda i: i["index"])],
    }


def self_check(
    geojson_path: Path, region: Dict[str, float], expected_features: int
) -> Dict[str, Any]:
    """回读输出文件做一次一致性校验，确保「可安全截取」。"""
    issues: List[str] = []
    try:
        with open(geojson_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001
        return {"passed": False, "issues": ["reload_failed: %s" % exc]}

    if data.get("type") != "FeatureCollection":
        issues.append("type != FeatureCollection")

    feats = data.get("features") or []
    if len(feats) != expected_features:
        issues.append("feature count %d != expected %d" % (len(feats), expected_features))

    x0, y0 = region["x"], region["y"]
    x1, y1 = x0 + region["width"], y0 + region["height"]
    region_box = shapely_box(x0, y0, x1, y1)

    n_checked, n_crossing, n_float_coords = 0, 0, 0
    for feat in feats:
        gd = feat.get("geometry")
        if not gd:
            issues.append("feature without geometry")
            continue
        try:
            geom = shapely_shape(gd)
        except Exception:  # noqa: BLE001
            issues.append("unparsable geometry in output")
            continue
        n_checked += 1
        bx = geom.bounds
        if not (
            bx[0] >= x0 and bx[2] <= x1 and bx[1] >= y0 and bx[3] <= y1
        ) or not region_box.covers(geom):
            n_crossing += 1
        if any(isinstance(v, float) and not float(v).is_integer() for v in _iter_flat(gd)):
            n_float_coords += 1

    if n_crossing:
        issues.append("%d feature(s) not fully inside region" % n_crossing)

    return {
        "passed": not issues,
        "issues": issues[:20],
        "features_checked": n_checked,
        "features_not_inside_region": n_crossing,
        "features_with_non_integer_coords": n_float_coords,
    }


def _iter_flat(geometry: Dict[str, Any]) -> Iterable[float]:
    stack: List[Any] = [geometry.get("coordinates")]
    while stack:
        node = stack.pop()
        if isinstance(node, (list, tuple)):
            if node and isinstance(node[0], (int, float)):
                yield from node
            else:
                stack.extend(node)


def _summarize_classes(infos: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for info in infos:
        key = info.get("classification") or "<未分类>"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------- 主流程
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="导出可安全截取的小型 GeoJSON + 对应 SVS 元数据（默认合计 < 50MB）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--svs", default=DEFAULT_SVS, help="输入 SVS 路径")
    p.add_argument("--geojson", default=DEFAULT_GEOJSON, help="输入 GeoJSON 路径")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR, help="输出目录")
    p.add_argument("--prefix", default=None, help="输出文件名前缀，默认取 GeoJSON 文件名")
    p.add_argument("--max-mb", type=float, default=DEFAULT_MAX_MB, help="输出合计体积上限(MB)")
    p.add_argument(
        "--mode",
        choices=("full", "auto", "manual"),
        default="full",
        help="区域选择：full=整张切片；auto=自动挑最大且不超预算的区域；manual=用 --region",
    )
    p.add_argument(
        "--region",
        default=None,
        help='manual 模式下的 level-0 像素区域，格式 "x,y,w,h"',
    )
    p.add_argument(
        "--on-overflow",
        choices=("auto", "keep"),
        default="auto",
        help="full/manual 区域超出体积上限时：auto=自动收缩；keep=只警告仍按原区域输出",
    )
    p.add_argument(
        "--max-dropped-in-report",
        type=int,
        default=2000,
        help="元数据中最多记录多少条被剔除的标注明细",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    svs_path = Path(args.svs)
    geojson_path = Path(args.geojson)
    outdir = Path(args.outdir)
    prefix = args.prefix or geojson_path.stem

    for p in (svs_path, geojson_path):
        if not p.is_file():
            print("[错误] 文件不存在：%s" % p, file=sys.stderr)
            return 2

    budget_bytes = int(args.max_mb * 1024 * 1024)
    manual_region: Optional[List[float]] = None
    if args.region:
        try:
            parts = [float(v) for v in args.region.replace(" ", "").split(",")]
            if len(parts) != 4:
                raise ValueError
            manual_region = parts
        except ValueError:
            print('[错误] --region 需要形如 "x,y,w,h" 的四个数字', file=sys.stderr)
            return 2

    # 1) SVS 元数据
    print("[1/5] 读取 SVS 元数据 ...")
    meta = read_svs_metadata(svs_path)
    slide_w, slide_h = meta["slide_dimensions_l0"]
    slide_box = shapely_box(0, 0, slide_w, slide_h)
    print(
        "      维度 %dx%d，%d 级金字塔，MPP=%s，物镜=%s"
        % (
            slide_w,
            slide_h,
            meta["level_count"],
            meta["mpp"]["x"],
            meta["objective_power"],
        )
    )

    # 2) GeoJSON 预处理
    print("[2/5] 读取并校验 GeoJSON ...")
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    if gj.get("type") != "FeatureCollection":
        print("[错误] 仅支持 FeatureCollection 类型的 GeoJSON", file=sys.stderr)
        return 2
    features: List[Dict[str, Any]] = gj.get("features") or []
    infos, pre_dropped = build_feature_infos(features, slide_box)
    print(
        "      共 %d 个要素，可用 %d 个，预处理剔除 %d 个"
        % (len(features), len(infos), len(pre_dropped))
    )

    # 3) 确定区域
    print("[3/5] 确定裁剪区域 (mode=%s) ..." % args.mode)
    region = choose_region(args.mode, manual_region, slide_w, slide_h, infos, budget_bytes)

    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = list(pre_dropped)
    if region is None:
        print("[错误] 在给定体积上限内找不到可用的安全区域，请提高 --max-mb", file=sys.stderr)
        return 3

    region_box = shapely_box(
        region["x"], region["y"], region["x"] + region["width"], region["y"] + region["height"]
    )
    kept, region_dropped = classify_features(infos, region_box)
    dropped += region_dropped
    out_obj = build_geojson(kept, region)
    payload = _dump_json_bytes(out_obj)

    # 体积超限 → 按需自动收缩
    if len(payload) > budget_bytes and args.on_overflow == "auto" and args.mode != "auto":
        print("      [警告] 当前区域输出 %.1fMB > 上限 %.1fMB，自动收缩区域 ..."
              % (len(payload) / 1048576.0, args.max_mb))
        shrunken = choose_region("auto", None, slide_w, slide_h, infos, budget_bytes)
        if shrunken is not None:
            region = shrunken
            region_box = shapely_box(
                region["x"],
                region["y"],
                region["x"] + region["width"],
                region["y"] + region["height"],
            )
            kept, region_dropped = classify_features(infos, region_box)
            dropped = list(pre_dropped) + region_dropped
            out_obj = build_geojson(kept, region)
            payload = _dump_json_bytes(out_obj)
            print("      收缩后区域 x=%d y=%d w=%d h=%d" % (
                region["x"], region["y"], region["width"], region["height"]))
        else:
            print("      [警告] 自动收缩失败，保留原区域输出")

    # 4) 写文件（原子）
    print("[4/5] 写出结果 ...")
    outdir.mkdir(parents=True, exist_ok=True)
    geojson_out = outdir / ("%s_small.geojson" % prefix)
    meta_out = outdir / ("%s_meta.json" % prefix)
    _atomic_write_bytes(geojson_out, payload)

    check = self_check(geojson_out, region, len(kept))

    keep_counts = _summarize_classes(kept)
    drop_counts: Dict[str, int] = {}
    for d in dropped:
        drop_counts[d["reason"]] = drop_counts.get(d["reason"], 0) + 1

    region_levels = _region_level_info(meta, region)
    meta_out_obj: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generator": "small_look.py",
        "source": {
            "svs": str(svs_path),
            "svs_size_bytes": svs_path.stat().st_size,
            "svs_mtime": datetime.fromtimestamp(svs_path.stat().st_mtime).isoformat(
                timespec="seconds"
            ),
            "geojson": str(geojson_path),
            "geojson_size_bytes": geojson_path.stat().st_size,
            "original_feature_count": len(features),
            "coordinate_space": {
                "units": "level0_pixels",
                "note": "GeoJSON 坐标与 SVS level-0 像素坐标一致，非地理经纬度（无 CRS）",
            },
        },
        "region": region,
        "region_in_levels": region_levels,
        "slide_dimensions_l0": meta["slide_dimensions_l0"],
        "level_count": meta["level_count"],
        "level_dimensions": meta["level_dimensions"],
        "level_downsamples": meta["level_downsamples"],
        "levels": meta["levels"],
        "mpp": meta["mpp"],
        "objective_power": meta["objective_power"],
        "vendor": meta["vendor"],
        "output": {
            "geojson_path": str(geojson_out),
            "geojson_size_bytes": len(payload),
            "feature_count": len(kept),
            "point_count": int(sum(i["points"] for i in kept)),
            "classes": keep_counts,
            "bbox": out_obj["bbox"],
        },
        "dropped": {
            "total": len(dropped),
            "by_reason": drop_counts,
            "by_classification": _summarize_classes(dropped),
            "items": dropped[: max(args.max_dropped_in_report, 0)],
            "items_truncated": max(0, len(dropped) - max(args.max_dropped_in_report, 0)),
        },
        "validation": {
            "safe_crop_semantics": (
                "只保留完全落在 region 内的标注；与边界相交或位于区域外的标注已剔除并记录"
            ),
            "self_check": check,
        },
        "openslide_properties": meta["openslide_properties"],
    }

    meta_payload = _dump_json_bytes(meta_out_obj)
    total = len(payload) + len(meta_payload)
    meta_out_obj["budget"] = {
        "max_bytes": budget_bytes,
        "max_mb": args.max_mb,
        "geojson_bytes": len(payload),
        "meta_bytes": len(meta_payload),
        "total_bytes": total,
        "within_budget": total <= budget_bytes,
    }
    _atomic_write_bytes(meta_out, _dump_json_bytes(meta_out_obj))

    # 5) 汇总
    print("[5/5] 完成")
    print("      GeoJSON : %s  (%.2f MB, %d 个要素)" % (geojson_out, len(payload) / 1048576.0, len(kept)))
    print("      元数据  : %s  (%.2f MB)" % (meta_out, meta_out.stat().st_size / 1048576.0))
    print("      区域    : x=%d y=%d w=%d h=%d (level-0 像素)" % (
        region["x"], region["y"], region["width"], region["height"]))
    print("      剔除    : %s" % (drop_counts or "无"))
    print("      自检    : %s" % ("通过" if check["passed"] else "失败 %s" % check["issues"]))
    if total > budget_bytes:
        print("      [警告] 输出合计 %.2fMB 超出上限 %.2fMB" % (total / 1048576.0, args.max_mb))
    return 0


if __name__ == "__main__":
    sys.exit(main())
