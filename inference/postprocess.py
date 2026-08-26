# -*- coding: utf-8 -*-
"""
postprocess.py — 统一检测结果结构与统计聚合
============================================
定义后端无关的统一输出数据结构 `Detection`，并提供：
  - 置信度过滤
  - 每图统计（总目标数、各类别数量、各类别平均置信度）
  - 批次统计（每类出现图片数、目标总数、平均/最高置信度）
  - JSON / CSV / TXT 格式化

DetectorBackend（当前 PaddleBackend，后续 RKNNBackend）只负责返回
`list[list[Detection]]`，本模块对后端完全无关，方便后续替换。
"""

from __future__ import annotations

import csv
import io
from dataclasses import asdict, dataclass, field


@dataclass
class Detection:
    """统一的单目标检测结果（字段名与需求一一对应）。"""

    image_name: str      # 图片文件名（不含路径）
    class_id: int        # COCO 类别 id（1-13，来自 val.json，不重新编号）
    class_name: str      # 英文类别名（严格取自 val.json）
    confidence: float    # 置信度 [0,1]
    x: float             # bbox 左上角 x（原图坐标）
    y: float             # bbox 左上角 y（原图坐标）
    width: float         # bbox 宽
    height: float        # bbox 高

    def to_dict(self) -> dict:
        d = asdict(self)
        # 保留需求要求的 8 个字段顺序，便于阅读
        return {
            "image_name": d["image_name"],
            "class_id": d["class_id"],
            "class_name": d["class_name"],
            "confidence": round(float(d["confidence"]), 6),
            "x": round(float(d["x"]), 2),
            "y": round(float(d["y"]), 2),
            "width": round(float(d["width"]), 2),
            "height": round(float(d["height"]), 2),
        }


# JSON/CSV 统一表头（与 Detection.to_dict 的键一致）
RECORD_FIELDS = [
    "image_name",
    "class_id",
    "class_name",
    "confidence",
    "x",
    "y",
    "width",
    "height",
]


def filter_by_conf(detections: list[Detection], conf: float) -> list[Detection]:
    """按置信度阈值过滤，保留 confidence >= conf 的目标。"""
    conf = float(conf)
    return [d for d in detections if d.confidence >= conf]


def detections_to_records(detections: list[Detection]) -> list[dict]:
    """转换为 JSON 友好的 dict 列表。"""
    return [d.to_dict() for d in detections]


# ---------------------------------------------------------------------------
# 每图统计
# ---------------------------------------------------------------------------
def image_summary(image_name: str, detections: list[Detection]) -> dict:
    """单图统计：总目标数、各类别数量、各类别平均置信度。"""
    per_class: dict[int, dict] = {}
    for d in detections:
        p = per_class.setdefault(
            d.class_id, {"class_id": d.class_id, "class_name": d.class_name,
                         "count": 0, "conf_sum": 0.0}
        )
        p["count"] += 1
        p["conf_sum"] += float(d.confidence)

    class_stats = []
    for cid in sorted(per_class):
        p = per_class[cid]
        class_stats.append({
            "class_id": cid,
            "class_name": p["class_name"],
            "count": p["count"],
            "avg_confidence": round(p["conf_sum"] / p["count"], 6) if p["count"] else 0.0,
        })

    return {
        "image_name": image_name,
        "total_targets": len(detections),
        "per_class": class_stats,
    }


# ---------------------------------------------------------------------------
# 批次统计
# ---------------------------------------------------------------------------
def batch_class_statistics(image_results: list[dict]) -> list[dict]:
    """批次级别按类统计：出现图片数、目标总数、平均置信度、最高置信度。

    入参 image_results 为 [{'image_name', 'detections': [...]}, ...]。
    返回按 class_id 升序的列表。
    """
    acc: dict[int, dict] = {}
    for img in image_results:
        image_name = img["image_name"]
        seen_classes: set[int] = set()
        for d in img["detections"]:
            cid = int(d.class_id)
            a = acc.setdefault(cid, {
                "class_id": cid,
                "class_name": d.class_name,
                "num_images": set(),   # 该图片的去重集合
                "num_targets": 0,
                "conf_sum": 0.0,
                "max_conf": 0.0,
            })
            a["num_targets"] += 1
            a["conf_sum"] += float(d.confidence)
            a["max_conf"] = max(a["max_conf"], float(d.confidence))
            if image_name not in a["num_images"]:
                a["num_images"].add(image_name)
            seen_classes.add(cid)

    rows = []
    for cid in sorted(acc):
        a = acc[cid]
        rows.append({
            "class_id": cid,
            "class_name": a["class_name"],
            "num_images": len(a["num_images"]),
            "num_targets": a["num_targets"],
            "avg_confidence": round(a["conf_sum"] / a["num_targets"], 6),
            "max_confidence": round(a["max_conf"], 6),
        })
    return rows


# ---------------------------------------------------------------------------
# 文件输出辅助（JSON / CSV / TXT）
# ---------------------------------------------------------------------------
def write_json(path, data, indent: int = 2) -> None:
    import json

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def write_csv(path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def write_txt(path, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def detection_rows_for_csv(detections: list[Detection]) -> list[list]:
    """把 Detection 列表转换为 CSV 行（与 RECORD_FIELDS 对齐）。"""
    rows = []
    for d in detections:
        rec = d.to_dict()
        rows.append([rec[k] for k in RECORD_FIELDS])
    return rows


def render_image_txt(image_name: str, detections: list[Detection]) -> str:
    """简洁 TXT 统计：总目标数 + 每类数量。"""
    summary = image_summary(image_name, detections)
    buf = io.StringIO()
    buf.write(f"image: {image_name}\n")
    buf.write(f"total_targets: {summary['total_targets']}\n")
    for p in summary["per_class"]:
        buf.write(
            f"  class_id={p['class_id']} name={p['class_name']} "
            f"count={p['count']} avg_conf={p['avg_confidence']:.4f}\n"
        )
    return buf.getvalue()
