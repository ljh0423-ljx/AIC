# -*- coding: utf-8 -*-
"""
web/services/severity.py — 病害严重程度评估模块（阶段1：基于规则的评估）
======================================================================
阶段1（当前）：不修改模型、不重训练，仅基于已有检测结果的 bbox / 图片尺寸 / confidence
             做规则评估：
    infection_ratio = 检测框并集面积 / 图片面积（重叠框不重复计数，上限 1.0）
    等级阈值：infection_ratio < 0.10 → 轻度；0.10 <= ratio < 0.30 → 中度；>= 0.30 → 重度
    无检测目标时判定为「健康」。

阶段2（预留，可扩展）：保持 calculate_severity 对外签名不变，内部替换为：
    - 病斑分割模型（像素级感染面积占比，更精确）
    - 深度学习严重程度分类模型（直接输出等级）
只需实现 SeverityAssessor.assess() 即可，Web 层与其余模块零改动。

本模块为纯函数 / 无状态规则评估，不依赖 Paddle、不依赖 inference 内部实现；
接受 Detection 对象或 dict（x/y/width/height/confidence/class_name）两种形式。
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 等级与阈值（阶段1 规则常量）
# ---------------------------------------------------------------------------
SEVERITY_HEALTHY = "健康"
SEVERITY_MILD = "轻度"
SEVERITY_MODERATE = "中度"
SEVERITY_SEVERE = "重度"

MILD_THRESHOLD = 0.10
MODERATE_THRESHOLD = 0.30

RECOMMENDATIONS = {
    SEVERITY_HEALTHY: "未检测到明显病害区域，作物状态良好，建议持续观察。",
    SEVERITY_MILD: "侵染面积占比较低，建议加强田间巡查、保持通风透光，必要时预防性管理。",
    SEVERITY_MODERATE: "侵染面积中等，建议及时针对性施药，并摘除病叶以控制扩散。",
    SEVERITY_SEVERE: "侵染面积较大，建议立即采取药剂防治、清理病残体，并咨询植保专业意见。",
}


# ---------------------------------------------------------------------------
# 规则函数
# ---------------------------------------------------------------------------
def severity_level(infection_ratio: float) -> str:
    """按感染面积占比划分等级（阶段1 规则）。"""
    if infection_ratio < MILD_THRESHOLD:
        return SEVERITY_MILD
    if infection_ratio < MODERATE_THRESHOLD:
        return SEVERITY_MODERATE
    return SEVERITY_SEVERE


def recommendation_for(severity: str) -> str:
    """返回对应等级的规则化建议。"""
    return RECOMMENDATIONS.get(severity, "请结合田间实际情况综合判断。")


# ---------------------------------------------------------------------------
# 检测结果归一化（兼容 Detection 对象与 dict）
# ---------------------------------------------------------------------------
def _get(det, key: str, default=0.0):
    if isinstance(det, dict):
        return det.get(key, default)
    return getattr(det, key, default)


# ---------------------------------------------------------------------------
# 阶段1 规则评估实现（后续可整体替换为分割 / 深度学习模型）
# ---------------------------------------------------------------------------
def _clip_box(x, y, w, h, image_width, image_height) -> tuple:
    """把 bbox 钳制到图片范围内，返回 (x0, y0, x1, y1)。"""
    x0 = max(0.0, min(float(x), image_width))
    y0 = max(0.0, min(float(y), image_height))
    x1 = max(0.0, min(float(x) + max(0.0, float(w)), image_width))
    y1 = max(0.0, min(float(y) + max(0.0, float(h)), image_height))
    return x0, y0, x1, y1


def _union_rect_area(rects) -> float:
    """axis-aligned 矩形并集面积（坐标压缩 + 网格，精确，避免重叠框重复计数）。"""
    if not rects:
        return 0.0
    xs = sorted({x for r in rects for x in (r[0], r[2])})
    ys = sorted({y for r in rects for y in (r[1], r[3])})
    area = 0.0
    for i in range(len(xs) - 1):
        x0, x1 = xs[i], xs[i + 1]
        cx = (x0 + x1) / 2.0
        for j in range(len(ys) - 1):
            y0, y1 = ys[j], ys[j + 1]
            cy = (y0 + y1) / 2.0
            if any(r[0] <= cx <= r[2] and r[1] <= cy <= r[3] for r in rects):
                area += (x1 - x0) * (y1 - y0)
    return area


def _rule_based_assess(detections, image_width: float, image_height: float) -> dict:
    """阶段1：基于检测框面积占比的规则评估，返回统一严重程度评估结果。

    infection_ratio = 检测框「并集」面积 / 图片面积（重叠框不重复计数，上限 1.0）。
    """
    dets = list(detections or [])
    iw = max(1.0, float(image_width))
    ih = max(1.0, float(image_height))
    image_area = iw * ih

    # 无检测目标 → 健康
    if not dets:
        return {
            "class_name": "",
            "confidence": 0.0,
            "bbox": "",
            "infection_ratio": 0.0,
            "severity": SEVERITY_HEALTHY,
            "recommendation": recommendation_for(SEVERITY_HEALTHY),
        }

    # 主导类别 = 面积之和最大的类别；置信度取该类别最高
    area_by_class: dict[str, float] = {}
    conf_by_class: dict[str, float] = {}
    rects: list[tuple] = []
    xs: list[float] = []
    ys: list[float] = []
    x2s: list[float] = []
    y2s: list[float] = []
    for d in dets:
        cls = str(_get(d, "class_name", "") or "unknown")
        x = float(_get(d, "x", 0.0))
        y = float(_get(d, "y", 0.0))
        w = float(_get(d, "width", 0.0))
        h = float(_get(d, "height", 0.0))
        conf = float(_get(d, "confidence", 0.0))
        area_by_class[cls] = area_by_class.get(cls, 0.0) + max(0.0, w) * max(0.0, h)
        conf_by_class[cls] = max(conf_by_class.get(cls, 0.0), conf)
        x0, y0, x1, y1 = _clip_box(x, y, w, h, iw, ih)
        rects.append((x0, y0, x1, y1))
        xs.append(x0)
        ys.append(y0)
        x2s.append(x1)
        y2s.append(y1)

    dominant = max(area_by_class, key=lambda k: area_by_class[k])
    confidence = conf_by_class[dominant]

    # 合并框（所有检测框的外接矩形）
    x0, y0 = min(xs), min(ys)
    x1, y1 = max(x2s), max(y2s)
    bbox_str = f"{x0:.0f},{y0:.0f},{x1 - x0:.0f},{y1 - y0:.0f}"

    infected_area = _union_rect_area(rects)
    infection_ratio = min(1.0, infected_area / image_area)
    severity = severity_level(infection_ratio)

    return {
        "class_name": dominant,
        "confidence": round(confidence, 4),
        "bbox": bbox_str,
        "infection_ratio": round(infection_ratio, 6),
        "severity": severity,
        "recommendation": recommendation_for(severity),
    }


class SeverityAssessor:
    """病害严重程度评估器。

    阶段1：基于规则的评估（见 _rule_based_assess）。
    阶段2（预留）：可替换为病斑分割模型（像素级感染占比）或深度学习
                   严重程度分类模型；只需实现 assess() 并保持签名一致。
    """

    def assess(self, detections, image_width: float, image_height: float) -> dict:
        return _rule_based_assess(detections, image_width, image_height)


_default_assessor = SeverityAssessor()


def calculate_severity(detections, image_width, image_height) -> dict:
    """对外统一入口：计算单张图片的病害严重程度评估结果。

    参数：
        detections   该图片的检测结果（Detection 对象列表或 dict 列表）
        image_width  图片宽度（像素）
        image_height 图片高度（像素）

    返回：
        {
            "class_name":      主导病害类别英文名（无检测为空字符串）
            "confidence":      该类别最高置信度
            "bbox":            合并检测框 "x,y,w,h"
            "infection_ratio": 感染面积占比 [0,1]
            "severity":        健康 / 轻度 / 中度 / 重度
            "recommendation":  规则化建议
        }
    """
    return _default_assessor.assess(detections, image_width, image_height)
