# -*- coding: utf-8 -*-
"""
demo.py — 比赛演示模式逻辑
============================
- 演示顺序：番茄 → 苹果 → 葡萄 → 密集场景
- 上一张 / 下一张 / 自动播放（Timer tick）/ 停止 / 重新检测当前案例
- 重新检测仅使用当前 VAL demo 图片，复用已加载模型（不重复加载），与 handlers 共用串行锁
"""

from __future__ import annotations

from web import backend
from web import config as web_config
from web.assets import DEMO_CASES, _friendly_case_name, resolve_demo_asset_path
from web.handlers import _INFER_LOCK
from web.render import _scene_case_info

# 演示顺序：番茄 → 苹果 → 葡萄 → 密集场景
_DEMO_ORDER = sorted(
    range(len(DEMO_CASES)),
    key=lambda i: (
        0 if DEMO_CASES[i]["scenario"] != "dense" else 1,   # 密集场景最后
        {"tomato": 0, "apple": 1, "grape": 2}.get(DEMO_CASES[i]["crop"], 9),
    ),
)


def _demo_case_view(index: int):
    """返回第 index 个演示案例的 (原图, AI检测结果, 信息 markdown)。

    演示模式只展示评委关心的四项：作物类型、病害类别、检测目标数、置信度/推理耗时；
    **隐藏 bbox 坐标与内部文件名**（bbox 仍保留在普通检测结果表中）。
    """
    if not DEMO_CASES:
        return None, None, "（暂无演示数据）"
    idx = index % len(_DEMO_ORDER)
    c = DEMO_CASES[_DEMO_ORDER[idx]]
    orig = resolve_demo_asset_path(c, "original")
    vis = resolve_demo_asset_path(c, "result")
    if orig is None or vis is None:
        return (None, None,
                f"（演示资源缺失：case={c['demo_id']}，未找到原图/检测结果图）")
    dets = c["detections"]
    n = len(dets)
    max_conf = max((d["confidence"] for d in dets), default=0.0)
    classes = sorted({d["class_name"] for d in dets})
    ms = c.get("inference_ms", 0.0)
    lines = [
        f"### 案例 {idx + 1} / {len(_DEMO_ORDER)} · {c['crop_cn']} · {c['scene_label']}",
        f"- **作物类型**：{c['crop_cn']}",
        f"- **病害类别**：{'、'.join(classes) if classes else '当前阈值下未检出'}",
        f"- **检测目标数**：{n}",
        f"- **最高置信度**：{max_conf:.2f} ｜ **推理耗时**：{ms:.0f} ms（真实计时）",
    ]
    return str(orig), str(vis), "\n".join(lines)


def _demo_prev(index: int):
    if not DEMO_CASES:
        return 0, None, None, "（暂无演示数据）"
    idx = (int(index) - 1) % len(_DEMO_ORDER)
    o, v, info = _demo_case_view(idx)
    return idx, o, v, info


def _demo_next(index: int):
    if not DEMO_CASES:
        return 0, None, None, "（暂无演示数据）"
    idx = (int(index) + 1) % len(_DEMO_ORDER)
    o, v, info = _demo_case_view(idx)
    return idx, o, v, info


def _demo_play():
    return True


def _demo_stop():
    return False


def _demo_timer_tick(index: int, playing: bool):
    """自动播放 tick：playing 时前进一张；停止时保持当前。"""
    if not DEMO_CASES:
        return 0, None, None, "（暂无演示数据）", False
    if playing:
        idx = (int(index) + 1) % len(_DEMO_ORDER)
        o, v, info = _demo_case_view(idx)
        return idx, o, v, info, True
    o, v, info = _demo_case_view(int(index))
    return int(index), o, v, info, False


def _demo_redetect(index: int):
    """重新检测当前演示案例：仅使用当前 VAL demo 图片；复用已加载模型，不重复加载。"""
    if not DEMO_CASES:
        return None, None, "（暂无演示数据）"
    c = DEMO_CASES[_DEMO_ORDER[int(index) % len(_DEMO_ORDER)]]
    img_path = resolve_demo_asset_path(c, "original")
    if img_path is None:
        return None, None, f"（演示资源缺失：case={c['demo_id']}，未找到原图）"
    import time as _time

    from inference.batch_infer import read_image
    from inference.postprocess import filter_by_conf
    from inference.visualize import draw_detections

    try:
        detector = backend.get_detector()
        img = read_image(img_path)
        with _INFER_LOCK:  # 串行推理
            t0 = _time.perf_counter()
            dets = detector.predict(img)
            t1 = _time.perf_counter()
            ms = (t1 - t0) * 1000.0
        dets = filter_by_conf(dets, 0.5)
        drawn = draw_detections(
            img, dets, detector.catid2name,
            catid2cn=web_config.CLASS_NAMES_CN,
            inference_seconds=t1 - t0,
        )
    except Exception as e:  # noqa: BLE001
        return str(img_path), None, f"重新检测失败：{e}"

    n = len(dets)
    max_conf = max((d.confidence for d in dets), default=0.0)
    classes = sorted({d.class_name for d in dets})
    lines = [
        f"### 🔄 重新检测 · {_friendly_case_name(c)} · {c['scene_label']}",
        f"- **作物类型**：{c['crop_cn']}",
        f"- **病害类别**：{'、'.join(classes) if classes else '当前阈值下未检出'}",
        f"- **检测目标数**：{n}",
        f"- **最高置信度**：{max_conf:.2f} ｜ **本次推理耗时**：{ms:.0f} ms（真实计时，复用已加载模型）",
    ]
    return str(img_path), drawn, "\n".join(lines)


def _demo_select(demo_id: str):
    """选择演示案例 -> 原图 / AI检测结果 / 友好详情（隐藏 bbox 与内部文件名）。"""
    for c in DEMO_CASES:
        if c["demo_id"] == demo_id:
            orig = resolve_demo_asset_path(c, "original")
            vis = resolve_demo_asset_path(c, "result")
            if orig is None or vis is None:
                return (None, None,
                        f"（演示资源缺失：case={demo_id}，未找到原图/检测结果图，请检查 web/demo_data 与 web/demo_outputs）")
            return str(orig), str(vis), _scene_case_info(c)
    return None, None, "未找到该演示案例。"
