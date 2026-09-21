# -*- coding: utf-8 -*-
"""
web/services/flowchart.py — 诊断流程图 / 可视化记录图生成（横版 4×2 网格信息图）
==============================================================================
复用当前会话（gr.State）已有检测/VLM/RAG/Agent 结果，生成「本次诊断」流程图
（PNG + PDF）。不重复执行检测、不调用任何付费 API。

节点顺序（保持不变，4 行 × 2 列网格）：
  第1行 原始图片 → PP-YOLOE检测
  第2行 检测框面积代理评估 → VLM视觉分析
  第3行 RAG知识检索 → Agent综合建议
  第4行 语音交互状态 → 验收报告状态

风格：深色农业科技风 + 横版信息图。每个节点为左右分栏卡片：
  左：步骤编号 + 标题 + 一行摘要 + 2~4 个关键字段（短字段，非长句）；
  右：状态 badge + 模型 badge +（仅①②）统一尺寸带边框缩略图。
底部「本次诊断结论摘要」汇总栏 + 免责声明。
严格区分「已执行 / 真实 / 真实向量检索 / 真实 LLM / Mock 回退 / 本次未执行」；
检测框面积占比标注为规则代理指标，VLM 标注为可见特征分析（非病原学确诊），无检出 ≠ 确认健康。
导出文件存 web/tmp/flowcharts/（UUID 命名，TTL 清理，会话隔离，清空重置）。
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无 GUI 后端

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

_FLOW_DIR = Path(__file__).resolve().parent.parent / "tmp" / "flowcharts"
_FONT_PATH = "C:/Windows/Fonts/simhei.ttf"
_TTL_HOURS = 24.0

# ---------------------------------------------------------------------------
# 深色农业科技风配色（与 Web 一致）
# ---------------------------------------------------------------------------
_BG = "#0d1b13"
_NODE_BG = "#16241c"
_NODE_EDGE = "#2c4a35"
_TITLE = "#8ee09a"
_TEXT = "#eaf4ec"
_DIM = "#93ab9d"
_ARROW = "#4f9c5d"

# 状态 badge（填充圆角）: kind -> (背景, 文字)
_BADGE = {
    "exec": ("#1f4a30", "#7ee99a"),   # 已执行
    "real": ("#1f4a30", "#7ee99a"),   # 真实 / 真实模型 / 真实向量检索 / 真实 LLM
    "mock": ("#4a3a14", "#e0b14f"),   # Mock 回退
    "skip": ("#33404a", "#8fa0b0"),   # 本次未执行
    "rule": ("#123f3f", "#7fd6d6"),   # 规则代理
}

# 模块组左侧强调色（轻量视觉区分）
_ACCENT = {
    "io": "#5cbf6a",       # 输入 / 检测
    "analysis": "#4fb6b0",  # 分析（面积代理 / VLM / RAG）
    "decision": "#e0b14f",  # 决策（Agent）
    "status": "#8fa0b0",    # 状态（语音 / 报告）
}


def _register_font() -> None:
    try:
        if not any(f.name == "SimHei" for f in font_manager.fontManager.ttflist):
            font_manager.fontManager.addfont(_FONT_PATH)
    except Exception:  # noqa: BLE001
        pass
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def _crop_of(class_name: str) -> str:
    n = (class_name or "").lower()
    if n.startswith("tomato"):
        return "番茄"
    if n.startswith("apple"):
        return "苹果"
    if n.startswith("grape"):
        return "葡萄"
    return "未确定"


def _trunc(s, n: int = 18) -> str:
    s = str(s or "")
    return s if len(s) <= n else s[: n - 1] + "…"


def cleanup_old_flowcharts() -> None:
    try:
        if not _FLOW_DIR.is_dir():
            return
        cutoff = time.time() - _TTL_HOURS * 3600
        for p in _FLOW_DIR.glob("*.*"):
            if p.suffix.lower() in (".png", ".pdf") and p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# 节点数据采集（8 节点，顺序不变）
# ---------------------------------------------------------------------------
def _collect_nodes(session_state: dict) -> tuple[list[dict], list[tuple]]:
    """从会话上下文提取 8 个流程节点 + 底部摘要，返回 (nodes, summary)。"""
    sev = session_state.get("severity_result") or {}
    vlm_r = session_state.get("vlm_result") or {}
    rag_r = session_state.get("rag_result") or {}
    agent_d = session_state.get("agent_decision") or {}
    dets = session_state.get("detection_result") or []

    disease = sev.get("class_name") or ""
    crop = _crop_of(disease)
    severity = sev.get("severity") or "—"
    ratio = float(sev.get("infection_ratio", 0)) * 100
    det_count = len(dets)

    vlm_real = bool(vlm_r.get("model"))
    rag_real = rag_r.get("retrieval_status") == "ok"
    agent_real = agent_d.get("backend_name") == "llm"
    voice_done = bool(session_state.get("voice_done"))
    report_done = bool(session_state.get("report_done"))

    if dets:
        from collections import Counter
        cnt = Counter(d.get("class_name", "") for d in dets)
        top = "、".join(k for k, _ in cnt.most_common(3))
        det_field = _trunc(top, 22) if top else "—"
    else:
        det_field = "未检出（≠确认健康）"

    nodes = [
        {
            "num": "①", "title": "原始图片", "group": "io",
            "summary": "本次诊断输入",
            "fields": [
                ("文件", _trunc(session_state.get("image_name") or "（无）", 16)),
                ("作物", crop),
            ],
            "badges": [("已执行" if session_state.get("image_name") else "本次未执行",
                        "exec" if session_state.get("image_name") else "skip")],
            "thumb": session_state.get("orig_path"),
            "thumb_label": "输入图",
            "exec": "exec" if session_state.get("image_name") else "skip",
        },
        {
            "num": "②", "title": "PP-YOLOE 检测", "group": "io",
            "summary": "PP-YOLOE+-m 推理",
            "fields": [
                ("目标数", f"{det_count} 个"),
                ("主要类别", det_field),
            ],
            "badges": [("已执行", "exec"), ("真实模型", "real")],
            "thumb": session_state.get("vis_path"),
            "thumb_label": "检测图",
            "exec": "exec",
        },
        {
            "num": "③", "title": "检测框面积代理评估", "group": "analysis",
            "summary": "规则代理指标",
            "fields": [
                ("面积占比", f"{ratio:.1f}%"),
                ("严重程度", severity),
                ("检测框", _trunc(str(sev.get("bbox", "—")), 20)),
            ],
            "badges": [("已执行", "exec"), ("规则代理", "rule")],
            "exec": "exec",
        },
        {
            "num": "④", "title": "VLM 视觉分析", "group": "analysis",
            "summary": "图像可见特征分析",
            "fields": [
                ("特征", _trunc(vlm_r.get("visual_evidence") or vlm_r.get("disease_description") or "—", 18)),
                ("模型", _trunc(vlm_r.get("model") or "未配置", 18)),
            ],
            "badges": [("真实" if vlm_real else "Mock 回退", "real" if vlm_real else "mock")],
            "exec": "exec" if vlm_real else "mock",
        },
        {
            "num": "⑤", "title": "RAG 知识检索", "group": "analysis",
            "summary": "农业知识库检索",
            "fields": [
                ("状态", rag_r.get("retrieval_status") or "未知"),
                ("来源", f"{len(rag_r.get('sources', []) or [])} 条"),
                ("方式", "缓存复用" if rag_r.get("cache_hit") else "实际检索"),
            ],
            "badges": [("真实向量检索" if rag_real else "Mock 回退", "real" if rag_real else "mock")],
            "exec": "exec" if rag_real else "mock",
        },
        {
            "num": "⑥", "title": "Agent 综合建议", "group": "decision",
            "summary": "综合决策建议",
            "fields": [
                ("优先级", agent_d.get("priority") or "—"),
                ("处置", _trunc(agent_d.get("action") or agent_d.get("reason") or "—", 20)),
                ("模型", _trunc(agent_d.get("model_name") or agent_d.get("backend_name") or "未配置", 16)),
            ],
            "badges": [("真实 LLM" if agent_real else "Mock 回退", "real" if agent_real else "mock")],
            "exec": "exec" if agent_real else "mock",
        },
        {
            "num": "⑦", "title": "语音交互状态", "group": "status",
            "summary": "语音问答",
            "fields": [
                ("状态", "已发起" if voice_done else "本次未发起"),
            ],
            "badges": [("已执行" if voice_done else "本次未执行", "exec" if voice_done else "skip")],
            "exec": "exec" if voice_done else "skip",
        },
        {
            "num": "⑧", "title": "验收报告状态", "group": "status",
            "summary": "诊断验收报告",
            "fields": [
                ("状态", "已生成 PDF" if report_done else "本次未生成"),
            ],
            "badges": [("已执行" if report_done else "本次未执行", "exec" if report_done else "skip")],
            "exec": "exec" if report_done else "skip",
        },
    ]

    # 底部「本次诊断结论摘要」：(标签, 值, 配色 kind)
    summary = [
        ("作物", crop, "plain"),
        ("病害", _trunc(disease, 14), "plain"),
        ("目标数", f"{det_count} 个", "plain"),
        ("严重程度", severity, "plain"),
        ("VLM", "真实" if vlm_real else "Mock", "real" if vlm_real else "mock"),
        ("RAG", "真实向量" if rag_real else "Mock", "real" if rag_real else "mock"),
        ("Agent", "真实 LLM" if agent_real else "Mock 规则", "real" if agent_real else "mock"),
        ("语音", "已执行" if voice_done else "未执行", "exec" if voice_done else "skip"),
        ("报告", "已生成" if report_done else "未生成", "exec" if report_done else "skip"),
    ]
    return nodes, summary


# ---------------------------------------------------------------------------
# 绘制辅助
# ---------------------------------------------------------------------------
def _badge(ax, x, y, text: str, kind: str, ha="center", fs: float = 9.2):
    bg, fg = _BADGE.get(kind, _BADGE["skip"])
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=fg,
            bbox=dict(boxstyle="round,pad=0.30", fc=bg, ec="none", lw=0, alpha=0.95))


def _draw_thumb(ax, path, cx, cy, size: float = 0.70, label: str = ""):
    """统一尺寸、带边框、带标题的缩略图（仅节点①②）。"""
    if not path or not Path(path).is_file():
        return
    try:
        from PIL import Image
        import numpy as np
        im = Image.open(path).convert("RGB")
        im.thumbnail((300, 300))
        arr = np.asarray(im)
        h, w = arr.shape[:2]
    except Exception:  # noqa: BLE001
        return
    half = size / 2
    border = FancyBboxPatch((cx - half, cy - half), size, size,
                            boxstyle="round,pad=0.01,rounding_size=0.05",
                            fc="#0a1510", ec=_NODE_EDGE, lw=1.2, zorder=4)
    ax.add_patch(border)
    scale = (size * 0.86) / max(h, w)
    dw, dh = w * scale, h * scale
    ax.imshow(arr, extent=[cx - dw / 2, cx + dw / 2, cy - dh / 2, cy + dh / 2],
              aspect="auto", zorder=5, interpolation="bilinear")
    if label:
        ax.text(cx, cy - half - 0.09, label, ha="center", va="top",
                fontsize=7.5, color=_DIM)


# ---------------------------------------------------------------------------
# 绘制主流程（横版 4×2 网格）
# ---------------------------------------------------------------------------
# 网格几何常量（画布 16×11，横版）
_CARD_W = 6.75
_CARD_H = 1.52
_LEFT_X = 0.55
_RIGHT_X = 8.70
_ROW_TOPS = [9.68, 7.60, 5.52, 3.44]   # 4 行卡片顶部 y
_ROW_GAP = 0.56                          # 行间距（行间连接箭头空间）


def _draw_card(ax, nd: dict, x0: float, y_top: float):
    """在 (x0, y_top) 绘制一个左右分栏的紧凑卡片。"""
    y_bot = y_top - _CARD_H
    ax.add_patch(FancyBboxPatch((x0, y_bot), _CARD_W, _CARD_H,
                                boxstyle="round,pad=0.05,rounding_size=0.12",
                                fc=_NODE_BG, ec=_NODE_EDGE, lw=1.2))
    # 左侧模块色强调条
    ax.add_patch(Rectangle((x0 + 0.05, y_bot + 0.13), 0.06, _CARD_H - 0.26,
                           fc=_ACCENT[nd["group"]], ec="none", zorder=3))

    # —— 左栏：编号 + 标题 + 摘要 + 字段 ——
    tx = x0 + 0.30
    ax.text(tx, y_top - 0.32, nd["num"], ha="left", va="center",
            fontsize=13, color=_ACCENT[nd["group"]], fontweight="bold")
    ax.text(tx + 0.62, y_top - 0.32, nd["title"], ha="left", va="center",
            fontsize=12.5, color=_TITLE, fontweight="bold")
    ax.text(tx, y_top - 0.66, nd["summary"], ha="left", va="center",
            fontsize=9.3, color=_DIM)
    fy = y_top - 0.95
    for label, value in nd["fields"]:
        ax.text(tx, fy, label, ha="left", va="center", fontsize=9.2, color=_DIM)
        ax.text(tx + 1.62, fy, _trunc(str(value), 20), ha="left", va="center",
                fontsize=9.5, color=_TEXT)
        fy -= 0.28

    # —— 右栏：状态/模型 badge（竖排，右对齐）+ 缩略图 ——
    rx = x0 + _CARD_W - 0.24
    for bi, (text, kind) in enumerate(nd["badges"]):
        _badge(ax, rx, y_top - 0.32 - bi * 0.33, text, kind, ha="right", fs=9.2)
    if nd.get("thumb"):
        _draw_thumb(ax, nd["thumb"], x0 + 5.42, y_top - 0.98, size=0.70,
                    label=nd.get("thumb_label", ""))


def _draw(nodes: list[dict], summary: list[tuple], out_path: str, fmt: str) -> str:
    _register_font()
    cleanup_old_flowcharts()

    fig = plt.figure(figsize=(16, 11), facecolor=_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis("off")

    # ---- 标题区 ----
    ax.text(8, 10.55, "智农慧眼 · 本次诊断流程图", ha="center", va="center",
            fontsize=18, color=_TITLE, fontweight="bold")
    ax.text(8, 10.15, "系统运行验收记录 · 不等同于病害专业确诊",
            ha="center", va="center", fontsize=10, color=_DIM)

    # ---- 4 行 × 2 列节点 ----
    for r in range(4):
        y_top = _ROW_TOPS[r]
        y_mid = y_top - _CARD_H / 2
        # 左卡 + 右卡
        _draw_card(ax, nodes[r * 2], _LEFT_X, y_top)
        _draw_card(ax, nodes[r * 2 + 1], _RIGHT_X, y_top)
        # 行内箭头（左→右）
        ax.add_patch(FancyArrowPatch(
            (_LEFT_X + _CARD_W, y_mid), (_RIGHT_X, y_mid),
            arrowstyle="-|>", mutation_scale=20, lw=2.4, color=_ARROW))
        # 行间连接（右卡底部 → 下一行左卡顶部，弧线回绕）
        if r < 3:
            ax.add_patch(FancyArrowPatch(
                (_RIGHT_X + _CARD_W / 2, y_top - _CARD_H),
                (_LEFT_X + _CARD_W / 2, _ROW_TOPS[r + 1]),
                connectionstyle="arc3,rad=-0.22", arrowstyle="-|>",
                mutation_scale=20, lw=2.0, color=_ARROW))

    # ---- 底部「本次诊断结论摘要」汇总栏（位于第④行节点下方，留足间距）----
    s_top = 1.50
    s_h = 1.05
    s_bot = s_top - s_h
    ax.add_patch(FancyBboxPatch((_LEFT_X, s_bot), (_RIGHT_X + _CARD_W) - _LEFT_X, s_h,
                                boxstyle="round,pad=0.05,rounding_size=0.14",
                                fc="#122018", ec="#2c4a35", lw=1.3))
    ax.text(_LEFT_X + 0.22, s_top - 0.34, "本次诊断结论摘要", ha="left", va="center",
            fontsize=12, color=_TITLE, fontweight="bold")
    for idx, (label, value, kind) in enumerate(summary):
        x = _LEFT_X + 0.28 + idx * 1.60
        color = _TEXT if kind == "plain" else _BADGE.get(kind, _BADGE["skip"])[1]
        ax.text(x, s_top - 0.74, f"{label}：{_trunc(str(value), 10)}",
                ha="left", va="center", fontsize=9.3, color=color,
                fontweight="bold" if kind != "plain" else "normal")

    # ---- 底部免责声明（与摘要区明显分开，最底部独立一行）----
    ax.text(8, 0.20, "※ 检测框面积占比为规则代理指标（不认定真实感染面积）；"
            "VLM 为图像可见特征分析；本记录为系统运行验收，不等同于病害专业确诊。",
            ha="center", va="center", fontsize=9, color=_DIM)

    _FLOW_DIR.mkdir(parents=True, exist_ok=True)
    out = _FLOW_DIR / out_path
    fig.savefig(out, format=fmt, facecolor=_BG, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return str(out)


def generate_flowchart(session_state: dict) -> tuple[str, str]:
    """生成流程图，返回 (PNG 路径, PDF 路径)。"""
    nodes, summary = _collect_nodes(session_state)
    uid = uuid.uuid4().hex[:8].upper()
    png = _draw(nodes, summary, f"flow_{uid}.png", "png")
    pdf = _draw(nodes, summary, f"flow_{uid}.pdf", "pdf")
    return png, pdf
