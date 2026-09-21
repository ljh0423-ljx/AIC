# -*- coding: utf-8 -*-
"""
charts.py — matplotlib 图表生成（类别统计柱状图 / 空状态占位图）
================================================================
- 中文字体自动检测（只执行一次）
- 柱状图深色主题，与比赛答辩 UI 协调
- 无数据时返回紧凑占位图，避免大片空白
本模块只依赖 matplotlib（惰性导入），不依赖 Web 其他模块。
"""

from __future__ import annotations

import logging

logger = logging.getLogger("agri.web.charts")

_CHINESE_FONT_READY = False
_CHINESE_FONT_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei",
    "Source Han Sans SC", "PingFang SC",
]


def _setup_matplotlib_font() -> None:
    """启动时自动检测系统可用中文字体并设置 matplotlib.rcParams（只执行一次）。

    修复 matplotlib 中文标题/坐标轴显示为方框乱码的问题。
    """
    global _CHINESE_FONT_READY
    if _CHINESE_FONT_READY:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.font_manager as fm

        names = {f.name for f in fm.fontManager.ttflist}
        chosen = next((c for c in _CHINESE_FONT_CANDIDATES if c in names), None)
        if chosen:
            matplotlib.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
        else:
            matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        logger.info("[matplotlib] 中文字体：%s", chosen or "未找到可用中文字体，回退 DejaVu Sans")
    except Exception as e:  # noqa: BLE001
        logger.warning("[matplotlib] 字体设置失败：%s", e)
    _CHINESE_FONT_READY = True


def _class_chart_figure(stats: list[dict]):
    """各病害类别检测数量柱状图（深色协调、中文字体、动态 Y 轴、单柱适中）。

    X=病害类别，Y=检测目标数，每类一根柱，柱顶数值；与「类别统计明细」表同一份数据。
    """
    _setup_matplotlib_font()
    if not stats:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(stats)
    # 高度 350~380px（不超过 380）
    height = 3.5 if n <= 3 else 3.8
    fig, ax = plt.subplots(figsize=(7.2, height))
    # 深色背景与当前比赛答辩 UI 协调（避免白色大块）
    bg = "#0f1a14"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.tick_params(colors="#cfe3d3")
    for sp in ax.spines.values():
        sp.set_color("#2e7d32")

    names = [str(s["class_name"]) for s in stats]
    counts = [int(s["num_targets"]) for s in stats]
    # 固定合理柱宽：单类 0.4（居中），多类 0.48/0.55（保持间距不挤压）
    width = 0.4 if n == 1 else (0.48 if n <= 4 else 0.55)
    bars = ax.bar(names, counts, width=width, color="#5aa96b", edgecolor="#3f7d52")
    # X 轴固定范围：单类别柱子始终居中，不随绘图区宽度自动拉伸
    ax.set_xlim(-0.5, max(n - 0.5, 0.5))
    ax.set_title("各病害类别检测数量", fontsize=12, color="#eafff1")
    ax.set_xlabel("病害类别", fontsize=10, color="#cfe3d3")
    ax.set_ylabel("检测目标数", fontsize=10, color="#cfe3d3")

    # 动态 Y 轴范围：max=1 -> 1.3；max>1 -> max*1.25（如 max=2 -> 2.5，max=4 -> 5）
    max_val = max(counts) if counts else 0
    ymax = 1.3 if max_val <= 1 else max_val * 1.25
    ax.set_ylim(0, ymax)

    # 柱顶数值
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, cnt + ymax * 0.03, str(cnt),
                ha="center", va="bottom", fontsize=9, color="#eafff1")

    # 轻网格（不过重）
    ax.grid(axis="y", alpha=0.2, linestyle="--", color="#3a5c46")
    ax.set_axisbelow(True)
    if n > 4:
        ax.tick_params(axis="x", rotation=30, labelsize=9)
    else:
        ax.tick_params(axis="x", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _class_empty_figure():
    """无数据时的紧凑占位图（不渲染大尺寸空 Plot，避免大片空白）。"""
    _setup_matplotlib_font()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 0.9))
    bg = "#0f1a14"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.axis("off")
    ax.text(0.5, 0.5, "暂无检测结果（图表）", ha="center", va="center",
            fontsize=11, color="#8fa896")
    fig.tight_layout()
    return fig
