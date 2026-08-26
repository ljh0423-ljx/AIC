# -*- coding: utf-8 -*-
"""
visualize.py — 检测结果可视化（答辩展示用）
=============================================
  - 在**原始尺寸**图像上绘制 bbox（保持原始图片比例，不拉伸变形）
  - 每框显示：类别中文名(如有) + 英文名 + 置信度
  - 左上角标题：'Agricultural Disease Detection' + 'PP-YOLOE+-m'
  - 显示检测目标总数与实际推理耗时（真实计时，不虚构）
  - 中文渲染：优先使用支持中文的字体（工程自带 simfang.ttf / Windows 系统字体）

类别中文名：项目当前没有官方中文映射，默认仅显示英文；
若用户在 config.CLASS_NAMES_CN 中填写中文，将自动显示"中文 英文 置信度"。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except Exception:  # noqa: BLE001
    _PIL_AVAILABLE = False

from . import config as app_config
from .postprocess import Detection


# ---------------------------------------------------------------------------
# 字体查找（不写死 Windows 路径，全部由 Path 构造）
# ---------------------------------------------------------------------------
def find_font() -> Optional[str]:
    """查找一个支持中文的 ttf/ttc 字体文件路径；找不到返回 None。"""
    candidates: list[Path] = []
    # 1) 工程内置字体（PaddleDetection 自带 simfang.ttf，支持中文，跨平台可用）
    candidates.append(
        app_config.PADDLE_DETECTION_DIR / "ppdet" / "utils" / "simfang.ttf"
    )
    # 2) 本模块同级 fonts/ 目录（用户可自行放置字体）
    candidates.append(Path(__file__).resolve().parent / "fonts" / "simhei.ttf")
    # 3) Windows 系统字体（通过环境变量构造，避免硬编码盘符）
    windir = os.environ.get("WINDIR")
    if windir:
        for name in (
            "msyh.ttc",
            "msyhbd.ttc",
            "simhei.ttf",
            "simsun.ttc",
            "simfang.ttf",
        ):
            candidates.append(Path(windir) / "Fonts" / name)
    for p in candidates:
        try:
            if p.is_file():
                return str(p)
        except OSError:
            continue
    return None


# 固定调色板（按类别 id 取色，保证同一类别颜色稳定）
_COLOR_PALETTE: list[tuple[int, int, int]] = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 190), (0, 128, 128), (230, 190, 255),
    (170, 110, 40),
]


def _load_font(size: int, font_path: Optional[str]):
    if _PIL_AVAILABLE and font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:  # noqa: BLE001
            pass
    try:
        return ImageFont.load_default()
    except Exception:  # noqa: BLE001
        return None


def draw_detections(
    image_bgr: np.ndarray,
    detections: list[Detection],
    catid2name: dict[int, str],
    catid2cn: Optional[dict[int, str]] = None,
    inference_seconds: Optional[float] = None,
    title: Optional[str] = None,
    model_tag: Optional[str] = None,
    font_path: Optional[str] = None,
) -> np.ndarray:
    """在原始尺寸 BGR 图像上绘制检测框与标注，返回 BGR 图像（尺寸不变）。

    参数：
      image_bgr        原始 BGR 图像（cv2 读取）
      detections       已按置信度过滤后的 Detection 列表
      catid2name       COCO id -> 英文名
      catid2cn         COCO id -> 中文名（可为空，空则只显示英文）
      inference_seconds 实际推理耗时（秒），用于信息栏展示
      title/model_tag   左上角标题与模型名
    """
    if not _PIL_AVAILABLE:
        raise RuntimeError(
            "[环境错误] 未安装 Pillow，无法进行可视化。"
            "请执行：pip install -r requirements_infer.txt"
        )

    catid2cn = catid2cn or {}
    title = title or app_config.APP_TITLE
    model_tag = model_tag or app_config.MODEL_TAG

    # BGR -> RGB（PIL 使用 RGB）
    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    title_font = _load_font(28, font_path)
    tag_font = _load_font(20, font_path)
    info_font = _load_font(20, font_path)
    label_font = _load_font(18, font_path)

    # ---- 左上角标题区 ----
    _draw_text_with_bg(
        draw, (8, 8), title, title_font, fill=(255, 255, 255), bg=(0, 0, 0)
    )
    _draw_text_with_bg(
        draw, (10, 40), model_tag, tag_font, fill=(255, 215, 0), bg=(0, 0, 0)
    )
    # ---- 信息栏：检测目标总数 + 实际推理耗时 ----
    if inference_seconds is not None:
        info_text = f"Objects: {len(detections)}   Inference: {inference_seconds:.2f} s"
    else:
        info_text = f"Objects: {len(detections)}"
    _draw_text_with_bg(
        draw, (10, 66), info_text, info_font, fill=(0, 255, 0), bg=(0, 0, 0)
    )

    # ---- 逐框绘制 ----
    img_w, img_h = pil_img.size
    for d in detections:
        x0, y0 = float(d.x), float(d.y)
        x1, y1 = x0 + float(d.width), y0 + float(d.height)
        # 钳制到图像范围内（模型偶尔会输出略微越界框）
        x0c = max(0.0, min(x0, img_w - 1))
        y0c = max(0.0, min(y0, img_h - 1))
        x1c = max(0.0, min(x1, img_w - 1))
        y1c = max(0.0, min(y1, img_h - 1))

        color = _COLOR_PALETTE[d.class_id % len(_COLOR_PALETTE)]
        draw.rectangle(
            [x0c, y0c, x1c, y1c], outline=color, width=2
        )

        # 标签文本：中文(如有) + 英文 + 置信度
        cn = catid2cn.get(d.class_id, "")
        parts = []
        if cn:
            parts.append(cn)
        parts.append(catid2name.get(d.class_id, f"class_{d.class_id}"))
        parts.append(f"{d.confidence:.2f}")
        label = " ".join(parts)

        # 标签绘制在框左上角上方（贴边），保证可见
        text_w = _text_width(label, label_font)
        bx0 = x0c
        by0 = y0c - 22 if y0c >= 22 else y0c
        bx1 = min(bx0 + text_w + 8, img_w - 1)
        draw.rectangle([bx0, by0, bx1, by0 + 20], fill=color)
        draw.text((bx0 + 4, by0 + 2), label, font=label_font, fill=(0, 0, 0))

    return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)


def _draw_text_with_bg(draw, pos, text, font, fill, bg):
    x, y = pos
    try:
        w, h = draw.textsize(text, font=font)
    except Exception:  # noqa: BLE001
        try:
            bbox = draw.textbbox((x, y), text, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:  # noqa: BLE001
            w, h = len(text) * 14, 20
    draw.rectangle([x, y, x + w + 6, y + h + 4], fill=bg)
    draw.text((x + 3, y + 2), text, font=font, fill=fill)


def _text_width(text: str, font):
    try:
        img_tmp = Image.new("RGB", (10, 10))
        d = ImageDraw.Draw(img_tmp)
        bbox = d.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]
    except Exception:  # noqa: BLE001
        return len(text) * 12
