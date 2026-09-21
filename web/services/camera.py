# -*- coding: utf-8 -*-
"""
web/services/camera.py — 实时摄像头视觉检测接口（阶段7.1）
==========================================================
把摄像头捕获的单帧图像转换为现有 _detect 流程可直接消费的图片文件路径，
复用 PP-YOLOE 推理（不重新实现）。

当前阶段：Gradio webcam 单帧（框架自带采集），无需真实摄像头硬件。
未来接入（预留）：RK3588 / USB 工业相机等真实摄像头后端，替换 save_frame 实现即可。

本模块不依赖 Paddle、不依赖 inference 内部实现，不影响 severity / vlm / rag /
agent / speech / robot 等已有模块。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger("agri.web.services.camera")

# 输出目录：web/tmp/（与上传暂存同目录）
_OUTPUT_DIR: Path = Path(__file__).resolve().parent.parent / "tmp"


def save_frame(frame, output_dir=None) -> Path | None:
    """把摄像头帧保存为图片文件，返回文件路径；帧为空/无法解析时返回 None。

    支持输入：
    - 文件路径字符串 / Path（Gradio Image type="filepath"）
    - numpy 数组（BGR，OpenCV 约定）
    - PIL.Image 对象
    - 带 .path 属性的对象（Gradio FileData 等）
    """
    if frame is None:
        return None

    out = Path(output_dir) if output_dir else _OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    # 1) 文件路径
    if isinstance(frame, (str, Path)):
        return Path(frame)

    # 2) numpy 数组（BGR）
    import numpy as np

    if isinstance(frame, np.ndarray):
        import cv2

        path = out / f"camera_{ts}.jpg"
        ok = cv2.imwrite(str(path), frame)
        return path if ok else None

    # 3) PIL.Image
    try:
        from PIL import Image

        if isinstance(frame, Image.Image):
            path = out / f"camera_{ts}.jpg"
            frame.save(str(path))
            return path if path.is_file() else None
    except Exception:  # noqa: BLE001
        pass

    # 4) 带 .path 属性的对象
    if hasattr(frame, "path"):
        return Path(frame.path)

    logger.warning("[camera] 无法识别的帧类型：%s", type(frame).__name__)
    return None


def get_camera_status() -> str:
    """返回摄像头模块状态。"""
    return "Gradio webcam 单帧（mock，未接入真实摄像头硬件）"
