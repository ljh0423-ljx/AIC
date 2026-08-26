# -*- coding: utf-8 -*-
"""
backend.py — Web 层检测后端单例与启动自检
==========================================
- Web 层**不直接依赖 Paddle 具体实现**，统一通过 inference.detector.DetectorBackend
  调用推理（当前为 PaddleBackend，后续 RKNNBackend 无缝替换）。
- 模型**只加载一次**：get_detector() 以单例方式缓存 DetectorBackend；
  启动自检 self_check() 触发首次加载，之后任何按钮点击都复用同一实例。
- 自检仅检查：模型文件 / 配置文件 / 类别映射 / 后端可用性，**不访问 TEST**。
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

from inference.config import build_class_mapping
from inference.detector import DetectorError, create_backend, get_gpu_info, resolve_device

from . import config as web_config

logger = logging.getLogger("agri.web.backend")

_detector = None
_detector_error: DetectorError | None = None
_lock = threading.Lock()

# 推理设备模式：auto / cpu / gpu。默认 auto（启动时自动检测）。
# 可通过环境变量 AGRI_DEVICE 或 Web 设备选择器 / 启动参数 --device 设置，无需改代码。
_DEVICE_MODE: str = os.environ.get("AGRI_DEVICE", "auto").lower()


def get_device_mode() -> str:
    """返回当前设备模式（auto / cpu / gpu）。"""
    return _DEVICE_MODE


def set_device_mode(mode: str) -> None:
    """设置设备模式并重置后端缓存（下次 get_detector 按新设备重新加载一次）。

    支持 auto / cpu / gpu；未知模式抛中文错误。
    """
    global _DEVICE_MODE, _detector, _detector_error
    mode = (mode or "auto").lower()
    if mode not in ("auto", "cpu", "gpu"):
        raise DetectorError(f"[配置错误] 未知设备模式：{mode}（支持 auto / cpu / gpu）")
    _DEVICE_MODE = mode
    _detector = None
    _detector_error = None
    logger.info("推理设备模式已设置：%s", mode)


def get_detector():
    """返回 DetectorBackend 单例；模型只加载一次；按设备模式 auto/cpu/gpu 自动解析真实设备。"""
    global _detector, _detector_error
    if _detector is not None:
        return _detector
    with _lock:
        if _detector is not None:
            return _detector
        if _detector_error is not None:
            raise _detector_error
        try:
            device = resolve_device(_DEVICE_MODE)   # auto → cpu/gpu 自动检测
            det = create_backend(
                kind="paddle",
                model_path=web_config.MODEL_PATH,
                config_path=web_config.CONFIG_PATH,
                dataset_dir=web_config.DATASET_DIR,
                anno_rel_path=web_config.ANNO_REL_PATH,
                device=device,
            )
            det.load()
            _detector = det
            logger.info("Web 后端已加载：%s | %s", det.backend_name, det.device)
            return _detector
        except DetectorError as e:
            _detector_error = e
            raise
        except Exception as e:  # noqa: BLE001
            _detector_error = DetectorError(f"[模型错误] 模型加载失败：{e}")
            raise _detector_error from e


def detector_info() -> dict:
    """当前后端信息（不触发加载；未加载时返回占位信息）。

    优先使用后端统一接口 info()（与后端解耦），后续 RKNNBackend 同样可用；
    真实 GPU 型号来自 Paddle 检测（无 GPU 显示 "无"，不虚构）。
    """
    gpu_info = get_gpu_info() or "无"
    if _detector is not None and hasattr(_detector, "info"):
        info = _detector.info()
        return {
            "backend": info.get("backend", "-"),
            "device": info.get("device", "-"),
            "paddle_version": info.get("paddle_version", "-"),
            "input_size": info.get("input_size", web_config.MODEL_RECORDS["input_size"]),
            "num_classes": info.get("num_classes", web_config.MODEL_RECORDS["num_classes"]),
            "model_name": info.get("model_name", "PP-YOLOE+-m"),
            "gpu_info": gpu_info,
        }
    return {
        "backend": "-",
        "device": "-",
        "paddle_version": "-",
        "input_size": web_config.MODEL_RECORDS["input_size"],
        "num_classes": web_config.MODEL_RECORDS["num_classes"],
        "model_name": web_config.MODEL_RECORDS["model_name"],
        "gpu_info": gpu_info,
    }


# ---------------------------------------------------------------------------
# 启动自检
# ---------------------------------------------------------------------------
def self_check() -> list[dict]:
    """启动自检：模型文件 / 配置文件 / 类别映射 / 后端可用性。不访问 TEST。

    返回 [{"name", "ok"(bool), "detail"(str)}, ...]。
    """
    checks: list[dict] = []

    # 1) 模型文件
    ok = web_config.MODEL_PATH.is_file()
    checks.append({
        "name": "模型文件",
        "ok": ok,
        "detail": str(web_config.MODEL_PATH) if ok
                 else f"模型文件不存在：{web_config.MODEL_PATH}",
    })

    # 2) 配置文件
    ok = web_config.CONFIG_PATH.is_file()
    checks.append({
        "name": "配置文件",
        "ok": ok,
        "detail": str(web_config.CONFIG_PATH) if ok
                  else f"配置文件不存在：{web_config.CONFIG_PATH}",
    })

    # 3) 类别映射（取自 val.json，非 TEST）
    anno = web_config.DATASET_DIR / web_config.ANNO_REL_PATH
    if anno.is_file():
        try:
            catid2name, _clsid2catid = build_class_mapping(anno)
            ok = len(catid2name) == 13
            checks.append({
                "name": "类别映射",
                "ok": ok,
                "detail": f"13 类（来自 {web_config.ANNO_REL_PATH}）"
                          if ok else f"类别数量异常：{len(catid2name)}（应为 13）",
            })
        except Exception as e:  # noqa: BLE001
            checks.append({"name": "类别映射", "ok": False, "detail": str(e)})
    else:
        checks.append({
            "name": "类别映射", "ok": False,
            "detail": f"标注文件不存在：{anno}",
        })

    # 4) 后端可用性（触发模型加载，仅此一次）
    try:
        det = get_detector()
        checks.append({
            "name": "推理后端",
            "ok": True,
            "detail": f"{det.backend_name} | device={det.device} | "
                      f"paddle={getattr(det, '_paddle_version', '')} | "
                      f"输入 {getattr(det, 'input_size', 640)}×{getattr(det, 'input_size', 640)}",
        })
    except DetectorError as e:
        checks.append({"name": "推理后端", "ok": False, "detail": str(e)})

    return checks
