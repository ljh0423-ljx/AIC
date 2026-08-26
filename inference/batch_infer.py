# -*- coding: utf-8 -*-
"""
batch_infer.py — 批量推理引擎
===============================
- 图片发现（单文件 / 文件夹 / 递归 / 数量限制）
- 单张失败隔离：一张图出错不终止整个批次，记录失败文件与原因
- 输出目录按时间戳自动创建（run_YYYYMMDD_HHMMSS/），不覆盖历史结果
- 每张图保存：原图副本、可视化图、JSON、简洁 TXT、CSV
- 批次级保存：predictions.json / predictions.csv / class_statistics.csv /
  inference_summary.json / inference.log
- 性能统计全部基于真实运行时间（time.perf_counter），不写死、不虚构 FPS

图片读取与后端完全解耦：本模块负责读图与产物落盘，
后端（DetectorBackend）只负责"图像数组 -> Detection 列表"。
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from . import config as app_config
from .postprocess import (
    Detection,
    RECORD_FIELDS,
    batch_class_statistics,
    detection_rows_for_csv,
    detections_to_records,
    image_summary,
    render_image_txt,
    write_csv,
    write_json,
    write_txt,
)


class ImageReadError(Exception):
    pass


class BatchError(Exception):
    pass


# ---------------------------------------------------------------------------
# 图片读取（稳健解码，支持 jpg/jpeg/png/bmp/webp 与非 ASCII 路径）
# ---------------------------------------------------------------------------
def read_image(path: Path) -> np.ndarray:
    """读取图片为 BGR 数组；失败抛出 ImageReadError（中文信息）。"""
    path = Path(path)
    if not path.is_file():
        raise ImageReadError(f"图片不存在：{path}")
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except Exception as e:  # noqa: BLE001
        raise ImageReadError(f"图片文件读取失败：{path}（{e}）") from e
    img = None
    if data is not None and data.size > 0:
        try:
            img = np.uint8(data)
        except Exception:  # noqa: BLE001
            img = None
    bgr = cv2_imdecode(img) if img is not None else None
    if bgr is None:
        raise ImageReadError(f"图片无法解码（可能已损坏或不支持的格式）：{path.name}")
    return bgr


def cv2_imdecode(data: np.ndarray):
    import cv2

    try:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# 图片发现
# ---------------------------------------------------------------------------
def discover_images(
    paths: list[Path],
    recursive: bool = False,
    limit: Optional[int] = None,
) -> list[Path]:
    """合并单文件与文件夹，返回去重排序后的图片路径列表。"""
    ext = app_config.SUPPORTED_EXTENSIONS
    found: list[Path] = []
    missing: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_file():
            if p.suffix.lower() in ext:
                found.append(p)
            else:
                missing.append(p)
        elif p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            found.extend(
                f for f in it if f.is_file() and f.suffix.lower() in ext
            )
        else:
            missing.append(p)

    # 去重（保持相对稳定顺序）
    seen: set[str] = set()
    unique: list[Path] = []
    for f in sorted(found, key=lambda x: str(x)):
        k = str(f.resolve())
        if k not in seen:
            seen.add(k)
            unique.append(f)

    if missing:
        for m in missing:
            logging.getLogger("agri.infer.batch").warning(
                "[跳过] 非图片或路径不存在：%s", m
            )

    if limit is not None and limit > 0 and len(unique) > limit:
        unique = unique[: limit]

    if not unique:
        raise BatchError(
            "[输入错误] 未找到任何可推理的图片。\n"
            "请确认 --image/--dir 指向的路径存在，且图片为 jpg/jpeg/png/bmp/webp 格式。"
        )
    return unique


def _unique_image_names(paths: list[Path]) -> list[str]:
    """为图片列表生成唯一 image_name（同名时追加序号，避免产物互相覆盖）。"""
    counts: dict[str, int] = {}
    result: list[str] = []
    for p in paths:
        name = p.name
        counts[name] = counts.get(name, 0) + 1
        if counts[name] > 1:
            name = f"{p.stem}__{counts[name]}{p.suffix}"
        result.append(name)
    return result


# ---------------------------------------------------------------------------
# 批次运行
# ---------------------------------------------------------------------------
def run_batch(
    detector,
    image_paths: list[Path],
    conf: float,
    output_root: Path | str,
    *,
    recursive: bool = False,
    limit: Optional[int] = None,
    enable_visualization: bool = True,
    catid2cn: Optional[dict] = None,
    backend_name: Optional[str] = None,
    paddle_version: str = "",
) -> dict:
    """执行批量推理并落盘所有产物，返回批次摘要 dict。

    参数 detector 为已 load() 的 DetectorBackend 实例（模型只加载一次）。
    """
    logger = logging.getLogger("agri.infer.batch")

    image_paths = discover_images(image_paths, recursive=recursive, limit=limit)
    image_names = _unique_image_names(image_paths)
    n_total = len(image_paths)

    # ---- 输出目录（时间戳，不覆盖历史）----
    output_root = Path(output_root)
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_dir = output_root / run_id
    try:
        out_dir.mkdir(parents=True, exist_ok=False)
        (out_dir / "originals").mkdir(exist_ok=True)
        (out_dir / "visualized").mkdir(exist_ok=True)
        (out_dir / "detections").mkdir(exist_ok=True)
    except OSError as e:
        raise BatchError(
            f"[输出错误] 无法创建输出目录 {out_dir}：{e}\n"
            f"请检查磁盘空间与权限，或使用 --output 指定其他位置。"
        ) from e

    _setup_file_logger(out_dir / "inference.log")

    # ---- 逐图处理 ----
    image_results: list[dict] = []   # 供批次统计
    failed: list[dict] = []          # 失败记录
    all_records: list[dict] = []     # 全部预测记录
    # 类别映射统一走后端公开接口 catid2name（与后端无关）
    catid2name = getattr(detector, "catid2name", getattr(detector, "_catid2name", {}))

    t_start = time.perf_counter()
    for idx, (path, name) in enumerate(zip(image_paths, image_names)):
        logger.info("[%d/%d] 处理：%s", idx + 1, n_total, name)
        try:
            img = read_image(path)
            t_inf0 = time.perf_counter()
            dets: list[Detection] = detector.infer(img)
            t_inf1 = time.perf_counter()
            inference_ms = (t_inf1 - t_inf0) * 1000.0
        except Exception as e:  # noqa: BLE001
            failed.append({"image_name": name, "image_path": str(path), "error": str(e)})
            logger.error("  ✗ 推理失败：%s", e)
            continue

        # 置信度过滤
        from .postprocess import filter_by_conf

        dets = filter_by_conf(dets, conf)
        for d in dets:
            d.image_name = name
            all_records.append(d.to_dict())

        image_results.append({"image_name": name, "detections": dets})

        # ---- 每图产物落盘 ----
        try:
            _save_per_image(path, name, img, dets, catid2name, catid2cn,
                            out_dir, inference_ms, enable_visualization)
        except Exception as e:  # noqa: BLE001
            # 检测成功但产物落盘失败：记录为警告，检测结果仍计入批次统计
            logger.warning("  ⚠ 产物保存异常（检测本身成功）：%s", e)
        logger.info(
            "  ✓ 目标数=%d 推理耗时=%.0f ms",
            len(dets),
            inference_ms,
        )

    t_end = time.perf_counter()
    total_time = t_end - t_start

    # ---- 统计 ----
    n_success = len(image_results)
    n_failed = n_total - n_success
    class_stats = batch_class_statistics(image_results)
    avg_time = (total_time / n_total) if n_total else 0.0
    fps = (n_total / total_time) if total_time > 0 else 0.0
    fps_success = (n_success / total_time) if total_time > 0 and n_success else 0.0

    # ---- 批次级产物 ----
    _save_batch_outputs(
        out_dir, all_records, class_stats, image_results, failed,
        n_total, n_success, n_failed, total_time, avg_time, fps,
        detector, conf, run_id, image_paths, paddle_version,
    )

    summary = {
        "run_id": run_id,
        "output_dir": str(out_dir),
        "total_images": n_total,
        "success_images": n_success,
        "failed_images": n_failed,
        "total_time": round(total_time, 4),
        "average_time": round(avg_time, 4),
        "fps": round(fps, 4),
        "fps_success_based": round(fps_success, 4),
        "failed": failed,
        "class_statistics": class_stats,
    }
    logger.info("批次完成：total=%d success=%d failed=%d 耗时=%.2fs FPS=%.2f",
                n_total, n_success, n_failed, total_time, fps)
    return summary


# ---------------------------------------------------------------------------
# 每图产物
# ---------------------------------------------------------------------------
def _save_per_image(
    path: Path,
    name: str,
    img: np.ndarray,
    dets: list[Detection],
    catid2name: dict,
    catid2cn: Optional[dict],
    out_dir: Path,
    inference_ms: float,
    enable_visualization: bool,
) -> None:
    import cv2

    stem = Path(name).stem
    # 1) 原图副本
    orig_path = out_dir / "originals" / name
    cv2.imwrite(str(orig_path), img)
    # 2) 检测 JSON
    rec = {
        "image_name": name,
        "total_targets": len(dets),
        "inference_ms": round(inference_ms, 2),
        "detections": detections_to_records(dets),
    }
    write_json(out_dir / "detections" / f"{stem}.json", rec)
    # 3) 简洁 TXT
    write_txt(out_dir / "detections" / f"{stem}.txt", render_image_txt(name, dets))
    # 4) CSV（检测明细行）
    write_csv(out_dir / "detections" / f"{stem}.csv", RECORD_FIELDS,
              detection_rows_for_csv(dets))
    # 5) 可视化
    if enable_visualization:
        from .visualize import draw_detections

        drawn = draw_detections(
            img, dets, catid2name, catid2cn=catid2cn,
            inference_seconds=inference_ms / 1000.0,
        )
        cv2.imwrite(str(out_dir / "visualized" / name), drawn)


# ---------------------------------------------------------------------------
# 批次级产物
# ---------------------------------------------------------------------------
def _save_batch_outputs(
    out_dir: Path,
    all_records: list[dict],
    class_stats: list[dict],
    image_results: list[dict],
    failed: list[dict],
    n_total: int,
    n_success: int,
    n_failed: int,
    total_time: float,
    avg_time: float,
    fps: float,
    detector,
    conf: float,
    run_id: str,
    image_paths: list[Path],
    paddle_version: str,
) -> None:
    # 1) predictions.json / csv
    write_json(out_dir / "predictions.json", all_records)
    write_csv(out_dir / "predictions.csv", RECORD_FIELDS,
              [[r[k] for k in RECORD_FIELDS] for r in all_records])
    # 2) class_statistics.csv
    header = ["class_id", "class_name", "num_images", "num_targets",
              "avg_confidence", "max_confidence"]
    write_csv(out_dir / "class_statistics.csv", header,
              [[r["class_id"], r["class_name"], r["num_images"],
                r["num_targets"], r["avg_confidence"], r["max_confidence"]]
               for r in class_stats])
    # 3) inference_summary.json
    info = detector.info() if hasattr(detector, "info") else {}
    summary = {
        "run_id": run_id,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "model_path": info.get("model_path", ""),
            "config_path": info.get("config_path", ""),
            "dataset_dir": info.get("dataset_dir", ""),
            "anno_path": info.get("anno_path", ""),
            "backend": info.get("backend", getattr(detector, "backend_name", "unknown")),
            "device": info.get("device", getattr(detector, "device", "unknown")),
            "paddle_version": info.get("paddle_version", paddle_version),
            "conf_threshold": conf,
            "input_size": info.get("input_size", None),
            "num_classes": info.get("num_classes", 0),
        },
        "performance": {
            "total_images": n_total,
            "success_images": n_success,
            "failed_images": n_failed,
            "total_time": round(total_time, 4),
            "average_time": round(avg_time, 4),
            "fps": round(fps, 4),
        },
        "failed": failed,
        "class_statistics": class_stats,
        "output_files": {
            "originals_dir": str(out_dir / "originals"),
            "visualized_dir": str(out_dir / "visualized"),
            "detections_dir": str(out_dir / "detections"),
            "predictions_json": str(out_dir / "predictions.json"),
            "predictions_csv": str(out_dir / "predictions.csv"),
            "class_statistics_csv": str(out_dir / "class_statistics.csv"),
            "inference_summary_json": str(out_dir / "inference_summary.json"),
            "inference_log": str(out_dir / "inference.log"),
        },
    }
    write_json(out_dir / "inference_summary.json", summary)


# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
def _setup_file_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("agri.infer.batch")
    logger.setLevel(logging.INFO)
    # 避免重复添加 handler
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path):
            return logger
    try:
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
    except OSError:
        pass
    return logger
