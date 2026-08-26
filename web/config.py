# -*- coding: utf-8 -*-
"""
config.py — Web 展示系统配置
=============================
复用 inference/config.py 的全部路径与 13 类映射（不重复定义模型/数据集路径）。
本文件只定义 Web 层 UI 常量与**固定模型性能记录**。

模型性能数据（VAL/TEST/参数量/大小）为固定记录，来源：
  - experiments/direction_m/M_SCALEUP_100e/M_SCALEUP_FINAL_REPORT.md
  - experiments/final_evaluation/M_FINAL_TEST_REPORT.md（§3.7 参数量/模型大小/FPS）
  - PROJECT_FINAL_STATUS.md
这些数值仅作展示，**不重新训练、不重新评估**。
TEST 数值已明确标注为"独立 TEST 最终评估结果"。
"""

from __future__ import annotations

from pathlib import Path

from inference import config as infer_config

# ---------------------------------------------------------------------------
# 复用推理配置（模型/配置/数据集/类别映射路径，全部 Path，无写死 Windows 路径）
# ---------------------------------------------------------------------------
MODEL_PATH: Path = infer_config.MODEL_PATH
CONFIG_PATH: Path = infer_config.CONFIG_PATH
DATASET_DIR: Path = infer_config.DATASET_DIR
ANNO_REL_PATH: str = infer_config.ANNO_REL_PATH
PADDLE_DETECTION_DIR: Path = infer_config.PADDLE_DETECTION_DIR

# Web 输出目录（独立于 CLI 推理的 inference/outputs，互不覆盖）
WEB_BASE_DIR: Path = Path(__file__).resolve().parent
WEB_OUTPUT_ROOT: Path = WEB_BASE_DIR / "outputs"      # 每次检测生成 run_YYYYMMDD_HHMMSS/
WEB_TMP_DIR: Path = WEB_BASE_DIR / "tmp"              # 上传文件暂存目录

# ---------------------------------------------------------------------------
# 标题与副标题
# ---------------------------------------------------------------------------
APP_TITLE_CN: str = "农业病害智能检测系统"
APP_TITLE_EN: str = "Agricultural Disease Intelligent Detection System"
MODEL_TAG: str = "PP-YOLOE+-m"
APP_SUBTITLE: str = "AI + Agriculture · Plant Disease Detection"

# ---------------------------------------------------------------------------
# 固定模型性能记录（仅展示，来源见文件头）
# ---------------------------------------------------------------------------
MODEL_RECORDS: dict = {
    "model_name": "PP-YOLOE+-m",
    "val_map": 0.428,     # VAL mAP@0.5:0.95（模型选择依据）
    "test_map": 0.417,    # TEST mAP@0.5:0.95（独立 TEST 最终评估结果，未重跑）
    "params_m": 23.57,    # 参数量（百万）
    "size_mb": 94.3,      # 模型大小（MB，pdparams）
    "input_size": 640,    # 模型输入边长
    "num_classes": 13,    # 类别数
    "test_note": "模型选择依据 VAL；TEST 仅作为最终独立泛化性能报告，未重新运行。",
}

# ---------------------------------------------------------------------------
# 类别中文名：项目当前无官方中文映射，遵循"没有则不要擅自创造"，默认英文。
# 与 inference/config.py 的 CLASS_NAMES_CN 保持一致（用户填写后自动启用双语）。
# ---------------------------------------------------------------------------
CLASS_NAMES_CN: dict[int, str] = infer_config.CLASS_NAMES_CN

# 默认推理置信度阈值（明确标注：这是推理置信度阈值，不是模型训练参数）
DEFAULT_CONF_THRESHOLD: float = 0.5
