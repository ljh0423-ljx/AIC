# -*- coding: utf-8 -*-
"""
config.py — 农业病害检测系统配置
===============================
集中管理所有路径、13 类映射、默认阈值与运行参数。
全部路径基于项目根目录动态计算（Path），**不写死任何 Windows 路径**，
后续迁移 RK3588 / 其他机器只需复制整个工程，无需改动本文件。

路径约定：
    BASE_DIR         = AIC2026_AgriVision/（inference/ 的上一级）
    PADDLE_DETECTION_DIR = BASE_DIR/PaddleDetection
    所有数据集/模型/配置均使用该工程内已有的绝对路径（运行时以 -o 覆盖注入）。

类别映射约束：
    13 类英文名**严格取自项目已有的 dataset/processed_detection_exiffix/annotations/val.json**
    （与 label_list.txt / test.json 完全一致，不允许重新编号）。
    中文名：项目当前**没有**官方中文映射，按"没有则不要擅自创造"约束，
    默认留空；用户在 CLASS_NAMES_CN 中填写后即可自动启用"中文 + 英文"双语标注。
"""

from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# 工程根目录与关键路径（全部 Path，无写死盘符）
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent  # AIC2026_AgriVision/

PADDLE_DETECTION_DIR: Path = BASE_DIR / "PaddleDetection"

# 最终锁定模型（PP-YOLOE+-m best checkpoint，VAL mAP@0.5:0.95 = 0.428）
MODEL_PATH: Path = (
    BASE_DIR
    / "experiments"
    / "direction_m"
    / "M_SCALEUP_100e"
    / "checkpoints"
    / "best_model.pdparams"
)

# 最终配置（PP-YOLOE+-m 100e，与正式训练/TEST 完全一致）
CONFIG_PATH: Path = BASE_DIR / "configs" / "ppyoloe_plus_crn_m_100e_agrivision.yml"

# 本地数据集（EXIF-fix 最终数据基础）；禁止使用 test.json / TEST 图片
DATASET_DIR: Path = BASE_DIR / "dataset" / "processed_detection_exiffix"

# 类别映射来源：使用 VAL 标注（非 TEST）。与 test.json 的 13 类完全一致。
ANNO_REL_PATH: str = "annotations/val.json"

# 推理输出根目录（每次运行在其下自动创建 run_YYYYMMDD_HHMMSS/，不覆盖历史）
OUTPUT_ROOT: Path = BASE_DIR / "inference" / "outputs"

# ---------------------------------------------------------------------------
# 推理默认参数
# ---------------------------------------------------------------------------
DEFAULT_CONF_THRESHOLD: float = 0.5   # 置信度阈值
DEFAULT_INPUT_SIZE: int = 640          # 模型输入边长（与正式评估一致）

# 支持的图片扩展名（含大写）
SUPPORTED_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 可视化标题 / 模型标识（答辩展示用）
APP_TITLE: str = "Agricultural Disease Detection"
MODEL_TAG: str = "PP-YOLOE+-m"

# ---------------------------------------------------------------------------
# 13 类映射（运行时从 val.json 加载；下面为读取函数）
# ---------------------------------------------------------------------------


def load_categories_from_anno(anno_path: Path) -> list[dict]:
    """从 COCO 标注文件读取 categories，按 category id 升序返回。

    返回 [{'id': 1, 'name': 'Tomato Early blight leaf'}, ...]。
    id 即 COCO 1-based 类别编号（1-13），与模型 0-12 输出索引通过
    clsid2catid 映射对齐；**不重新编号**。
    """
    anno_path = Path(anno_path)
    if not anno_path.is_file():
        raise FileNotFoundError(
            f"[配置错误] 类别映射标注文件不存在：{anno_path}。\n"
            f"请确认数据集目录完整（应含 annotations/val.json），且未误用 test.json。"
        )
    with open(anno_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cats = data.get("categories", [])
    if not cats:
        raise ValueError(f"[配置错误] 标注文件 {anno_path} 中 categories 为空。")
    cats = sorted(cats, key=lambda c: int(c["id"]))
    return cats


def build_class_mapping(anno_path: Path) -> tuple[dict[int, str], dict[int, int]]:
    """构建 类别id->英文名 与 模型索引->类别id 两张映射。

    返回 (catid2name, clsid2catid)。
    clsid2catid：模型输出 0-12 索引 -> COCO 1-based 类别 id（1-13），
    顺序严格等于 val.json categories 的升序，**不允许重新编号**。
    """
    cats = load_categories_from_anno(anno_path)
    catid2name: dict[int, str] = {int(c["id"]): str(c["name"]) for c in cats}
    clsid2catid: dict[int, int] = {i: int(c["id"]) for i, c in enumerate(cats)}
    return catid2name, clsid2catid


# 项目当前没有官方中文映射，默认留空。
# 用户可按下表格式填写（例如 {1: "番茄早疫病", ...}）以启用"中文 + 英文"标注；
# 未填写时可视化仅显示英文（遵循"没有则不要擅自创造"约束）。
CLASS_NAMES_CN: dict[int, str] = {}
