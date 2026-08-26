# -*- coding: utf-8 -*-
"""
make_demo_data.py — 生成比赛答辩演示数据（web/demo_data/）
===========================================================
- 从 dataset/processed_detection_exiffix/images/val/（**仅 VAL，禁止 TEST**）
  选取 15 张代表性图片，**仅复制、不修改原图**。
- 覆盖 3 作物（番茄/苹果/葡萄）、单目标/多目标/密集场景、12/13 个类别。
- 生成 demo_manifest.json / demo_manifest.csv（图片路径、作物、类别、场景类型等）。

场景说明：作物级中文场景标签（番茄/苹果/葡萄病害检测）为通用描述；
类别名称严格使用项目已有 val.json 英文映射，不擅自创造官方中文类名。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from inference import config as infer_config  # noqa: E402

# 演示选取的 15 张 VAL 图片（file_name -> (demo_id, crop, scene_label)）
# 依据 val.json 标注选定，覆盖不同作物/类别/场景
DEMO_SELECTION: list[tuple[str, str, str, str]] = [
    # (文件名, demo_id, crop, scene_label)
    ("TRAIN_000029_early-blight-of-tomato-tomato-1.jpg", "tomato_01_early_blight", "tomato", "番茄病害检测"),
    ("TRAIN_000037_Tomato+Problems+Septoria+Leaf+Spot.jpg", "tomato_02_septoria", "tomato", "番茄病害检测"),
    ("TRAIN_000040_Septoria_leaf_spot_tomato.jpg", "tomato_03_septoria", "tomato", "番茄病害检测"),
    ("TEST_000079_9511.img.jpg", "tomato_04_mosaic_dense", "tomato", "番茄病害检测（密集）"),
    ("TEST_000088_tylcv-seminar-1-638.jpg", "tomato_05_yellow_virus", "tomato", "番茄病害检测"),
    ("TEST_000099_fungus-univ-of-minnesoeta.jpg", "tomato_06_mold", "tomato", "番茄病害检测"),
    ("TRAIN_000057_bacterial-spot-of-tomato-7-638.jpg", "tomato_07_bacterial_spot", "tomato", "番茄病害检测"),
    ("TRAIN_000069_IMG_5360.jpg", "tomato_08_late_blight", "tomato", "番茄病害检测"),
    ("TRAIN_000004_apple_scab.jpg", "apple_01_scab", "apple", "苹果叶部病害检测"),
    ("TEST_000018_0605_Rust-induced_leafspot.jpg", "apple_02_rust", "apple", "苹果叶部病害检测"),
    ("TRAIN_000020_PLPATH-FRU-02-cedar-apple-rust-figure-1.jpg", "apple_03_rust", "apple", "苹果叶部病害检测"),
    ("TRAIN_000014_stock-photo-green-apple-leaf-clipping-path-258728936.jpg", "apple_04_healthy_leaf", "apple", "苹果叶部病害检测"),
    ("TEST_000106_depositphotos_3443387-stock-photo-the-green-grape-leaf-on.jpg", "grape_01_healthy_leaf", "grape", "葡萄叶部病害检测"),
    ("TEST_000108_5-29black-rot-chardRR.jpg", "grape_02_black_rot", "grape", "葡萄叶部病害检测"),
    ("TEST_000112_Black%20rot%20on%20foliage.jpg", "grape_03_black_rot", "grape", "葡萄叶部病害检测"),
]

CROP_CN = {"tomato": "番茄", "apple": "苹果", "grape": "葡萄"}


def scenario_of(n_gt: int) -> str:
    return "dense" if n_gt >= 10 else ("multi" if n_gt >= 2 else "single")


def build() -> dict:
    val_dir = infer_config.DATASET_DIR / "images" / "val"
    anno_path = infer_config.DATASET_DIR / infer_config.ANNO_REL_PATH
    with open(anno_path, "r", encoding="utf-8") as f:
        ann = json.load(f)
    catid2name = {int(c["id"]): str(c["name"]) for c in ann["categories"]}
    name2id = {im["file_name"]: im["id"] for im in ann["images"]}

    demo_dir = Path(__file__).resolve().parent / "demo_data"
    demo_dir.mkdir(parents=True, exist_ok=True)

    cases: list[dict] = []
    for fn, demo_id, crop, scene in DEMO_SELECTION:
        src = val_dir / fn
        if not src.is_file():
            raise FileNotFoundError(f"[演示数据] VAL 图片不存在：{src}")
        # 仅复制，不修改原图
        dst = demo_dir / fn
        dst.write_bytes(src.read_bytes())

        img_id = name2id.get(fn)
        gts = [a for a in ann["annotations"] if a["image_id"] == img_id]
        gt_classes = sorted({catid2name[a["category_id"]] for a in gts})
        cases.append({
            "demo_id": demo_id,
            "crop": crop,
            "crop_cn": CROP_CN[crop],
            "scene_label": scene,
            "scenario": scenario_of(len(gts)),
            "file_name": fn,
            "source_path": str(src),
            "gt_count": len(gts),
            "gt_classes": gt_classes,
        })

    manifest = {
        "title": "农业病害智能检测系统 · 演示数据（VAL 子集）",
        "note": "图片均来自 dataset/processed_detection_exiffix/images/val（仅 VAL，未使用 TEST）；仅复制、未修改原图。",
        "count": len(cases),
        "cases": cases,
    }
    with open(demo_dir / "demo_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    import csv

    with open(demo_dir / "demo_manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["demo_id", "crop", "crop_cn", "scene_label", "scenario",
                    "file_name", "gt_count", "gt_classes", "source_path"])
        for c in cases:
            w.writerow([c["demo_id"], c["crop"], c["crop_cn"], c["scene_label"],
                        c["scenario"], c["file_name"], c["gt_count"],
                        "|".join(c["gt_classes"]), c["source_path"]])
    return manifest


if __name__ == "__main__":
    m = build()
    print(f"演示数据生成完成：{m['count']} 个案例 -> {Path(__file__).resolve().parent / 'demo_data'}")
    from collections import Counter

    print("作物分布:", dict(Counter(c["crop"] for c in m["cases"])))
    print("场景分布:", dict(Counter(c["scenario"] for c in m["cases"])))
