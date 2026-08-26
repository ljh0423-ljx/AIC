#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO TXT -> COCO JSON 转换 + 自动校验 (项目内自包含版本)。

用法:
  python tools/yolo_to_coco.py [--dataset DATASET_DIR]

默认 DATASET_DIR = <project>/dataset/processed_detection

输出: <DATASET_DIR>/annotations/{train,val,test}.json
校验: image_id 连续 / category_id 1..13 / bbox 合法 / area 正确 / 1:1 / 类名一致 / 无跨split
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
SPLITS = ["train", "val", "test"]


def to_long(p: str) -> str:
    a = os.path.abspath(p)
    return a if a.startswith("\\\\?\\") else "\\\\?\\" + a


def load_label_list(dataset: str) -> List[str]:
    with open(to_long(os.path.join(dataset, "label_list.txt")), encoding="utf-8") as f:
        return [ln.strip().split("\t")[1] for ln in f if ln.strip() and len(ln.strip().split("\t")) >= 2]


def md5_file(path: str) -> str:
    with open(to_long(path), "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def convert_split(dataset: str, split: str, class_names: List[str]) -> Tuple[dict, Dict[str, int]]:
    img_dir = os.path.join(dataset, "images", split)
    lbl_dir = os.path.join(dataset, "labels", split)
    images, annotations = [], []
    img_id = ann_id = 0
    f2id: Dict[str, int] = {}
    for fn in sorted(os.listdir(to_long(img_dir))):
        if os.path.splitext(fn)[1].lower() not in IMG_EXTS:
            continue
        img_path = os.path.join(img_dir, fn)
        try:
            with open(to_long(img_path), "rb") as f:
                raw = f.read()
            img = Image.open(io.BytesIO(raw)); img.load()
            W, H = img.size
        except Exception as e:
            print(f"[warn] {split}/{fn} 读取失败: {e}"); continue
        img_id += 1
        f2id[fn] = img_id
        images.append({"id": img_id, "file_name": fn, "width": W, "height": H})
        lbl_path = os.path.join(lbl_dir, os.path.splitext(fn)[0] + ".txt")
        if not os.path.isfile(to_long(lbl_path)):
            continue
        with open(to_long(lbl_path), encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        for ln in lines:
            p = ln.split()
            if len(p) < 5:
                continue
            clsid = int(p[0]); xc, yc, w, h = map(float, p[1:5])
            bw, bh = w * W, h * H
            bx, by = xc * W - bw / 2, yc * H - bh / 2
            bx = max(0.0, min(bx, W - 1)); by = max(0.0, min(by, H - 1))
            bw = max(0.0, min(bw, W - bx)); bh = max(0.0, min(bh, H - by))
            if bw <= 0 or bh <= 0:
                continue
            bx, by, bw, bh = round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)
            ann_id += 1
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": clsid + 1,
                                "bbox": [bx, by, bw, bh], "area": round(bw * bh, 2), "iscrowd": 0})
    cats = [{"id": i + 1, "name": n, "supercategory": "plant_leaf"} for i, n in enumerate(class_names)]
    return {"images": images, "annotations": annotations, "categories": cats}, f2id


def validate(coco: dict, split: str, class_names: List[str]) -> List[str]:
    errs = []
    ids = [im["id"] for im in coco["images"]]
    if len(set(ids)) != len(ids):
        errs.append(f"{split}: image_id 重复")
    if ids and sorted(ids) != list(range(1, len(ids) + 1)):
        errs.append(f"{split}: image_id 不连续")
    cat_ids = sorted({c["id"] for c in coco["categories"]})
    if cat_ids != list(range(1, 14)):
        errs.append(f"{split}: category_id 不连续 1..13")
    names = [c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])]
    if names != class_names:
        errs.append(f"{split}: 类名与 label_list.txt 不一致")
    iw = {im["id"]: im["width"] for im in coco["images"]}
    ih = {im["id"]: im["height"] for im in coco["images"]}
    bad = bad_a = 0
    for a in coco["annotations"]:
        bx, by, bw, bh = a["bbox"]; W, H = iw[a["image_id"]], ih[a["image_id"]]
        if bx < 0 or by < 0 or bw <= 0 or bh <= 0 or bx + bw > W + 1 or by + bh > H + 1:
            bad += 1
        if abs(a["area"] - bw * bh) > 0.5:
            bad_a += 1
    if bad:
        errs.append(f"{split}: {bad} bbox 越界")
    if bad_a:
        errs.append(f"{split}: {bad_a} area 不符")
    ann_ids = {a["image_id"] for a in coco["annotations"]}
    no_ann = [im["file_name"] for im in coco["images"] if im["id"] not in ann_ids]
    if no_ann:
        errs.append(f"{split}: {len(no_ann)} 图无标注")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--dataset", default=os.path.normpath(os.path.join(here, "..", "dataset", "processed_detection")))
    a = ap.parse_args()
    dataset = a.dataset
    names = load_label_list(dataset)
    assert len(names) == 13, f"期望13类, 实际{len(names)}"
    ann_dir = os.path.join(dataset, "annotations")
    os.makedirs(to_long(ann_dir), exist_ok=True)
    errs_all = []
    fids = {}
    for sp in SPLITS:
        coco, f2id = convert_split(dataset, sp, names)
        fids[sp] = set(f2id.keys())
        with open(to_long(os.path.join(ann_dir, f"{sp}.json")), "w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False)
        print(f"[{sp}] img={len(coco['images'])} ann={len(coco['annotations'])}")
        for e in validate(coco, sp, names):
            errs_all.append(e); print(f"  [FAIL] {e}")
    cross = (fids["train"] & fids["val"]) | (fids["train"] & fids["test"]) | (fids["val"] & fids["test"])
    if cross:
        errs_all.append(f"跨split 同名: {len(cross)}"); print(f"[FAIL] 跨split 同名 {len(cross)}")
    print("OK 全部校验通过" if not errs_all else f"完成, {len(errs_all)} 问题")
    return 1 if errs_all else 0


if __name__ == "__main__":
    raise SystemExit(main())
