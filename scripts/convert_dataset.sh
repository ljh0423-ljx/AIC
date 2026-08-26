#!/bin/bash
# convert_dataset.sh — YOLO TXT -> COCO JSON (复现用, 数据集已含转换结果)
set -e
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "[convert] YOLO TXT -> COCO JSON ..."
python "$PROJECT/tools/yolo_to_coco.py" --dataset "$PROJECT/dataset/processed_detection"
echo "[convert] 完成。结果在 dataset/processed_detection/annotations/"
