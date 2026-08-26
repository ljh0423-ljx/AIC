#!/bin/bash
# infer_baseline.sh — 单张图片推理 (结果存 baseline_v1/predictions)
# 用法: bash scripts/infer_baseline.sh IMAGE_PATH [WEIGHTS_PATH]
set -e
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PD="$PROJECT/PaddleDetection"
CFG="$PROJECT/configs/ppyoloe_plus_crn_s_100e_agrivision.yml"
IMG="${1:?用法: bash scripts/infer_baseline.sh IMAGE_PATH [WEIGHTS_PATH]}"
WEIGHTS="${2:-$PROJECT/experiments/baseline_v1/checkpoints/ppyoloe_plus_crn_s_100e_agrivision/model_final.pdparams}"
OUT_DIR="$PROJECT/experiments/baseline_v1/predictions"

echo "============================================================"
echo " PP-YOLOE+-s Baseline v1 — 推理"
echo " 图片: $IMG"
echo " 权重: $WEIGHTS"
echo "============================================================"

[[ -f "$WEIGHTS" ]] || { echo "[ERROR] 权重不存在: $WEIGHTS"; exit 1; }
[[ -f "$IMG" ]] || { echo "[ERROR] 图片不存在: $IMG"; exit 1; }

mkdir -p "$PD/dataset"
ln -sfn "$PROJECT/dataset/processed_detection_clean" "$PD/dataset/processed_detection"
mkdir -p "$OUT_DIR"

cd "$PD"
python -u tools/infer.py \
  -c "$CFG" \
  -w "$WEIGHTS" \
  --infer_img="$IMG" \
  --draw_threshold=0.5 \
  --output_dir="$OUT_DIR"

echo ""
echo "推理完成。结果图片: $OUT_DIR"
