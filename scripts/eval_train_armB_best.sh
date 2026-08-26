#!/bin/bash
# eval_train_armB_best.sh — 在 TRAIN 集评估 Arm B best_model, 用于生成 train-only confusion prior
# 仅 TRAIN; VAL/TEST 不参与。输出 train_bbox.json + train_eval.log
# 注意: 评估 TRAIN 不会把 TRAIN 数据泄漏到 VAL/TEST 评估中; 混淆先验只来自 TRAIN。
set -e
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PD="$PROJECT/PaddleDetection"
CFG="$PROJECT/configs/ppyoloe_plus_crn_s_100e_agrivision.yml"
WEIGHTS="$PROJECT/experiments/ablation_data_exiffix/checkpoints/best_model.pdparams"
DS="$PROJECT/dataset/processed_detection_exiffix"
OUT="$PROJECT/experiments/direction2/train_prior_eval"

mkdir -p "$PD/dataset"
ln -sfn "$DS" "$PD/dataset/processed_detection"
mkdir -p "$OUT"
cd "$PD"
python -u tools/eval.py \
  -c "$CFG" \
  -o weights="$WEIGHTS" \
     EvalDataset.image_dir=images/train \
     EvalDataset.anno_path=annotations/train.json \
     EvalDataset.dataset_dir=dataset/processed_detection \
  --output_eval="$OUT/train_eval.json" \
  2>&1 | tee "$OUT/train_eval.log"
echo "完成: $OUT/train_eval.json/bbox.json"
