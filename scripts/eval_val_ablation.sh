#!/bin/bash
# eval_val_ablation.sh — 对 Ablation 某臂的某 checkpoint 做 VAL 集评估
# 仅 VAL; TEST 不参与。输出预测 bbox.json 供 per-class 消融分析。
# 用法: bash scripts/eval_val_ablation.sh <clean|exiffix> <best|final>
set -e
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PD="$PROJECT/PaddleDetection"
CFG="$PROJECT/configs/ppyoloe_plus_crn_s_100e_agrivision.yml"
ARM="${1:?ARM=clean|exiffix}"
TAG="${2:?TAG=best|final}"
[[ "$ARM" == "clean" || "$ARM" == "exiffix" ]] || { echo "[ERROR] ARM 无效"; exit 1; }
EXP="$PROJECT/experiments/ablation_data_${ARM}"
DS="$PROJECT/dataset/processed_detection_${ARM}"
# 权重定位: 兼容带/不带模型名子目录两种结构
case "$TAG" in
  best)  WEIGHTS="$(find "$EXP/checkpoints" -name 'best_model.pdparams' | head -1)";;
  final) WEIGHTS="$(find "$EXP/checkpoints" -name 'model_final.pdparams' | head -1)";;
  *) echo "[ERROR] TAG 无效"; exit 1;;
esac
[[ -n "$WEIGHTS" && -f "$WEIGHTS" ]] || { echo "[ERROR] 权重不存在 (tag=$TAG): $EXP/checkpoints"; exit 1; }
WEIGHTS="$(readlink -f "$WEIGHTS")"

mkdir -p "$PD/dataset"
ln -sfn "$DS" "$PD/dataset/processed_detection"
mkdir -p "$EXP/metrics/val_eval_$TAG"
cd "$PD"
echo "[VAL eval] Arm=$ARM tag=$TAG 权重=$WEIGHTS 数据=$DS"
python -u tools/eval.py \
  -c "$CFG" \
  -o weights="$WEIGHTS" \
     EvalDataset.image_dir=images/val \
     EvalDataset.anno_path=annotations/val.json \
     EvalDataset.dataset_dir=dataset/processed_detection \
  --classwise \
  --output_eval="$EXP/metrics/val_eval_$TAG.json" \
  2>&1 | tee "$EXP/metrics/val_eval_$TAG.log"
echo "完成: $EXP/metrics/val_eval_$TAG/"
