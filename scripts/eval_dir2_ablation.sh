#!/bin/bash
# eval_dir2_ablation.sh — 方向2 消融 A0/A1/A2 × best/final 的 VAL 集评估
# 仅 VAL; TEST 封闭。三臂均用 EXIF-Fix 数据 (A0 = Arm B 已有 checkpoint)。
# 用法: bash scripts/eval_dir2_ablation.sh <A0|A1|A2> <best|final>
set -e
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PD="$PROJECT/PaddleDetection"
CFG="$PROJECT/configs/ppyoloe_plus_crn_s_100e_agrivision.yml"
ARM="${1:?ARM=A0|A1|A2}"
TAG="${2:?TAG=best|final}"
case "$ARM" in
  A0) EXP="$PROJECT/experiments/ablation_data_exiffix";;
  A1) EXP="$PROJECT/experiments/ablation_dir2_A1";;
  A2) EXP="$PROJECT/experiments/ablation_dir2_A2";;
  *) echo "[ERROR] ARM 无效"; exit 1;;
esac
case "$TAG" in
  best|final) :;;
  *) echo "[ERROR] TAG 无效"; exit 1;;
esac
DS="$PROJECT/dataset/processed_detection_exiffix"
OUT="$PROJECT/experiments/direction2/val_eval/${ARM}_${TAG}"
case "$TAG" in
  best)  WNAME="best_model.pdparams";;
  final) WNAME="model_final.pdparams";;
esac
WEIGHTS="$(find "$EXP/checkpoints" -name "$WNAME" | head -1)"
[[ -n "$WEIGHTS" && -f "$WEIGHTS" ]] || { echo "[ERROR] 权重不存在: $EXP/checkpoints/$WNAME"; exit 1; }
WEIGHTS="$(readlink -f "$WEIGHTS")"

mkdir -p "$PD/dataset"
ln -sfn "$DS" "$PD/dataset/processed_detection"
mkdir -p "$OUT"
cd "$PD"
echo "[VAL eval] arm=$ARM tag=$TAG 权重=$WEIGHTS 数据=$DS"
python -u tools/eval.py \
  -c "$CFG" \
  -o weights="$WEIGHTS" \
     EvalDataset.image_dir=images/val \
     EvalDataset.anno_path=annotations/val.json \
     EvalDataset.dataset_dir=dataset/processed_detection \
  --classwise \
  --output_eval="$OUT.json" \
  2>&1 | tee "$OUT.log"
echo "完成: $OUT/ (bbox.json + .log)"
