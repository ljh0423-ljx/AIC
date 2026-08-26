#!/bin/bash
# eval_baseline.sh — 在 TEST 集评估 (test 仅用于最终评估, 不参与训练)
# 用法: bash scripts/eval_baseline.sh [WEIGHTS_PATH]
set -e
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PD="$PROJECT/PaddleDetection"
CFG="$PROJECT/configs/ppyoloe_plus_crn_s_100e_agrivision.yml"
WEIGHTS="${1:-$PROJECT/experiments/baseline_v1/checkpoints/model_final.pdparams}"
# 解析为绝对路径 (eval.py 在 PaddleDetection 目录下运行, 相对路径会解析错)
[[ "$WEIGHTS" != /* ]] && WEIGHTS="$PROJECT/$WEIGHTS"
METRIC_DIR="$PROJECT/experiments/baseline_v1/metrics"

echo "============================================================"
echo " PP-YOLOE+-s Baseline v1 — TEST 集评估"
echo " 权重: $WEIGHTS"
echo " (TEST 集仅用于最终评估, 不参与训练/调参)"
echo "============================================================"

[[ -f "$WEIGHTS" ]] || { echo "[ERROR] 权重不存在: $WEIGHTS"; exit 1; }

# 数据集软链
mkdir -p "$PD/dataset"
ln -sfn "$PROJECT/dataset/processed_detection_clean" "$PD/dataset/processed_detection"

mkdir -p "$METRIC_DIR"
cd "$PD"

# 评估在 TEST 集 (覆盖 EvalDataset 指向 test; TrainDataset 不加载, 不影响)
python -u tools/eval.py \
  -c "$CFG" \
  -o weights="$WEIGHTS" \
     EvalDataset.image_dir=images/test \
     EvalDataset.anno_path=annotations/test.json \
     EvalDataset.dataset_dir=dataset/processed_detection \
  --classwise \
  --output_eval="$METRIC_DIR/test_eval.json" \
  2>&1 | tee "$METRIC_DIR/test_eval.log"

echo ""
echo "评估完成。结果:"
echo "  日志: $METRIC_DIR/test_eval.log"
echo "  JSON: $METRIC_DIR/test_eval.json"
echo "  (COCO metric 输出 mAP@50:95 / mAP@50 / 每类 AP / P / R)"
