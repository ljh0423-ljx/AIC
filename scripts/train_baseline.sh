#!/bin/bash
# train_baseline.sh — 启动 PP-YOLOE+-s Baseline 训练 (AutoDL GPU)
# 自动: 进入PaddleDetection / 建数据集软链 / 检查配置 / 训练 / 存日志/checkpoint/best
# 用法:
#   bash scripts/train_baseline.sh                       # 用配置默认 (bs=8, lr=0.001)
#   BATCH_SIZE=4 BASE_LR=0.0005 bash scripts/train_baseline.sh   # 按显存调整 (lr 须同步)
set -e
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PD="$PROJECT/PaddleDetection"
# 数据源: 使用清洁副本 processed_detection_clean
# (原始 processed_detection 含 2 张 0 字节损坏图: train 1 张 / val 1 张, 会致训练/验证崩溃。
#  清洁副本已移除这 2 张图及其 COCO 标注/空标签, 原始数据保持不动。统计见 experiments/baseline_v1/data_prep_notes.txt)
DS="$PROJECT/dataset/processed_detection_clean"
CFG="$PROJECT/configs/ppyoloe_plus_crn_s_100e_agrivision.yml"
EXP="$PROJECT/experiments/baseline_v1"
SAVE_DIR="$EXP/checkpoints"
VDL_DIR="$EXP/logs"

echo "============================================================"
echo " PP-YOLOE+-s Baseline v1 训练"
echo " 配置: $CFG"
echo " 输出: $EXP"
echo "============================================================"

# 1. 数据集软链 (PaddleDetection 约定 dataset/<name>)
mkdir -p "$PD/dataset"
ln -sfn "$DS" "$PD/dataset/processed_detection"
echo "[1] 数据集软链: $PD/dataset/processed_detection -> $DS"

# 2. 检查配置
[[ -f "$CFG" ]] || { echo "[ERROR] 配置不存在: $CFG"; exit 1; }
echo "[2] 配置就绪"

# 3. 环境快速检查
echo "[3] 环境:"
python -c "import paddle;print('  Paddle',paddle.__version__,'GPU',paddle.device.cuda.device_count())" \
  || { echo "Paddle 未装或 GPU 不可用, 先 bash scripts/install_env.sh"; exit 1; }

# 4. batch_size / lr 覆盖 (默认沿用配置; 环境变量非空则覆盖并提示)
BATCH_SIZE="${BATCH_SIZE:-}"
BASE_LR="${BASE_LR:-}"
OVERRIDE="save_dir=$SAVE_DIR"
if [[ -n "$BATCH_SIZE" ]]; then
  OVERRIDE="$OVERRIDE TrainReader.batch_size=$BATCH_SIZE"
  echo "[4] 覆盖 batch_size=$BATCH_SIZE"
fi
if [[ -n "$BASE_LR" ]]; then
  OVERRIDE="$OVERRIDE LearningRate.base_lr=$BASE_LR"
  echo "    覆盖 base_lr=$BASE_LR (请确认已按 bs 线性缩放)"
fi
# 一致性校验: 改了 bs 却没改 lr -> 警告
if [[ -n "$BATCH_SIZE" && -z "$BASE_LR" ]]; then
  echo "    [WARN] 改了 batch_size 未改 base_lr, 建议 BASE_LR=0.001*${BATCH_SIZE}/8"
fi

# 5. 输出目录
mkdir -p "$SAVE_DIR" "$VDL_DIR"

# 6. 训练 (--eval 每 epoch 评估 val; --use_vdl 记录曲线; test 集不参与训练)
cd "$PD"
echo "[5] 启动训练 ..."
python -u tools/train.py \
  -c "$CFG" \
  --eval \
  --use_vdl=true \
  --vdl_log_dir="$VDL_DIR" \
  -o $OVERRIDE

echo ""
echo "训练完成。"
echo "  checkpoints: $SAVE_DIR/ppyoloe_plus_crn_s_100e_agrivision/"
echo "  VDL 曲线:    $VDL_DIR  (visualdl --logdir $VDL_DIR)"
echo "  评估(test):  bash scripts/eval_baseline.sh"
