#!/bin/bash
# train_ablation.sh — Ablation-Data 实验单臂训练 (设计: ABLATION_PLAN.md §6)
# 用法: DATA_ARM=clean|exiffix bash scripts/train_ablation.sh
# 两臂除数据副本外逐字段一致: 模型 PP-YOLOE+-s / epoch=100 / bs=16 / lr=0.002 /
#   seed=0 (--enable_ce True) / 优化器 / 增强 / 评估 (仅 VAL)。
# TEST 集全程不参与训练与评估。
set -e

DATA_ARM="${DATA_ARM:?必须指定 DATA_ARM=clean 或 exiffix}"
case "$DATA_ARM" in
  clean|exiffix) ;;
  *) echo "[ERROR] DATA_ARM 仅允许 clean / exiffix"; exit 1 ;;
esac

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PD="$PROJECT/PaddleDetection"
DS="$PROJECT/dataset/processed_detection_${DATA_ARM}"
CFG="$PROJECT/configs/ppyoloe_plus_crn_s_100e_agrivision.yml"
EXP="$PROJECT/experiments/ablation_data_${DATA_ARM}"
SAVE_DIR="$EXP/checkpoints"
VDL_DIR="$EXP/logs"

echo "============================================================"
echo " Ablation-Data ${DATA_ARM} 臂训练"
echo " 数据: $DS"
echo " 输出: $EXP"
echo "============================================================"

# 0. 前置校验: exiffix 臂必须先过 validate_exiffix.py
if [[ "$DATA_ARM" == "exiffix" ]]; then
  [[ -f "$PROJECT/experiments/baseline_v1/metrics/VALIDATION_REPORT.txt" ]] || \
    { echo "[ERROR] 未找到 VALIDATION_REPORT.txt, 先运行 validate_exiffix.py"; exit 1; }
  grep -q "ALL PASS" "$PROJECT/experiments/baseline_v1/metrics/VALIDATION_REPORT.txt" || \
    { echo "[ERROR] validate_exiffix.py 未全部 PASS, 禁止训练 exiffix 臂"; exit 1; }
  echo "[0] validate_exiffix.py 已 ALL PASS"
fi

# 1. 数据副本存在性 + 软链
[[ -d "$DS" ]] || { echo "[ERROR] 数据副本不存在: $DS"; exit 1; }
mkdir -p "$PD/dataset"
ln -sfn "$DS" "$PD/dataset/processed_detection"
echo "[1] 数据集软链: $PD/dataset/processed_detection -> $DS"

# 2. 输出目录
mkdir -p "$SAVE_DIR" "$VDL_DIR"

# 3. 记录环境 / git commit / 命令 (用户要求: 保存完整日志、checkpoint、环境、git commit、配置)
{
  echo "Ablation-Data ${DATA_ARM} — 环境快照 ($(date '+%F %T'))"
  echo "===="
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
  python -c "import paddle;print('Paddle',paddle.__version__,'CUDA',paddle.version.cuda,'GPU',paddle.device.cuda.device_count(),'dev',paddle.device.get_device())" 2>/dev/null || true
  python -c "import platform;print('Python',platform.python_version())"
  python -c "import sys;print('site',sys.prefix)"
} > "$EXP/environment.txt"
(cd "$PD" && git rev-parse HEAD > "$EXP/git_commit.txt" 2>/dev/null || echo "not-a-git-repo" > "$EXP/git_commit.txt")
cp "$CFG" "$EXP/config.yml"
cp "$PROJECT/configs/datasets/agrivision_detection.yml" "$EXP/config_dataset.yml" 2>/dev/null || true
{
  echo "Ablation-Data ${DATA_ARM} — 训练命令 (seed=0 via --enable_ce True)"
  echo "======================================================"
  echo "bash scripts/train_ablation.sh  (DATA_ARM=${DATA_ARM})"
  echo "配置: $CFG   数据: $DS"
} > "$EXP/train_command.txt"

# 4. 训练 (--eval 每 epoch VAL; --enable_ce True -> set_random_seed(0); TEST 不参与)
cd "$PD"
echo "[2] 启动训练 (seed=0, bs=16, lr=0.002, epoch=100) ..."
python -u tools/train.py \
  -c "$CFG" \
  --eval \
  --use_vdl=true \
  --vdl_log_dir="$VDL_DIR" \
  --enable_ce True \
  -o save_dir="$SAVE_DIR" \
     TrainReader.batch_size=16 \
     LearningRate.base_lr=0.002 \
  2>&1 | tee "$EXP/train.log"

echo ""
echo "训练完成。"
echo "  checkpoints: $SAVE_DIR/ppyoloe_plus_crn_s_100e_agrivision/"
echo "  日志:        $EXP/train.log"
