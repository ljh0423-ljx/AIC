#!/bin/bash
# train_dir2_ablation.sh — 方向2 消融训练 (A1: CGPM guided / A2: CGPM fixed)
# 唯一变量: 分类损失机制; 数据=exiffix, epoch=100, bs=16, lr=0.002, seed=0 (--enable_ce True)。
# A0 = 复用 Arm B (exiffix, VFL) 已有 checkpoint, 不重训。
# 用法: ARM=A1 bash scripts/train_dir2_ablation.sh    (ARM ∈ A1|A2)
set -e
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARM="${ARM:?必须指定 ARM=A1 或 A2}"
case "$ARM" in
  A1) CFG="$PROJECT/configs/ablation_dir2_A1_cgpm.yml";;
  A2) CFG="$PROJECT/configs/ablation_dir2_A2_fixed.yml";;
  *) echo "[ERROR] ARM 仅允许 A1 / A2"; exit 1;;
esac

PD="$PROJECT/PaddleDetection"
DS="$PROJECT/dataset/processed_detection_exiffix"
EXP="$PROJECT/experiments/ablation_dir2_${ARM}"
SAVE_DIR="$EXP/checkpoints"
VDL_DIR="$EXP/logs"

echo "============================================================"
echo " 方向2 消融 ${ARM} 训练 (${CFG})"
echo " 数据: $DS (EXIF 修复, Arm B 数据基础)"
echo " 输出: $EXP"
echo "============================================================"

# 0. 前置校验: exiffix 副本 + confusion prior
[[ -f "$PROJECT/experiments/baseline_v1/metrics/VALIDATION_REPORT.txt" ]] && \
  grep -q "ALL PASS" "$PROJECT/experiments/baseline_v1/metrics/VALIDATION_REPORT.txt" || \
  { echo "[ERROR] EXIF 修复校验未 ALL PASS"; exit 1; }
[[ -f "$PROJECT/experiments/direction2/confusion_prior_train_only.json" ]] || \
  { echo "[ERROR] confusion prior 缺失"; exit 1; }
echo "[0] EXIF 校验 ALL PASS; confusion prior 存在"

# 1. 数据软链
[[ -d "$DS" ]] || { echo "[ERROR] 数据副本不存在: $DS"; exit 1; }
mkdir -p "$PD/dataset"
ln -sfn "$DS" "$PD/dataset/processed_detection"

# 2. 输出目录 + 存档 (环境/git/命令/配置)
mkdir -p "$SAVE_DIR" "$VDL_DIR"
{
  echo "方向2 消融 ${ARM} — 环境快照 ($(date '+%F %T'))"
  echo "===="
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
  python -c "import paddle;print('Paddle',paddle.__version__,'CUDA',paddle.version.cuda,'GPU',paddle.device.cuda.device_count())" 2>/dev/null || true
  python -c "import platform;print('Python',platform.python_version())"
} > "$EXP/environment.txt"
(cd "$PD" && git rev-parse HEAD > "$EXP/git_commit.txt" 2>/dev/null || echo "not-a-git-repo" > "$EXP/git_commit.txt")
cp "$CFG" "$EXP/config.yml"
cp "$PROJECT/configs/ppyoloe_plus_crn_s_100e_agrivision.yml" "$EXP/config_base.yml"
cp "$PROJECT/experiments/direction2/confusion_prior_train_only.json" "$EXP/confusion_prior_train_only.json"
{
  echo "方向2 消融 ${ARM} — 训练命令 (seed=0 via --enable_ce True, 数据=exiffix)"
  echo "======================================================"
  echo "bash scripts/train_dir2_ablation.sh  (ARM=${ARM})"
  echo "配置: $CFG   数据: $DS"
} > "$EXP/train_command.txt"

# 3. 训练 (--eval 每 epoch VAL; seed=0; TEST 不参与)
cd "$PD"
echo "[2] 启动训练 (seed=0, bs=16, lr=0.002, epoch=100) ..."
python -u tools/train.py \
  -c "$CFG" \
  --eval \
  --use_vdl=true \
  --vdl_log_dir="$VDL_DIR" \
  --enable_ce True \
  -o save_dir="$SAVE_DIR" \
  2>&1 | tee "$EXP/train.log"

echo ""
echo "训练完成: $EXP/checkpoints/"
