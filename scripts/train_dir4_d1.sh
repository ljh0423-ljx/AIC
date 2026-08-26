#!/bin/bash
# train_dir4_d1.sh — 方向4 D1 密集感知 VFL 目标校准
# 唯一变量: PPYOLOEHead.use_density_calibration=True (dense>=5 正锚 gt_score'=max(IoU,0.65))
# 数据=exiffix | epoch=100 | bs=16 | lr=0.002 | seed=0 (--enable_ce True)
# A0 对照组复用方向2 EXIF-Fix best checkpoint, 不重训。
# 用法: MODE=smoke|full bash scripts/train_dir4_d1.sh    (默认 full)
set -e
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${MODE:-full}"
case "$MODE" in
  smoke) CFG="$PROJECT/configs/ablation_dir4_D1_smoke.yml"; EXP="$PROJECT/experiments/direction4/d1_smoke";;
  full)  CFG="$PROJECT/configs/ablation_dir4_D1.yml";       EXP="$PROJECT/experiments/direction4/d1_formal";;
  *) echo "[ERROR] MODE 仅允许 smoke / full"; exit 1;;
esac

PD="$PROJECT/PaddleDetection"
DS="$PROJECT/dataset/processed_detection_exiffix"
SAVE_DIR="$EXP/checkpoints"
VDL_DIR="$EXP/logs"

echo "============================================================"
echo " 方向4 D1 (density-aware VFL) ${MODE} 训练 (${CFG})"
echo " 数据: $DS | 输出: $EXP"
echo "============================================================"

# 0. 前置校验: exiffix 数据副本存在 (与 A0 同数据基础)
[[ -d "$DS" ]] || { echo "[ERROR] 数据副本不存在: $DS"; exit 1; }
echo "[0] 数据副本存在 (exiffix, 与 A0 一致)"

# 1. 数据软链 (实验数据基础 = exiffix, 与 A0 一致)
mkdir -p "$PD/dataset"
ln -sfn "$DS" "$PD/dataset/processed_detection"

# 2. 输出目录 + 存档 (环境/git/命令/配置)
mkdir -p "$SAVE_DIR" "$VDL_DIR"
{
  echo "方向4 D1 (${MODE}) — 环境快照 ($(date '+%F %T'))"
  echo "===="
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
  python -c "import paddle;print('Paddle',paddle.__version__,'CUDA',paddle.version.cuda,'GPU',paddle.device.cuda.device_count())" 2>/dev/null || true
  python -c "import platform;print('Python',platform.python_version())"
} > "$EXP/environment.txt"
(cd "$PD" && git rev-parse HEAD > "$EXP/git_commit.txt" 2>/dev/null || echo "not-a-git-repo" > "$EXP/git_commit.txt")
cp "$CFG" "$EXP/config.yml"
cp "$PROJECT/configs/ppyoloe_plus_crn_s_100e_agrivision.yml" "$EXP/config_base.yml"
{
  echo "方向4 D1 (${MODE}) — 训练命令 (seed=0 via --enable_ce True, 数据=exiffix)"
  echo "======================================================"
  echo "MODE=${MODE} bash scripts/train_dir4_d1.sh"
  echo "配置: $CFG   数据: $DS"
} > "$EXP/train_command.txt"

# 3. 机制校验 (仅 smoke): 35 项单元检查必须 0 FAIL
if [ "$MODE" = "smoke" ]; then
  echo "[3] 运行 D1 机制校验 (35 项单元检查) ..."
  python "$PROJECT/scripts/d1_mechanism_check.py" 2>&1 | tee "$EXP/mechanism_check.log"
  grep -q "0 FAIL" "$EXP/mechanism_check.log" || {
    echo "[ERROR] 机制校验未通过, 立即停止, 不得进入 smoke 训练"; exit 1; }
fi

# 4. 训练 (smoke 2 epoch, D1_DEBUG 输出真实批次校准统计; seed=0; TEST 不参与)
cd "$PD"
echo "[4] 启动 ${MODE} 训练 (seed=0, bs=16, lr=0.002) ..."
if [ "$MODE" = "smoke" ]; then export D1_DEBUG=1; fi
python -u tools/train.py \
  -c "$CFG" \
  --eval \
  --use_vdl=true \
  --vdl_log_dir="$VDL_DIR" \
  --enable_ce True \
  -o save_dir="$SAVE_DIR" \
  2>&1 | tee "$EXP/train.log"

echo ""
echo "训练完成: $SAVE_DIR/"
