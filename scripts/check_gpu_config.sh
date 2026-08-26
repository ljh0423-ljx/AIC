#!/bin/bash
# check_gpu_config.sh — 检测 GPU 显存, 给出 batch_size / base_lr 建议 (不自动改配置, 仅建议)
# 用法: bash scripts/check_gpu_config.sh
echo "============================================================"
echo " GPU 显存检测与 batch_size / lr 配置建议"
echo " (PP-YOLOE+-s, 输入640, 13类; 线性缩放规则: lr = 0.001 * batch_size/8)"
echo " (CosineDecay.max_epochs = 100 = epoch, lr 余弦衰减到 0)"
echo "============================================================"

if ! python -c "import paddle; assert paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count()>0" 2>/dev/null; then
  echo "[ERROR] PaddlePaddle GPU 不可用, 先运行 bash scripts/install_env.sh"
  exit 1
fi

python - <<'PY'
import paddle
p = paddle.device.cuda.get_device_properties(0)
vram_gb = p.total_memory / 1024**3
print(f"GPU: {p.name}")
print(f"显存: {vram_gb:.2f} GB")
print(f"cuDNN: {paddle.version.cudnn()}")
print()

# 建议 (PP-YOLOE+-s @ 640, 经验估算: bs=8 约需 6~8GB)
if vram_gb >= 16:
    bs, note = 16, "充裕, 可用 bs=16 加速; 也可升级 PP-YOLOE+-m"
elif vram_gb >= 11:
    bs, note = 8, "推荐配置, 保持默认 bs=8"
elif vram_gb >= 8:
    bs, note = 8, "可用 bs=8 (接近上限, 若 OOM 降到 4)"
elif vram_gb >= 6:
    bs, note = 4, "建议 bs=4"
elif vram_gb >= 4:
    bs, note = 2, "仅够 bs=2, 训练较慢"
else:
    bs, note = 2, "显存不足, 强制 bs=2, 极可能 OOM, 建议换更大显存 GPU"

lr = 0.001 * bs / 8.0
print(f"建议 batch_size = {bs}   ({note})")
print(f"建议 base_lr    = {lr:.6f}   (线性缩放: 0.001 * {bs}/8)")
print(f"CosineDecay.max_epochs 保持 100 (=epoch), lr 余弦衰减到 0")
print(f"LinearWarmup.epochs 保持 5")
print()
print("应用方式 (二选一):")
print(f"  [A] 环境变量启动:  BATCH_SIZE={bs} BASE_LR={lr} bash scripts/train_baseline.sh")
print(f"  [B] 直接改配置 configs/ppyoloe_plus_crn_s_100e_agrivision.yml 中:")
print(f"        TrainReader.batch_size: {bs}")
print(f"        LearningRate.base_lr: {lr}")
print()
print("注意: 改 batch_size 必须同步改 base_lr, 否则训练不稳定; scheduler (max_epochs=100, warmup=5) 无需改。")
PY
echo "============================================================"