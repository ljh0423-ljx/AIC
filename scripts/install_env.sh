#!/bin/bash
# install_env.sh — AutoDL 环境安装 (先检测 GPU/CUDA, 再装兼容 PaddlePaddle)
# 用法: bash scripts/install_env.sh
#   可用环境变量覆盖:
#     PADDLE_VERSION  指定 PaddlePaddle 版本 (默认 2.6.2, 可覆盖)
#     CUDA_TAG        强制指定 CUDA tag: cu118 / cu120 (默认自动按驱动版本选)
set -e
echo "============================================================"
echo " AIC2026 农业视觉 — 环境安装"
echo "============================================================"
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PD="$PROJECT/PaddleDetection"

# ---- 0. 检测 GPU (nvidia-smi 为唯一硬件门槛) ----
echo "[0] 检测 GPU ..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] 未检测到 nvidia-smi, GPU 硬件/驱动不可用, 终止安装。"
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
DRIVER_CUDA=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" || echo "")
echo "  驱动版本: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
echo "  驱动支持 CUDA: ${DRIVER_CUDA:-未知}"

# nvcc 仅作辅助记录, 不存在不阻塞
TOOLKIT_CUDA=""
if command -v nvcc >/dev/null 2>&1; then
  TOOLKIT_CUDA=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+" || echo "")
  echo "  nvcc CUDA Toolkit: $TOOLKIT_CUDA"
else
  echo "  [WARN] nvcc 未安装 (不阻塞训练, 仅影响需编译的算子)"
fi

# ---- 1. 确定 CUDA tag (按驱动支持的最高 CUDA 版本选) ----
# 核心规则:
#   1) 若用户强制指定 CUDA_TAG, 直接使用
#   2) 否则用驱动支持的 CUDA 版本 (DRIVER_CUDA), 而非 nvcc
#   3) 驱动 CUDA >= 12.0 → cu120; 驱动 CUDA >= 11.0 → cu118
CUDA_TAG="${CUDA_TAG:-}"
if [[ -z "$CUDA_TAG" ]]; then
  REF="${DRIVER_CUDA:-$TOOLKIT_CUDA}"
  if [[ -z "$REF" ]]; then
    echo "[WARN] 无法识别 CUDA 版本, 默认 cu118"
    CUDA_TAG="cu118"
  else
    MAJOR=$(echo "$REF" | cut -d. -f1)
    if [[ "$MAJOR" == "12" ]]; then
      CUDA_TAG="cu120"   # 12.x 用 cu120 wheel (前向兼容 12.1/12.2)
    elif [[ "$MAJOR" == "11" ]]; then
      CUDA_TAG="cu118"
    else
      echo "[WARN] CUDA 主版本 $MAJOR 不在 11/12, 默认 cu118"
      CUDA_TAG="cu118"
    fi
  fi
fi
echo "  选用 CUDA tag: $CUDA_TAG"

# ---- 2. 确定 PaddlePaddle 版本 ----
# PaddleDetection v2.9.0 兼容 PaddlePaddle 2.5.x / 2.6.x。
# 默认推荐 2.6.2 (最新稳定 2.6 系列, AutoDL 实测兼容性最佳)。
# 如需 2.5: PADDLE_VERSION=2.5.2 bash scripts/install_env.sh
PADDLE_VERSION="${PADDLE_VERSION:-2.6.2}"
echo "  PaddlePaddle 版本: $PADDLE_VERSION (兼容 PaddleDetection v2.9.0)"

# ---- 3. 安装 PaddlePaddle-GPU ----
echo "[3] 安装 paddlepaddle-gpu==$PADDLE_VERSION ($CUDA_TAG) ..."
URL="https://www.paddlepaddle.org.cn/whl/linux/linux-gpu-${CUDA_TAG}.html"
python -m pip install "paddlepaddle-gpu==$PADDLE_VERSION" -f "$URL" \
  || { echo "[ERROR] 安装失败。请检查: CUDA 版本 / pip 源 / 网络。"; exit 1; }

# ---- 4. GPU 环境三重验证 (nvidia-smi 已通过, 现在验证 Paddle 实际可用) ----
echo "[4] GPU 环境验证 (Paddle 实际检查) ..."
python - <<'PY'
import sys, traceback

passed = True
checks = []

# Check 1: import paddle
try:
    import paddle
    checks.append(("import paddle", True, paddle.__version__))
except Exception as e:
    checks.append(("import paddle", False, str(e)))
    passed = False

# Check 2: compiled_with_cuda
try:
    ok = paddle.is_compiled_with_cuda()
    checks.append(("paddle.is_compiled_with_cuda()", ok, str(ok)))
    if not ok:
        passed = False
except Exception as e:
    checks.append(("paddle.is_compiled_with_cuda()", False, str(e)))
    passed = False

# Check 3: GPU count
try:
    cnt = paddle.device.cuda.device_count()
    checks.append(("paddle.device.cuda.device_count()", cnt > 0, f"count={cnt}"))
    if cnt == 0:
        passed = False
except Exception as e:
    checks.append(("paddle.device.cuda.device_count()", False, str(e)))
    passed = False

# Check 4: GPU properties
try:
    p = paddle.device.cuda.get_device_properties(0)
    vram_gb = p.total_memory / 1024**3
    checks.append(("GPU properties", True, f"{p.name}, {vram_gb:.2f} GB, cuDNN={paddle.version.cudnn()}"))
except Exception as e:
    checks.append(("GPU properties", False, str(e)))
    passed = False

# Check 5: paddle.utils.run_check() — Paddle 官方环境自检
try:
    paddle.utils.run_check()
    checks.append(("paddle.utils.run_check()", True, "PASS"))
except Exception as e:
    checks.append(("paddle.utils.run_check()", False, str(e)))
    passed = False

# Check 6: GPU Smoke Test — 实际 forward/backward
try:
    import paddle
    paddle.device.set_device("gpu:0")
    x = paddle.randn([2, 3, 64, 64])
    # flatten(x,1) 为 [2, 3*64*64]=[2, 12288], Linear 输入维度须为 12288
    fc = paddle.nn.Linear(12288, 10)
    opt = paddle.optimizer.Momentum(parameters=fc.parameters(), learning_rate=0.001)
    out = fc(paddle.flatten(x, 1))
    loss = paddle.sum(out)
    loss.backward()
    opt.step()
    opt.clear_grad()
    checks.append(("GPU Smoke Test (forward+backward)", True, f"loss={float(loss):.4f}"))
except Exception as e:
    checks.append(("GPU Smoke Test (forward+backward)", False, str(e)))
    passed = False

# Print results
for name, ok, info in checks:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}: {info}")

if not passed:
    print("\n[FAIL] GPU 环境验证未通过!")
    print("  可能原因: Paddle 版本与 CUDA 不匹配 / GPU 驱动异常 / CUDA toolkit 缺失。")
    print("  建议: 检查 nvidia-smi 输出; 尝试 PADDLE_VERSION=2.5.2 bash install_env.sh")
    sys.exit(1)

print("\n[OK] GPU 环境验证全部通过。")
PY

# ---- 5. 安装 PaddleDetection 依赖 ----
echo "[5] 安装 PaddleDetection 依赖 (v2.9.0) ..."
cd "$PD"
python -m pip install -r requirements.txt
python -m pip install -e . 2>/dev/null || echo "    (ppdet -e 安装跳过, 不影响训练)"

# ---- 6. 最终验证: PaddleDetection import ----
echo "[6] PaddleDetection 验证 ..."
python - <<'PY'
try:
    import ppdet
    print(f"  PaddleDetection: {ppdet.__version__}")
except Exception as e:
    print(f"  [WARN] PaddleDetection import 失败: {e}")
    print("    (不影响训练主流程, 训练脚本会自行处理)")

import paddle
print(f"  Paddle: {paddle.__version__} | GPU: {paddle.device.cuda.device_count()}")
PY

echo ""
echo "============================================================"
echo " 安装完成!"
echo "============================================================"
echo " 下一步: bash scripts/check_gpu_config.sh"
echo "   → 检测显存, 给出 batch_size / lr 建议"
echo "   → 不自动修改配置, 仅输出建议值"
echo "============================================================"