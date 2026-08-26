#!/bin/bash
# check_env.sh — 输出 AutoDL 训练环境完整信息 + GPU Smoke Test
set -e
echo "============================================================"
echo " AIC2026 农业视觉 — 环境检查"
echo "============================================================"

echo "[1] Python"
python --version 2>&1
which python

echo ""
echo "[2] 系统驱动/CUDA"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  DRIVER_CUDA=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" || echo "NA")
  echo "驱动支持 CUDA: $DRIVER_CUDA"
else
  echo "[ERROR] 未检测到 nvidia-smi, GPU 不可用!"
  exit 1
fi

# nvcc 仅辅助记录, 不存在不阻塞
if command -v nvcc >/dev/null 2>&1; then
  TOOLKIT_CUDA=$(nvcc --version 2>/dev/null | grep -oP "release \K[0-9]+\.[0-9]+" || echo "未安装")
  echo "nvcc CUDA Toolkit: $TOOLKIT_CUDA"
else
  TOOLKIT_CUDA="未安装"
  echo "[WARN] nvcc 未安装 (不阻塞训练)"
fi

echo ""
echo "[3] PaddlePaddle / PaddleDetection / cuDNN / GPU可用性"
python - <<'PY'
import paddle
print("PaddlePaddle:", paddle.__version__)
print("compiled_with_cuda:", paddle.is_compiled_with_cuda())
print("GPU 数量:", paddle.device.cuda.device_count())
if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
    for i in range(paddle.device.cuda.device_count()):
        p = paddle.device.cuda.get_device_properties(i)
        print(f"  GPU[{i}]: {p.name}, 显存={p.total_memory/1024**3:.2f} GB")
    try:
        print("cuDNN 版本:", paddle.version.cudnn())
    except Exception as e:
        print("cuDNN 查询失败:", e)
else:
    print("[ERROR] PaddlePaddle 未启用 GPU!")
try:
    import ppdet
    print("PaddleDetection:", ppdet.__version__)
    print("ppdet 路径:", ppdet.__file__)
except Exception as e:
    print("PaddleDetection 未安装:", e)
PY

echo ""
echo "[4] GPU Smoke Test (forward + backward)"
python - <<'PY'
import paddle
try:
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
    print(f"  [PASS] GPU 前向/反向传播正常, loss={float(loss):.4f}")
except Exception as e:
    print(f"  [FAIL] GPU Smoke Test 失败: {e}")
    import sys; sys.exit(1)

# paddle.utils.run_check() 官方自检
try:
    paddle.utils.run_check()
    print("  [PASS] paddle.utils.run_check() 通过")
except Exception as e:
    print(f"  [FAIL] paddle.utils.run_check(): {e}")
PY

echo ""
echo "[5] 数据集快速核对"
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DS="$PROJECT/dataset/processed_detection"
for sp in train val test; do
  ni=$(ls "$DS/images/$sp" 2>/dev/null | wc -l)
  nl=$(ls "$DS/labels/$sp" 2>/dev/null | wc -l)
  nj=$(python -c "import json;print(len(json.load(open('$DS/annotations/$sp.json'))['images']))" 2>/dev/null || echo "NA")
  echo "  $sp: img=$ni lbl=$nl coco_img=$nj"
done
echo "  label_list 类别数: $(wc -l < "$DS/label_list.txt")"

# 回填 baseline_v1/environment.txt (机器可读段追加)
ENV_TXT="$PROJECT/experiments/baseline_v1/environment.txt"
if [[ -d "$(dirname "$ENV_TXT")" ]]; then
  {
    echo ""
    echo "# ===== check_env.sh 自动回填 $(date) ====="
    python - <<'PY'
import paddle, platform, subprocess
print("python:", platform.python_version())
print("paddle:", paddle.__version__)
print("compiled_with_cuda:", paddle.is_compiled_with_cuda())
print("gpu_count:", paddle.device.cuda.device_count())
if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count()>0:
    p=paddle.device.cuda.get_device_properties(0)
    print("gpu_name:", p.name)
    print("gpu_vram_gb:", round(p.total_memory/1024**3,2))
    print("cudnn:", paddle.version.cudnn())
try:
    import ppdet; print("ppdet:", ppdet.__version__)
except Exception as e:
    print("ppdet:", e)
PY
    echo "cuda_driver: ${DRIVER_CUDA:-NA}"
    echo "nvcc: ${TOOLKIT_CUDA:-NA}"
  } >> "$ENV_TXT"
  echo "  (已追加到 $ENV_TXT)"
fi
echo "============================================================"