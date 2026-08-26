# GPU_SINGLE_INFERENCE_REPORT — AutoDL GPU 单图推理验证报告

- **验证日期**: 2026-08-18
- **项目根目录**: `/root/autodl-tmp/AIC2026_AgriVision/`
- **验证性质**: 最终 GPU 单图推理验证。**未修改任何项目代码 / 未修改 best_model / 未修改原始数据集 / 未训练·重训·调参 / 未访问 TEST（test.json、images/test）/ 未做模型转换 / 未做 RK3588 部署 / 未启动 Web / 未做批量推理。**
- **锁定模型**: `experiments/direction_m/M_SCALEUP_100e/checkpoints/best_model.pdparams`
- **锁定配置**: `configs/ppyoloe_plus_crn_m_100e_agrivision.yml`
- **验证图片**: `dataset/processed_detection_exiffix/images/val/TRAIN_000004_apple_scab.jpg`（仅 VAL）

---

## 一、GPU / 环境

| 项 | 值 |
|---|---|
| GPU 型号 | **NVIDIA GeForce RTX 4090**（compute capability 8.9） |
| 显存 | **24564 MiB（24 GB）**；推理前占用 3 MiB → 模型加载后 609 MiB → 推理后 741 MiB（模型确实驻留 GPU） |
| PaddlePaddle | **2.6.2（cu118）** |
| `is_compiled_with_cuda` | True |
| CUDA（编译期 runtime） | **11.8**（驱动 API 13.0 / cuDNN 运行时 9.1，paddle 编译期 8.6.0） |
| 设备解析 | `paddle.device.get_device()` = **gpu:0**；模型参数 `place = Place(gpu:0)`；backend = **paddle_gpu** |
| **设备确认** | **模型实际运行在 gpu:0，非 CPU**（backend.device=gpu、backend_name=paddle_gpu、参数 place=gpu:0、显存占用上升三重确认） |

## 二、模型 / 配置 / 图片

- 模型：PP-YOLOE+-m（13 类，input 640×640，参数量 23.57M，pdparams 94.3 MB）
- 配置：`ppyoloe_plus_crn_m_100e_agrivision.yml`（_BASE_ 链已解析，类别映射来自 `annotations/val.json`，13 类）
- 图片：`TRAIN_000004_apple_scab.jpg`（**300×360**，VAL 子集；该图 GT = 6 × Apple Scab Leaf）
- 置信度阈值：**0.5**（`DEFAULT_CONF_THRESHOLD`）

## 三、检测结果（GPU，阈值 0.5）

| 项 | 值 |
|---|---|
| 原始候选（> NMS 阈值） | 300 个 |
| 阈值过滤后目标数 | **1** |
| class_id | **9** |
| class_name | **Apple Scab Leaf** |
| confidence | **0.569851** |
| bbox | **x=166.68, y=105.84, width=104.96, height=87.61**（左上角+宽高，原图坐标） |

> 单图 GT 为 6 个目标，模型在此图上以 0.5 阈值检出 1 个高置信目标——该行为与既有 CPU 验证（同样检出 1 个）完全一致，属模型在该图上的既有表现，非本次环境引入。

## 四、真实耗时 / 实际 FPS（单次完整推理，无 warmup，GPU）

| 阶段 | 耗时 |
|---|---|
| 模型加载（一次性） | **2.191 s** |
| 预处理 | **4.3 ms** |
| 纯推理（前向，含输入上板） | **504.3 ms** |
| 后处理 | **1.3 ms** |
| **总耗时（流水线：预处理+推理+后处理）** | **509.8 ms** |
| 总耗时（含模型加载） | 2701.3 ms |
| **实际 FPS（流水线）** | **1.96** |
| FPS（纯前向） | 1.98 |

> 说明：单图首跑包含 GPU 内核初始化等一次性开销；流水线 FPS≈1.96 为真实单图吞吐。迁移清单 §11 预估 GPU FPS≈16 为稳态连续推理口径，本次为**单图一次性**验证，属不同口径，特此注明。

## 五、CPU / GPU 结果对比

- **CPU 基线来源**: `web/demo_outputs/run_20260817_173307/detections/TRAIN_000004_apple_scab.json`（Windows CPU，Paddle 3.3.0，阈值 0.5）；另有 `inference/outputs/run_20260817_154849`（CPU 单图，FPS 0.42）。

| 对比项 | CPU（基线） | GPU（本次） | 一致性 |
|---|---|---|---|
| 目标数 | 1 | 1 | ✅ 一致 |
| class_id / class_name | 9 / Apple Scab Leaf | 9 / Apple Scab Leaf | ✅ 一致 |
| confidence | 0.569842 | 0.569851 | ✅ 差 9e-6（浮点非逐位一致，量级一致） |
| bbox (x,y,w,h) | (166.68, 105.84, 104.96, 87.61) | (166.68, 105.84, 104.96, 87.61) | ✅ 完全一致（各轴 delta=0.0） |
| bbox 格式 | x,y,width,height（原图坐标） | 同左 | ✅ 格式一致 |
| 推理耗时 | 1651.91 ms | 504.29 ms | GPU 约 3.3× 快（单图首跑口径） |
| 整体 FPS | ≈0.42（CPU 单图） | 1.96（GPU 流水线） | GPU ≈ 4.7× |

**结论**：类别映射、bbox 格式、目标数与置信度量级均正常一致；GPU 与 CPU 结果对齐，未出现类别错位、bbox 坐标系错乱或数量异常。未要求逐位一致（符合约束）。

## 六、TEST 隔离检查

- **静态**: 本次仅读取 VAL 图片、模型、配置、`annotations/val.json`；未引用 test.json / images/test。
- **运行时**: 脚本安装 `sys.addaudithook` 审计钩子，全程监控 `open/os.open` 事件——
  **violations = []，未打开任何 `test.json` / `images/test` 文件** ✅
- 输出写入 `experiments/final_evaluation/gpu_inference_validation/`（独立 GPU 目录），**未覆盖** `experiments/final_evaluation/` 下既有 CPU/TEST 验证文件（`test_eval.log`、`test_predictions.json` 等均未触碰）。

## 七、产物清单（GPU 验证目录）

`experiments/final_evaluation/gpu_inference_validation/gpu_run_20260818_033342/`
```
detections/TRAIN_000004_apple_scab.json    # 检测结果 JSON（镜像 CPU 格式）
detections/TRAIN_000004_apple_scab.csv
detections/TRAIN_000004_apple_scab.txt
predictions.json                           # 逐目标列表
inference_summary.json                     # 环境/配置/计时/检测/隔离全量记录
cpu_gpu_comparison.json                    # CPU-GPU 一致性对比
gpu_validation.log                         # 人工可读验证日志
originals/TRAIN_000004_apple_scab.jpg      # 原图副本（来自 VAL）
visualized/TRAIN_000004_apple_scab.jpg     # 检测框可视化（复用项目 visualize，未改代码）
```

## 八、结论

- GPU 单图推理在 **RTX 4090 + Paddle 2.6.2 (cu118)** 上验证通过：模型确认运行于 **gpu:0**。
- 检出 1 目标 **Apple Scab Leaf**（conf 0.569851，bbox 166.68,105.84,104.96,87.61），与既有 CPU 结果一致。
- 真实耗时：流水线 509.8 ms、实际 FPS 1.96（单图一次性口径）。
- TEST 全程隔离，未访问 test.json / images/test。
- 验证完成，**立即停止**：未启动 Web、未做批量推理，等待下一步指令。
