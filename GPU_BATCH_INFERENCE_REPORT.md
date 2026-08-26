# GPU_BATCH_INFERENCE_REPORT — AutoDL GPU 批量性能验证报告

- **验证日期**: 2026-08-18
- **项目根目录**: `/root/autodl-tmp/AIC2026_AgriVision/`
- **验证性质**: 最终 GPU 批量性能验证（与此前 CPU 批量阶段同一组 8 张 VAL 图片、同一 `batch_infer.run_batch` 引擎与口径）。**未修改任何项目代码 / 未修改 best_model / 未修改原始数据集 / 未训练·重训·调参 / 未访问 TEST（test.json、images/test）/ 未做模型转换 / 未做 RK3588 部署 / 未启动 Web。**
- **锁定模型**: `experiments/direction_m/M_SCALEUP_100e/checkpoints/best_model.pdparams`
- **锁定配置**: `configs/ppyoloe_plus_crn_m_100e_agrivision.yml`
- **验证图片**: 与此前 CPU 批量（`inference/outputs/run_20260817_154953`）完全相同的 8 张 **VAL** 图片

---

## 一、GPU / 环境

| 项 | 值 |
|---|---|
| GPU 型号 | **NVIDIA GeForce RTX 4090**（compute capability 8.9，24 GB） |
| PaddlePaddle | **2.6.2（cu118）** |
| `is_compiled_with_cuda` | True |
| 设备解析 | `resolve_device("auto")` → **gpu**；`paddle.device.get_device()` = **gpu:0**；模型参数 `place = Place(gpu:0)`；backend = **paddle_gpu** |
| **设备确认** | **模型实际运行于 gpu:0，非 CPU**（backend_name=paddle_gpu、device=gpu、param place=gpu:0 三重确认） |

## 二、模型 / 配置 / 8 张图片

- 模型：PP-YOLOE+-m（13 类，input 640×640，参数量 23.57M，pdparams 94.3 MB），**只加载一次**。
- 配置：`ppyoloe_plus_crn_m_100e_agrivision.yml`（类别映射来自 `annotations/val.json`，13 类）。
- 置信度阈值：**0.5**（`DEFAULT_CONF_THRESHOLD`）。
- 8 张图片均位于 `dataset/processed_detection_exiffix/images/val/`（**仅 VAL**；其中 4 张文件名以 `TEST_` 开头为原始数据集命名习惯，其路径在 VAL 目录内，非 TEST 目录）：

| # | 文件名（images/val/） |
|---|---|
| 1 | `TEST_000018_0605_Rust-induced_leafspot.jpg` |
| 2 | `TEST_000028_early_blight1-150x150.jpg` |
| 3 | `TEST_000079_9511.img.jpg` |
| 4 | `TEST_000106_depositphotos_3443387-stock-photo-the-green-grape-leaf-on.jpg` |
| 5 | `TRAIN_000004_apple_scab.jpg` |
| 6 | `TRAIN_000029_early-blight-of-tomato-tomato-1.jpg` |
| 7 | `TRAIN_000037_Tomato+Problems+Septoria+Leaf+Spot.jpg` |
| 8 | `TRAIN_000040_Septoria_leaf_spot_tomato.jpg` |

## 三、验证流程

1. 模型加载 **1 次**（计时 `load_seconds`）。
2. **warmup 2 次**（单图 `predict`，用于 GPU 内核预热，**不计入正式 FPS**，计时 `warmup_seconds`）。
3. **正式批量推理 8 张**：复用与 CPU 阶段相同的 `batch_infer.run_batch`（模型不重复加载、逐图 `predict`、单图失败隔离、落盘产物）——同一引擎、同一口径，保证 FPS 可比。

## 四、真实耗时 / 真实 FPS（正式批量阶段实际计时）

| 阶段 | 耗时 |
|---|---|
| 模型加载（一次性） | **2.358 s** |
| warmup（2 次，不计入 FPS） | **0.612 s** |
| **正式批量总耗时（8 图，含读图+推理+后处理+落盘）** | **0.6233 s** |
| 平均单图（正式阶段） | **0.0779 s/图** |
| 纯推理之和（每图 `inference_ms` 累加） | 0.395 s（其余为读图/落盘/可视化 I/O） |
| **真实 FPS = 8 / 0.6233** | **12.84** |

> **FPS 口径**：12.84 = 8 张 / 正式批量阶段实际耗时 0.6233 s，由真实计时得出；**非单图结果推算、非虚构**。warmup 未计入。
>
> **与单图口径区分**：GPU 单图验证（`gpu_run_20260818_033342`）为冷启动一次性推理（含首调内核初始化），流水线 FPS≈1.96；本次为**预热后的稳态连续批量吞吐**，FPS=12.84，与迁移清单预估的稳态 GPU FPS≈16 同口径、量级一致（I/O+可视化占其余开销）。
>
> **与 CPU 批量对比**：CPU 同 8 图同引擎 `run_20260817_154953` 总耗时 18.84 s、FPS 0.42 → GPU 总耗时 0.6233 s、FPS 12.84，**GPU 批量吞吐约为 CPU 的 30 倍**。

## 五、每图检测结果（阈值 0.5，原图坐标 x,y,w,h）

| 图片 | 目标数 | 检测明细（class_id / class_name / conf / bbox） |
|---|---|---|
| TEST_000018_0605_Rust-induced_leafspot | 1 | 11 Apple rust leaf · 0.836040 · (0.40,7.89,304.56,175.95) |
| TEST_000028_early_blight1-150x150 | 0 | —（无 ≥0.5 目标，与 CPU 基线一致） |
| TEST_000079_9511.img | 3 | 3 Tomato leaf · 0.571680 · (276.97,176.73,136.67,130.35)；3 Tomato leaf · 0.542335 · (276.00,175.86,93.33,131.56)；3 Tomato leaf · 0.508498 · (46.24,266.97,72.76,125.47) |
| TEST_000106_depositphotos_3443387-stock-photo-the-green-grape-leaf-on | 1 | 12 grape leaf · 0.926409 · (186.02,55.40,574.21,553.14) |
| TRAIN_000004_apple_scab | 1 | 9 Apple Scab Leaf · 0.569851 · (166.68,105.84,104.96,87.61) |
| TRAIN_000029_early-blight-of-tomato-tomato-1 | 2 | 2 Tomato Septoria leaf spot · 0.609751 · (50.33,9.36,248.51,272.39)；4 Tomato leaf bacterial spot · 0.523734 · (50.33,9.36,248.51,272.39) |
| TRAIN_000037_Tomato+Problems+Septoria+Leaf+Spot | 2 | 2 Tomato Septoria leaf spot · 0.670609 · (71.03,184.91,209.86,188.58)；2 Tomato Septoria leaf spot · 0.642136 · (19.90,429.08,186.80,185.91) |
| TRAIN_000040_Septoria_leaf_spot_tomato | 1 | 2 Tomato Septoria leaf spot · 0.533621 · (17.42,16.94,275.34,284.41) |

## 六、成功/失败与总检测目标

- **成功 8 / 8，失败 0**（`failed=[]`）。
- **总检测目标 = 11**，类别统计：

| class_id | class_name | 图片数 | 目标数 | avg_conf | max_conf |
|---|---|---|---|---|---|
| 2 | Tomato Septoria leaf spot | 3 | 4 | 0.614029 | 0.670609 |
| 3 | Tomato leaf | 1 | 3 | 0.540838 | 0.571680 |
| 4 | Tomato leaf bacterial spot | 1 | 1 | 0.523734 | 0.523734 |
| 9 | Apple Scab Leaf | 1 | 1 | 0.569851 | 0.569851 |
| 11 | Apple rust leaf | 1 | 1 | 0.836040 | 0.836040 |
| 12 | grape leaf | 1 | 1 | 0.926409 | 0.926409 |

## 七、GPU / CPU 批量结果一致性

- 基准：`inference/outputs/run_20260817_154953`（Windows CPU、Paddle 3.3.0、同 8 图、同阈值 0.5，11 目标）。
- **目标数**：CPU 11 = GPU 11 ✅
- **逐图逐类目标数**：8/8 张一致（含 0 目标的 TEST_000028、3 目标的 TEST_000079、2 目标的 TEST_000037 等多目标场景）✅
- **逐目标贪心 bbox 最近邻匹配**：11/11 全部配对，未匹配 0；`max_conf_delta=0.000175`、`max_bbox_delta=0.0400 px`（含多目标同类的 TEST_000079：GPU 与 CPU 检出的 3 个对象 bbox 完全一致，conf 仅差 ~1e-4）✅
- **结论**：类别映射、bbox 格式（x,y,w,h 原图坐标）、目标数与置信度量级均与既有 CPU 批量结果一致，无类别错位/坐标系错乱/数量异常。未要求逐位一致（符合约束）。

## 八、GPU 显存

| 时刻 | 显存占用（MiB） |
|---|---|
| 运行前基线 | 3 |
| **峰值（warmup + 正式批量全程，nvidia-smi 采样 13 次取最大值）** | **741** |
| 采样期间 GPU 利用率峰值 | 16 % |

> 8 图批量推理的显存峰值与单图验证相同（741 MiB），远低于 24 GB 显存，GPU 无显存压力。

## 九、TEST 隔离检查

- **静态**：本次仅读取 8 张 VAL 图片、模型、配置、`annotations/val.json`；未引用 test.json / images/test。8 张图片均位于 `images/val/`。
- **运行时**：脚本安装 `sys.addaudithook` 审计钩子，全程（含模型加载、warmup、正式批量）监控 `open/os.open` —— **violations = []，未打开任何 `test.json` / `images/test` 文件** ✅
- 输出写入 `experiments/final_evaluation/gpu_inference_validation/run_20260818_033913/`（**新独立目录**），未覆盖单图结果 `gpu_run_20260818_033342/`，也未触碰 `experiments/final_evaluation/` 下既有 CPU/TEST 验证文件。

## 十、产物清单

`experiments/final_evaluation/gpu_inference_validation/run_20260818_033913/`
```
inference_summary.json             # 批次环境/配置/性能/类别统计（与 CPU 阶段同构）
gpu_batch_validation_meta.json     # 本验证补充元数据（load/warmup/显存峰值/violations/每图明细）
predictions.json / predictions.csv # 11 个目标逐条记录
class_statistics.csv               # 类别统计
inference.log                      # 批次日志
detections/<8 张>.json/.csv/.txt   # 每图检测结果（含每图 inference_ms）
originals/<8 张>.jpg               # 原图副本（来自 VAL）
visualized/<8 张>.jpg              # 检测框可视化（复用项目 visualize，未改代码）
```

## 十一、结论

- GPU 批量推理在 **RTX 4090 + Paddle 2.6.2 (cu118)** 上验证通过：模型确认运行于 **gpu:0**，**只加载一次**。
- **真实 FPS = 12.84**（8 图 / 正式批量实际耗时 0.6233 s，含读图/推理/后处理/落盘；warmup 未计入），约为同口径 CPU 批量（FPS 0.42）的 **30 倍**。
- 8/8 成功，0 失败，总检测目标 11，与 CPU 基线逐目标一致（max_conf_delta 0.000175、max_bbox_delta 0.04 px）。
- 显存峰值 741 MiB，远低于 24 GB。
- **TEST 全程隔离**，未访问 test.json / images/test。
- 验证完成，**立即停止**：未启动 Web，等待下一步指令。
