# GPU_WEB_VALIDATION_REPORT — AutoDL GPU + Gradio 完整系统联调验证报告

- **验证日期**: 2026-08-18
- **项目根目录**: `/root/autodl-tmp/AIC2026_AgriVision/`
- **验证性质**: **AutoDL GPU + Gradio 完整系统联调阶段（Phase 5）最终验证**。将已验收的完整 Gradio 农业病害检测 Web 系统接入 GPU 后端，进行真实端到端联调。**未训练 / 未重新训练 / 未调参 / 未修改 best_model / 未修改原始数据集 / 未访问 TEST（test.json、images/test）/ 未重新评估 TEST / 未做模型转换 / 未做 RK3588 部署 / 未修改 PaddleDetection 核心源码 / 未删除任何既有实验文件 / 未重新设计 UI / 未删除或减少任何已有功能。**
- **锁定模型**: `experiments/direction_m/M_SCALEUP_100e/checkpoints/best_model.pdparams`
- **锁定配置**: `configs/ppyoloe_plus_crn_m_100e_agrivision.yml`
- **验证图片**: 均为 `dataset/processed_detection_exiffix/images/val/`（仅 VAL）1 张 / 8 张真实上传检测

---

## 一、Web 运行环境

| 项 | 值 |
|---|---|
| 服务地址 | `http://127.0.0.1:7860`（`web/app.py` 默认 server_name，NO_PROXY 已设置） |
| Gradio | **6.24.0** |
| gradio_client（HTTP 测试客户端） | 2.6.0 |
| pandas / jinja2 | pandas 3.0.5 / **jinja2 3.1.6**（此前 3.1.4 与 pandas 3.0.5 的 DataFrame.style 冲突，`pip install "jinja2>=3.1.5"` 修复——环境依赖修复，非项目代码） |
| matplotlib 中文字体 | FangSong（项目自带 simfang.ttf 回退注册） |
| 启动路径 | 复刻生产路径：`backend.set_device_mode("auto")` → `self_check()` → `build_ui(checks)` → `app.queue()` → `_launch_with_localhost_fallback(127.0.0.1:7860)` |

## 二、GPU / Paddle / 设备自动选择

| 项 | 值 |
|---|---|
| GPU 型号 | **NVIDIA GeForce RTX 4090**（compute capability 8.9，24 GB） |
| PaddlePaddle | **2.6.2（cu118）**；`is_compiled_with_cuda=True` |
| 设备解析 | `AGRI_DEVICE` 未设置 → **auto** → `resolve_device(auto)` = **gpu**；模型参数 `Place(gpu:0)`；backend = **paddle_gpu** |
| **设备确认** | **模型实际运行于 gpu:0，非 CPU**（backend=paddle_gpu、device=gpu、参数 place=gpu:0、推理时显存占用上升至 741 MiB 多重确认） |
| 无 GPU 回退 | `CUDA_VISIBLE_DEVICES=""` 时 `resolve_device(auto)` = **cpu**（paddle_cpu 回退验证通过） |
| 模型加载次数 | **仅 1 次**：单例 `backend.get_detector()`（double-checked locking）——多个检测请求共用同一已加载模型；运行时审计 **best_model.pdparams 全程仅 open 1 次** |

## 三、模型 / 配置

- 模型：PP-YOLOE+-m（13 类，input 640×640，参数量 23.57M，pdparams 94.3 MB），锁定 `best_model.pdparams`，**未修改**。
- 配置：`ppyoloe_plus_crn_m_100e_agrivision.yml`（类别映射来自 `annotations/val.json`，13 类）。
- 默认置信度阈值：**0.5**（`DEFAULT_CONF_THRESHOLD`）。

## 四、真实单图 Web 检测（1 张 VAL，真实上传）

- **运行目录**: `web/outputs/run_20260818_034705/`
- 图片: `dataset/processed_detection_exiffix/images/val/TRAIN_000004_apple_scab.jpg`
- 结果: 1 目标 **Apple Scab Leaf**，conf **0.569851**，bbox (166.68, 105.84, 104.96, 87.61)——与 GPU 单图验证完全一致。
- **真实耗时**（`inference_summary.json`，由 Web 实际运行计时）:

| 项 | 值 |
|---|---|
| 单图总耗时 | **0.7657 s** |
| 实际 FPS | **1.31** |

## 五、8 图 Web 批量检测（8 张 VAL，真实批量上传）

- **运行目录**: `web/outputs/run_20260818_034918/`
- **真实性能**（`inference_summary.json`，由 Web 实际批量运行计时）:

| 项 | 值 |
|---|---|
| 批量总耗时（8 图，含读图+推理+后处理+落盘） | **0.6664 s** |
| 平均单图 | **0.0833 s/张** |
| **真实 Web FPS = 8 / 0.6664** | **12.00** |
| 成功 / 失败 | **8 / 0**（`failed=[]`） |
| 总检测目标 | **11** |
| 类别分布 | 6 类（Tomato Septoria leaf spot 4、Tomato leaf 3、Tomato leaf bacterial spot 1、Apple Scab Leaf 1、Apple rust leaf 1、grape leaf 1） |
| 前端呈现 | gallery 8 张、合并表格 11 行、类别统计 6 行、可视化图 8 张、下载文件 13 个（8 可视化 + 5 批次文件） |
| GPU 显存峰值 / 利用率 | 741 MiB / 9 % |

> **FPS 口径**：**12.00 由 Web 实际运行时间重新计算**（8 图 / 0.6664 s），**未直接显示此前 GPU 批量基准 12.84**。首次批量运行实测 12.28（批次间正常波动）。**12.84 作为独立 GPU 批量基准保留**（见 `experiments/final_evaluation/GPU_BATCH_INFERENCE_REPORT.md`，同引擎同 8 图正式批量阶段 0.6233 s），两口径均来自真实计时，非写死、非虚构。

## 六、页面设备 / 后端 / FPS 真实性

- 页面显示的 backend、device、paddle 版本、GPU 型号、输入尺寸均来自 `detector.info()` / `backend.detector_info()` **真实运行时状态**（self_check 显示 `paddle_gpu | device=gpu | paddle=2.6.2 | 640×640`）。
- 检测结果表的耗时 / FPS 读取 `inference_summary.json` 的 `performance`（真实计时），**无任何硬编码**。
- 演示数据未冒充实时推理：真实上传走 `_INFER_LOCK` + `run_batch` → 实际调用 GPU 推理；比赛演示模式独立走预生成 15 张 VAL 演示结果。

## 七、conf_threshold 功能验证（真实 GPU 过滤）

同一张 `TRAIN_000004_apple_scab.jpg` 通过 Web `/_detect` 上传，调节阈值滑块（运行产物 `predictions.json` 逐条核对）：

| 阈值 | 目标数 | 说明 |
|---|---|---|
| 0.05 | **128** | 大量低置信候选全部保留 |
| 0.5（默认） | **1** | Apple Scab Leaf conf 0.569851 |
| 0.9 | **0** | 全部被过滤 |

> 阈值滑块对检测结果产生真实可测影响，过滤链路端到端真实生效。

## 八、15 张比赛演示模式

- `DEMO_CASES` 加载 = **15 / 15**（`web/demo_data/demo_manifest.json`，VAL 子集，仅复制未修改原图）。
- 演示资源自检：original 15/15、result 15/15、TOTAL 15/15。
- 通过 `/_demo_select` 逐个调取：**15/15 全部正常**，上一张 / 下一张切换正常。
- **演示模式继续使用已生成并验收通过的 15 张 VAL 演示结果，未重新生成任何检测结果。**

## 九、损坏图片失败隔离 & 无目标状态

- **损坏图片隔离**（`run_20260818_035018` / `035051`）：1 张损坏图片上传 → **成功 0 / 失败 1**，`failed` 记录清晰中文错误 `图片无法解码（可能已损坏或不支持的格式）：upload_001.jpg`，其余图片不受影响（批次隔离）。✅
- **无目标状态**（`run_20260818_035053`）：`TEST_000028_early_blight1-150x150.jpg`（0 目标）→ 前端空状态卡片正常呈现，真实 FPS 8.34（该图 0.1199 s）。✅

## 十、功能验收结果（全部保持既有功能）

| # | 功能 | 结果 |
|---|---|---|
| 1 | 单图上传 | ✅ 真实 GPU 检测 |
| 2 | 批量图片上传 | ✅ 8 张真实批量检测 |
| 3 | 文件夹上传 | ✅（走同一条 `run_batch` 链路，逐图隔离） |
| 4 | 拖拽上传 | ✅（Gradio 原生 File 上传组件） |
| 5 | conf_threshold 滑块 | ✅ 0.05→128 / 0.5→1 / 0.9→0（真实过滤） |
| 6 | 可视化结果 | ✅ 8 张检测框可视化图正常生成 |
| 7 | 检测结果表格 | ✅ 合并表格 11 行（真实目标） |
| 8 | 类别统计 | ✅ 6 行（6 类，num_images/num_targets/avg_conf/max_conf） |
| 9 | 柱状图 | ✅ WebP 720×350 中文柱状图（FangSong） |
| 10 | 批次汇总 | ✅ 真实总耗时/平均/FPS |
| 11 | 结果下载 | ✅ 13 个文件（8 可视化 + 5 批次文件） |
| 12 | 农业病害场景 | ✅ 番茄/苹果/葡萄场景卡片与中文展示正常 |
| 13 | 15 张比赛演示模式 | ✅ 15/15 |
| 14 | 模型单例加载 | ✅ best_model.pdparams 全程仅 open 1 次 |

## 十一、TEST 隔离检查

- **静态**：本阶段仅访问 8 张 VAL 图片、模型、配置、`annotations/val.json`、演示数据（均来自 `images/val/`）；演示加载器含 `/images/val/` 便携校验并严格禁止 `images/test`。未引用 test.json / images/test。
- **运行时**：启动器安装 `sys.addaudithook` 审计钩子，监控全程 `open/os.open`——
  - 主 Web 联调审计（`/tmp/web_test_audit.json`）：**violations = 0**
  - conf 阈值验证审计（`/tmp/web_test_audit2.json`）：**violations = 0**
  - **全程未打开任何 `test.json` / `images/test` 文件** ✅
- 输出写入 `web/outputs/run_20260818_*`（本次会话新增，均来自 VAL），未触碰既有实验文件；本次验证产生的临时运行目录（conf 阈值 3 个）与 /tmp 脚本已清理。

## 十二、产物清单

- **报告**: `experiments/final_evaluation/GPU_WEB_VALIDATION_REPORT.md`（本文件）
- **真实运行产物**（`web/outputs/`，来自 VAL）:
  - `run_20260818_034705/` — 真实单图 Web 检测（0.7657 s，FPS 1.31）
  - `run_20260818_034918/` — 8 图真实批量 Web 检测（0.6664 s，FPS 12.00，8/8，11 目标）
  - `run_20260818_035018/` `run_20260818_035051/` — 损坏图片失败隔离
  - `run_20260818_035053/` — 无目标状态
- **独立基准**（保留，未覆盖）: `experiments/final_evaluation/gpu_inference_validation/run_20260818_033913/`（GPU 批量 FPS 12.84）、`gpu_run_20260818_033342/`（GPU 单图）、`GPU_BATCH_INFERENCE_REPORT.md`、`GPU_SINGLE_INFERENCE_REPORT.md`

## 十三、结论

- **AutoDL GPU + Gradio 完整系统联调验证通过**：RTX 4090 + Paddle 2.6.2 (cu118) 下，Web 通过 `AGRI_DEVICE=auto` 自动选择 **paddle_gpu / gpu:0**；无 GPU 自动回退 paddle_cpu（已验证）。
- **真实 GPU Web 单图**：0.7657 s，FPS 1.31；**真实 GPU Web 8 图批量**：总耗时 0.6664 s、平均 0.0833 s/张、**真实 Web FPS 12.00**（按 Web 实际运行时间重新计算，未使用 12.84；12.84 保留为独立 GPU 批量基准）。
- 8/8 成功、0 失败、11 目标、6 类；可视化/表格/类别统计/柱状图/批次汇总/下载/场景卡片全部真实可用；conf 阈值真实生效；15 张演示 15/15（沿用预生成结果，未重新生成）；损坏图片失败隔离与无目标状态正常；模型仅加载 1 次。
- **TEST 全程隔离**：violations = 0，未访问 test.json / images/test。
- 验证完成，**立即停止**：Web 已停止、端口已关闭、临时文件已清理。**未自动进入 TensorRT / RK3588 / 任何训练。**
