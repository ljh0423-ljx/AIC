# DOC_ASSETS_INDEX — 比赛文档素材索引

> 生成时间：2026-08-18
> 素材目录：`docs/assets/`（本文件位于 `docs/`，索引其子目录 assets/）
> 生成脚本：`docs/make_assets.py`（只读，不修改任何原始数据/模型/代码；可重复运行）
> 所有图表/CSV **数据均来自现有文件**，未虚构、未平滑补数据；图片均为真实模型输出或真实 GT 标注渲染。

---

## 一、素材清单

### A. 训练过程曲线（来源：`experiments/direction_m/M_SCALEUP_100e/logs/formal_train.out`）

| 文件名 | 用途 | 数据来源 | 证据报告 | 可用于PPT |
|---|---|---|---|---|
| `training_loss_curve.png` | 训练 loss 随 epoch 变化曲线（高清 150dpi） | formal_train.out 逐迭代 loss，按 epoch 取均值 | M_SCALEUP_FINAL_REPORT.md | ✅ |
| `training_loss_raw.csv` | 原始 300 个迭代采样点（epoch/iter/lr/loss/loss_cls/loss_iou/loss_dfl/loss_l1） | formal_train.out（log_iter=20 → 每 epoch 3 个采样点） | 同上 | ✅（数据底表） |
| `training_loss_epoch.csv` | 每 epoch 平均 loss（100 点） | 由 raw CSV 聚合 | 同上 | ✅（数据底表） |
| `training_map_curve.png` | VAL mAP@0.5:0.95 随 epoch 变化曲线（每 5 轮评估一次，20 点） | formal_train.out 每 5 epoch 的 COCO 评估 | M_SCALEUP_FINAL_REPORT.md | ✅ |
| `training_map_epoch.csv` | mAP 逐评估点（checkpoint epoch / 已训练轮数 / mAP） | 同上 | 同上 | ✅（数据底表） |

> mAP 曲线关键点：epoch50 达到 best **0.428**，epoch100 final **0.408**（与 M_SCALEUP_FINAL_REPORT 一致）。

### B. 13 类数据分布（来源：`dataset/processed_detection_exiffix/label_list.txt`）

| 文件名 | 用途 | 数据来源 | 证据报告 | 可用于PPT |
|---|---|---|---|---|
| `class_distribution.png` | 13 类「Train 图片数」与「Train 目标数」双面板横向柱状图 | label_list.txt（每类图片数/目标数，Train 子集） | PROJECT_FINAL_STATUS.md §4 | ✅ |
| `class_distribution.csv` | 13 类 class_id/名称/作物/图片数/目标数 | 同上 | 同上 | ✅（数据底表） |

### C. 13 类 AP（来源：`experiments/final_evaluation/M_FINAL_TEST_REPORT.md` §3.3/§5.3）

| 文件名 | 用途 | 数据来源 | 证据报告 | 可用于PPT |
|---|---|---|---|---|
| `class_ap_val_test.png` | 13 类 VAL vs 独立 TEST 的 AP 对比柱状图 | M_FINAL_TEST_REPORT.md §3.3（TEST）、§5.3（VAL） | M_FINAL_TEST_REPORT.md | ✅ |
| `class_ap_val_test.csv` | 13 类 AP_VAL / AP_TEST 逐项数据 | 同上 | 同上 | ✅（数据底表） |

### D. 系统技术架构图（来源：`web/` 与 `inference/` 实际代码结构）

| 文件名 | 用途 | 数据来源 | 证据报告 | 可用于PPT |
|---|---|---|---|---|
| `system_architecture.png` | 系统数据流架构图（上传→Web→DetectorBackend→Paddle GPU/CPU→预处理→PP-YOLOE+-m→后处理→可视化/统计/下载；RKNN/RK3588 以虚线标注为预留接口、未部署） | web/app.py、web/backend.py、inference/detector.py、inference/postprocess.py、inference/visualize.py、inference/batch_infer.py 实际结构 | WEB_DEMO_FINAL_REPORT.md §6、INFERENCE_VALIDATION_REPORT.md §10 | ✅ |

### E. 数据集样例 + GT bbox 可视化（来源：`val.json` + `images/val/`，仅 VAL）

| 文件名 | 内容 | 数据来源 | 证据报告 | 可用于PPT |
|---|---|---|---|---|
| `dataset_sample_01_apple_scab_multi.jpg` | 苹果疮痂病，6 个 GT bbox（多目标） | val.json 标注渲染 | DATASET_FINAL_REPORT.md | ✅ |
| `dataset_sample_02_tomato_mosaic_dense.jpg` | 番茄花叶病，12 个 GT bbox（密集） | 同上 | 同上 | ✅ |
| `dataset_sample_03_tomato_yellow_virus_multi.jpg` | 番茄黄化病毒，5 个 GT bbox（多目标） | 同上 | 同上 | ✅ |
| `dataset_sample_04_apple_leaf_single.jpg` | 苹果健康叶，1 个 GT bbox（单目标） | 同上 | 同上 | ✅ |
| `dataset_sample_05_grape_leaf_single.jpg` | 葡萄健康叶，1 个 GT bbox（单目标） | 同上 | 同上 | ✅ |
| `dataset_sample_06_grape_black_rot_single.jpg` | 葡萄黑腐病，1 个 GT bbox（单目标） | 同上 | 同上 | ✅ |
| `dataset_samples_grid.jpg` | 上述 6 张合并网格图 | 同上 | 同上 | ✅ |

> 覆盖番茄(2)/苹果(2)/葡萄(2)，单目标(3)/多目标(2)/密集(1)；均为 VAL 子集 GT 标注渲染，**未访问 TEST**。

### F. 演示检测结果样例（来源：`web/demo_outputs/run_20260817_173307/visualized/`，模型真实输出）

| 文件名 | 内容 | 数据来源 | 证据报告 | 可用于PPT |
|---|---|---|---|---|
| `demo_result_01_apple_scab_multi.jpg` | 苹果疮痂病检测结果 | demo_outputs visualized（真实推理渲染） | DEMO_CASES_REPORT.md | ✅ |
| `demo_result_02_tomato_mosaic_dense.jpg` | 番茄花叶病密集场景检测结果 | 同上 | 同上 | ✅ |
| `demo_result_03_tomato_yellow_virus_multi.jpg` | 番茄黄化病毒多目标检测结果 | 同上 | 同上 | ✅ |
| `demo_result_04_apple_leaf_single.jpg` | 苹果健康叶单目标检测结果 | 同上 | 同上 | ✅ |
| `demo_result_05_grape_leaf_single.jpg` | 葡萄健康叶单目标检测结果 | 同上 | 同上 | ✅ |
| `demo_result_06_grape_black_rot_single.jpg` | 葡萄黑腐病单目标检测结果 | 同上 | 同上 | ✅ |
| `demo_results_grid.jpg` | 上述 6 张合并网格图 | 同上 | 同上 | ✅ |

> 演示图均为 `inference/infer.py` 正式推理程序对 PP-YOLOE+-m best_model 的真实输出（含置信度与检测框），由 `demo_outputs/visualized/` 复制，**未修改**。

---

## 二、无法生成 / 待补充项（禁止猜测、禁止虚构）

| 项目 | 状态 | 说明 |
|---|---|---|
| **Web 界面截图**（主界面 / 检测结果 / 类别统计 / 农业病害场景 / 比赛演示模式） | ⛔ 无法生成 | 仓库中**不存在**任何现有 UI 截图（已全文搜索 *.png，仅实验诊断图）。截图必须来自真实运行系统，但：① 本机为 Windows 无 GPU、Paddle 3.3.0 CPU 环境，且沙箱会拦截 127.0.0.1 预启动探测；② 需启动 `web/app.py` 并用浏览器自动化（playwright/selenium）逐状态截图，属「启动 Web + 执行推理」，超出本阶段「不启动 Web / 不执行推理 / 不伪造界面」约束。**不能**用绘图伪造界面。**建议**：在正式展示环境（AutoDL GPU 或本地正常终端）启动 Web 后手动截图，或授权我启动 Web + 浏览器自动化后生成。 |

---

## 三、使用说明

1. 所有 PNG 为 150 dpi 高清图，可直接插入 PPT / 技术文档；CSV 为可复核的数据底表。
2. 引用 FPS/指标时务必注明口径与来源报告（见 `PROJECT_EXPERIMENT_INDEX.md` §3.5、`PROJECT_FINAL_FACTS.md`）。
3. 数据集样例（E 组）为 GT 标注渲染；演示结果（F 组）为模型预测渲染——两者语义不同，PPT 中请分别标注。
4. 架构图中 RKNN/RK3588 为虚线「预留接口，未部署」，PPT 中不得表述为已部署。
5. 重新生成：在工程根目录运行 `python docs/make_assets.py`（需 matplotlib/pandas/numpy/opencv）。

---

*本索引由只读盘点生成，未修改任何原始报告、模型、数据集、代码与 Web 系统。*
