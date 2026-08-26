# PROJECT_EXPERIMENT_INDEX — 实验资料总索引与事实库

> 生成时间：2026-08-18
> 性质：**只读盘点**。本文件仅对已存在的实验报告/验证报告/日志做读取、摘录与交叉核对；**未修改任何原始报告，未修改模型/数据集/代码/Web，未训练、未推理、未访问 TEST、未做模型转换与 RK3588 部署**。
> 本文件只记录能被原始报告直接支持的事实；无证据的写「暂无实测数据」，冲突的单独列为「待确认数据」，不自行裁决正误。

---

## 一、盘点范围与结论概要

- 主项目根目录：`D:\Fruit\AIC2026_AgriVision\`
- 数据审计类报告另存于：`D:\Fruit\`（DATASET_AUDIT_REPORT.md、DATASET_TRANSFER_AUDIT*.md、PLANTDOC_DETECTION_AUDIT.md、README_AUTODL.md）
- 搜到 **实验/验证类 Markdown 报告约 52 份**（另有 archive/ 同名归档副本若干、README 若干），以及大量支撑性 JSON/CSV/日志：
  - 训练/评估日志（train.log、val_eval*.log、test_eval.log 等）
  - 指标 JSON（bbox.json、test_analyze.json、FREEZE_CHECK.json、DRIFT_GUARD.json 等）
  - 推理产物（web/outputs/、inference/outputs/ 下每 run 的 inference_summary.json、predictions.json/csv、class_statistics.csv、detections/*）
- 项目状态（据 `PROJECT_FINAL_STATUS.md`）：全部训练与消融实验已结束，模型已锁定，进入归档/交付阶段。

---

## 二、报告总清单（52 份，按类别分组）

> 路径除特别注明外，均相对主项目根 `D:\Fruit\AIC2026_AgriVision\`。**archive/ 目录存在与当前同名报告的归档副本，本表只列当前版，归档副本不重复列**。

### A. 最终结论 / 状态 / 归档类

| 报告 | 类型 | 日期 | 核心内容 |
|---|---|---|---|
| `PROJECT_FINAL_STATUS.md` | 项目最终状态审计 | 2026-08-17 | 最终模型 PP-YOLOE+-m 锁定；VAL 0.428 / TEST 0.417；参数 23.57M / 94.3MB / FPS 16.6；全部方向结论汇总 |
| `PROJECT_ARCHIVE_STATUS.md` | 归档状态审计 | 2026-08-16 | experiments 861 文件 ≈15.0GB：167 个 pdparams、21 份报告、27 CSV、44 日志等 |
| `NOVELTY_DESIGN_REPORT.md` | 创新方案可行性分析 | 2026-08-16 | 半/弱监督方案 S1(CG-ACS)/S2(CTPL)/S3(CRL) 设计；PV_CORE 19233 图 |

### B. GPU 推理 / 系统联调验证类（最终实测）

| 报告 | 类型 | 日期 | 核心结论 |
|---|---|---|---|
| `GPU_SINGLE_INFERENCE_REPORT.md` | GPU 单图推理验证 | 2026-08-18 | RTX4090 + Paddle2.6.2；单图冷启动 509.8ms / FPS 1.96；1 目标 Apple Scab Leaf conf 0.569851 |
| `GPU_BATCH_INFERENCE_REPORT.md` | GPU 批量性能验证 | 2026-08-18 | 8 张 VAL；总耗时 0.6233s / FPS 12.84；8/8 成功、11 目标；与 CPU 逐目标一致 |
| `GPU_WEB_VALIDATION_REPORT.md` | GPU+Gradio 完整系统联调 | 2026-08-18 | Web 单图 0.7657s / FPS 1.31；8 图批量 0.6664s / FPS 12.00；conf 阈值真实生效；15 演示 15/15 |

### C. 环境验收 / 迁移 / 部署类

| 报告 | 类型 | 日期 | 核心内容 |
|---|---|---|---|
| `AUTODL_ENV_CHECK.md` | AutoDL 环境验收 | 2026-08-18 | RTX4090 24GB / Paddle2.6.2(cu118) / Python3.12.3；模型加载 ~3.5s |
| `AUTODL_WEB_FIX_REPORT.md` | 迁移问题修复 | 2026-08-18 | gradio 6.24.0、fonts-noto-cjk、AGRI_DEVICE=auto→gpu |
| `AUTO_DL_MIGRATION_MANIFEST.md` | 迁移清单 | 2026-08-17 | Windows→AutoDL RTX4090 迁移；预估稳态 GPU FPS≈16 |
| `LOCAL_DEPLOYMENT_AUDIT.md` | 本地部署可行性审计 | 2026-08-17 | 本地 Windows 无 GPU；Paddle 3.3.0 仅验证用；CPU 单图纯前向 2.230s |

### D. 最终 TEST 评估类

| 报告 | 类型 | 日期 | 核心结论 |
|---|---|---|---|
| `experiments/final_evaluation/M_FINAL_TEST_REPORT.md` | 最终 TEST 独立评估 | 2026-08-17 | TEST mAP@0.5:0.95=**0.417**；mAP@0.5=0.601；P/R/F1=0.685/0.452/0.544；FPS 16.6；13 类 AP 全表 |

### E. 推理程序验证类

| 报告 | 类型 | 日期 | 核心结论 |
|---|---|---|---|
| `inference/INFERENCE_VALIDATION_REPORT.md` | 本地 CPU 推理验证 | 2026-08-17 | 单图 2304ms；8 图批量 18.84s / FPS 0.42；8/8 成功 11 目标；失败隔离/文件夹/异常处理全过 |
| `inference/README.md` | 推理程序说明 | — | 推理模块用法与结构 |

### F. Web 系统验证类

| 报告 | 类型 | 日期 | 核心结论 |
|---|---|---|---|
| `web/WEB_APP_VALIDATION_REPORT.md` | Web 功能验证 | 2026-08-17 | 单图 2.44s/FPS 0.41；8 图 20.24s/FPS 0.40；功能清单全过 |
| `web/WEB_DEMO_FINAL_REPORT.md` | 比赛答辩版 Web | 2026-08-17 | 演示 15 张 25.46s/FPS 0.59；Backend 抽象与 RKNN 预留 |
| `web/DEMO_CASES_REPORT.md` | 15 例演示案例 | 2026-08-17 | 15 案例清单（番茄8/苹果4/葡萄3；25 目标） |
| `web/README.md` | Web 说明 | — | Web 用法 |

### G. 数据集审计 / 预处理类

| 报告 | 类型 | 日期 | 核心内容 |
|---|---|---|---|
| `dataset/processed_detection/DATASET_FINAL_REPORT.md` | 数据集最终预处理报告 | 2026-08-13 | PlantDoc 13 类/1144 图/3861 目标；train916/val114/test114 |
| `dataset/processed_detection/FINAL_VISUAL_AUDIT.md` | 最终可视化审查 | 2026-08-13 | 1144 图/3861 bbox；bbox 结构 0 缺陷 |
| `D:\Fruit\DATASET_AUDIT_REPORT.md` | Cropped-PlantDoc 分类集审计 | 2026-08-13 | 分类集 28 类 2485 图（非检测集） |
| `D:\Fruit\DATASET_TRANSFER_AUDIT.md` | PlantVillage/IP102 迁移审计 | 2026-08-16 | PlantVillage 54305 图/38 类；13 类精确子集 19233 图 |
| `D:\Fruit\DATASET_TRANSFER_AUDIT_V2.md` | 迁移审计（口径修正版） | 2026-08-16 | 同上，PV-CORE 19233 图/263MB |
| `D:\Fruit\PLANTDOC_DETECTION_AUDIT.md` | PlantDoc 检测集审计 | 2026-08-13 | **官方来源/论文/license 唯一明确记录处**（见下） |
| `D:\Fruit\README_AUTODL.md` | AutoDL 部署说明 | — | 显存/镜像/版本建议 |

### H. 基线实验类（baseline_v1）

| 报告 | 类型 | 日期 | 核心结论 |
|---|---|---|---|
| `experiments/BASELINE_REPORT.md` | 基线报告**模板** | 2026-08-13 | 指标均为「训练后填」，无实测 |
| `experiments/baseline_v1/BASELINE_SUMMARY.md` | Baseline v1 最终汇总 | 2026-08-14 | VAL best **0.408**/final 0.388；TEST best 0.396/final 0.401；参数 7.70M |
| `experiments/baseline_v1/BASELINE_DIAGNOSTIC_REPORT.md` | VAL 深度诊断 | 2026-08-14 | 番茄类内混淆为主；候选 3 方向 |
| `experiments/baseline_v1/DATASET_CONSISTENCY_REPORT.md` | 数据一致性核对 | 2026-08-14 | 916→915 差异=移除 2 张损坏图 |
| `experiments/baseline_v1/ABLATION_PLAN.md` | 消融计划 | 2026-08-14 | EXIF 修复两臂对照设计 |
| `experiments/baseline_v1/EXIF_FIX_PLAN.md` | EXIF 修复方案 | 2026-08-14 | 11 张 EXIF 异常图修复设计 |

### I. 消融对比 / 方向2（CGPM）

| 报告 | 类型 | 日期 | 核心结论 |
|---|---|---|---|
| `experiments/ablation_compare/ABLATION_VAL_SUMMARY.md` | EXIF 消融对比汇总 | 2026-08-14 | Arm A(clean) best 0.386 / Arm B(exiffix,A0) best 0.401，+0.0148 ✅ |
| `experiments/direction2/DIRECTION2_ABLATION_REPORT.md` | CGPM 消融 | 2026-08-15 | A0 0.401 / A1 0.397 / A2 0.405，均未过门槛 ❌ |
| `experiments/direction2/DIRECTION2_DIAGNOSTIC.md` | 方向2 诊断 | 2026-08-15 | 分类混淆为主瓶颈 |
| `experiments/direction2/DIRECTION2_DESIGN.md` | 方向2 设计 | 2026-08-15 | CGPM/CCDH/ATRR 设计 |
| `experiments/direction2/ABLATION_DIRECTION2_PLAN.md` | 方向2 计划 | 2026-08-15 | A0/A1/A2/A3 计划 |

### J. 方向3（Class-Weighted VFL）

| 报告 | 类型 | 日期 | 核心结论 |
|---|---|---|---|
| `experiments/ablation_dir3_B1/DIRECTION3_B1_REPORT.md` | B1 消融 | 2026-08-15 | A0 best 0.401 / B1 best 0.405（+0.004 差 0.001）❌ |
| `experiments/direction3/DIRECTION3_DIAGNOSTIC.md` | 方向3 诊断 | 2026-08-15 | 漏检=置信度压低+密集场景 |
| `experiments/direction3/DIRECTION3_DESIGN.md` | 方向3 设计 | 2026-08-15 | B1 class-weighted VFL 设计 |
| `experiments/direction3/ABLATION_DIRECTION3_PLAN.md` | 方向3 计划 | 2026-08-15 | A0→B1 单变量计划 |

### K. 方向4（Density-Aware 校准）

| 报告 | 类型 | 日期 | 核心结论 |
|---|---|---|---|
| `experiments/direction4/DIRECTION4_DIAGNOSTIC.md` | 方向4 诊断 | 2026-08-15 | 核心瓶颈=置信度校准缺陷 |
| `experiments/direction4/DIRECTION4_DESIGN.md` | 方向4 设计 | 2026-08-15 | D1/D2/D3 设计 |
| `experiments/direction4/ABLATION_DIRECTION4_PLAN.md` | 方向4 计划 | 2026-08-15 | D1/D2 执行计划 |
| `experiments/direction4/d1_formal/D1_REPORT.md` | D1 正式实验 | 2026-08-16 | D1 best 0.4061 < 门槛 0.4062 ❌ |
| `experiments/direction4/d1_smoke/D1_SMOKE_REPORT.md` | D1 smoke | 2026-08-16 | 机制 35/35 全过 |
| `experiments/direction4/d2_results/D2_REPORT.md` | D2 消融 | 2026-08-15 | D2 0.4008 < A0 0.4012 ❌ |

### L. 方向5（CG-ACS / PV 预训练）与方向 M（容量放大）

| 报告 | 类型 | 日期 | 核心结论 |
|---|---|---|---|
| `experiments/direction5/B1_PVPRETRAIN_100e/B1_PVPRETRAIN_FINAL_REPORT.md` | PV 预训练最终报告 | 2026-08-17 | B1 best 0.211 ❌ 负迁移 |
| `experiments/direction5/S1_CG_ACS_FINAL_REPORT.md` | S1 最终报告 | 2026-08-16 | S1 best 0.3668 ❌ FAIL |
| `experiments/direction5/CG_ACS_DESIGN.md` | CG-ACS 设计 | 2026-08-16 | S1 辅助监督设计 |
| `experiments/direction5/CG_ACS_MECHANISM_CHECK.md` | 机制核验 | 2026-08-16 | 三点机制全 ✅ |
| `experiments/direction5/CG_ACS_STAGE0_REPORT.md` | 阶段0 报告 | 2026-08-16 | 门槛 1-4 PASS + smoke |
| `experiments/direction_m/M_SCALEUP_100e/M_SCALEUP_FINAL_REPORT.md` | **M 容量放大最终报告** | 2026-08-17 | **M best VAL 0.428，最终模型候选** ✅ |
| `experiments/direction_m/M_SCALEUP_100e/smoke/SMOKE_GATE_RESULT.md` | M smoke 门禁 | 2026-08-17 | 门控全过进入正式训练 |

---

## 三、已确认事实索引（有原始报告支撑）

### 3.1 最终模型

| 事实 | 值 | 证据 |
|---|---|---|
| 最终模型架构 | PP-YOLOE+-m（depth_mult=0.67, width_mult=0.75；CSPResNet+CustomCSPPAN+PPYOLOEHead，零自定义模块） | PROJECT_FINAL_STATUS.md §1/§3.1 |
| 最终 checkpoint | `experiments/direction_m/M_SCALEUP_100e/checkpoints/best_model.pdparams` | PROJECT_FINAL_STATUS.md §3.1 |
| best_model md5 | `18bd0e99329be2113f57280188c227b6` | FREEZE_CHECK.json / PROJECT_FINAL_STATUS.md |
| 参数量 | 23,568,416（23.57M） | PROJECT_FINAL_STATUS.md §6 / M_SCALEUP_FINAL_REPORT.md |
| 模型大小 | 94.3 MB（94,337,197 B） | PROJECT_FINAL_STATUS.md §3.1 |
| 预训练 | obj365-m（`ppyoloe_crn_m_obj365_pretrained.pdparams`） | configs/ppyoloe_plus_crn_m_100e_agrivision.yml / PROJECT_FINAL_STATUS.md |
| 最终配置 | `configs/ppyoloe_plus_crn_m_100e_agrivision.yml` | PROJECT_FINAL_STATUS.md §3.2 |

### 3.2 数据集与类别

| 事实 | 值 | 证据 |
|---|---|---|
| 数据集 | PlantDoc 目标检测集（3 作物 / 13 类 / 1144 图 / 3861 目标） | PROJECT_FINAL_STATUS.md §1 / DATASET_FINAL_REPORT.md |
| **官方来源（唯一明确记录处）** | GitHub `pratikkayal/PlantDoc-Object-Detection-Dataset`；论文 "PlantDoc: A Dataset for Visual Plant Disease Detection (CoDS-COMAD 2020)"；许可证 CC-BY-4.0 | `D:\Fruit\PLANTDOC_DETECTION_AUDIT.md` |
| 官方下载 URL | **未记录（待补充）** | — |
| 原始划分 | train 916/3084，val 114/354，test 114/423 | PROJECT_ARCHIVE_STATUS.md |
| EXIF-fix（最终锁定） | train 915/3079，val 113/350，test 114/423 | PROJECT_FINAL_STATUS.md §1 |
| 类别数 | 13 | 多处一致 |

**13 类名称（全项目一致，COCO category_id 1–13，模型输出 0–12）**：
0 Tomato Early blight leaf；1 Tomato Septoria leaf spot；2 Tomato leaf；3 Tomato leaf bacterial spot；4 Tomato leaf late blight；5 Tomato leaf mosaic virus；6 Tomato leaf yellow virus；7 Tomato mold leaf；8 Apple Scab Leaf；9 Apple leaf；10 Apple rust leaf；11 grape leaf；12 grape leaf black rot。（证据：`dataset/processed_detection/label_list.txt`、PROJECT_FINAL_STATUS.md §4、FREEZE_CHECK.json）

### 3.3 训练环境与参数

| 事实 | 值 | 证据 |
|---|---|---|
| 训练平台 | AutoDL Linux，NVIDIA RTX 4090 24GB | PROJECT_FINAL_STATUS.md §1 / AUTODL_ENV_CHECK.md |
| PaddlePaddle | 2.6.2（cu118） | AUTODL_ENV_CHECK.md / 多个 GPU 报告 |
| Python | 3.12.3 | AUTODL_ENV_CHECK.md |
| CUDA | 11.8（编译期）；cuDNN 编译 8.6 / 运行 9.1 | AUTODL_ENV_CHECK.md / GPU_SINGLE |
| PaddleDetection | v2.9.0（git HEAD `b25522a0`） | PROJECT_FINAL_STATUS.md / FREEZE_CHECK.json |
| epoch | 100 | config / PROJECT_FINAL_STATUS.md §3.2 |
| batch_size（实际） | 16（配置默认 8，`-o` 覆盖） | PROJECT_FINAL_STATUS.md §3.2 / FREEZE_CHECK.json |
| base_lr（实际） | 0.002（配置默认 0.001，线性缩放） | 同上 |
| lr 调度 | CosineDecay(max_epochs=100) + LinearWarmup(5) | config / PROJECT_FINAL_STATUS.md |
| EMA | 开启（0.9998） | PROJECT_FINAL_STATUS.md |
| 输入尺寸 | 640（多尺度 320~768） | config / 多处 |
| 训练耗时 | ≈2.5 h（100 epoch） | PROJECT_FINAL_STATUS.md §3.2 |
| 训练显存峰值 | ~20.3 GB / 25.9 GB（78%） | PROJECT_FINAL_STATUS.md §3.2 |

### 3.4 验证结果（VAL / TEST）

| 指标 | 最终 M | 证据 |
|---|---|---|
| VAL mAP@0.5:0.95（选模型用） | **0.428** | PROJECT_FINAL_STATUS.md / M_FINAL_TEST_REPORT.md |
| **TEST mAP@0.5:0.95** | **0.417** | M_FINAL_TEST_REPORT.md / PROJECT_FINAL_STATUS.md |
| TEST mAP@0.5 | 0.601 | 同上 |
| TEST mAP@0.75 | 0.475 | 同上 |
| TEST AP small/med/large | 0.408 / 0.294 / 0.444 | 同上 |
| TEST AR@1/10/100 | 0.264 / 0.598 / 0.686 | 同上 |
| TEST Precision/Recall/F1 (conf=0.5) | 0.685 / 0.452 / 0.544 | 同上 |
| TEST 密集场景 Recall（10 图/137GT） | 0.372（51/137） | 同上 |

（13 类 TEST/VAL 每类 AP 全表见 `M_FINAL_TEST_REPORT.md` §3.3，此处不重复。）

### 3.5 FPS（多口径，均为真实计时）

| 口径 | 值 | 设备/方法 | 证据 |
|---|---|---|---|
| TEST eval FPS（最终 M） | 16.6 | RTX4090，tools/eval.py，114 图 | M_FINAL_TEST_REPORT.md |
| VAL eval FPS（最终 M） | 17.8 | RTX4090，113 图 | PROJECT_FINAL_STATUS.md §6 |
| VAL eval FPS（A0/s） | 19.8 | RTX4090 | PROJECT_FINAL_STATUS.md §6 |
| GPU 批量 FPS | 12.84 | 8 张 VAL，warmup 排除，run_batch | GPU_BATCH_INFERENCE_REPORT.md |
| GPU Web 批量 FPS | 12.00 | 8 张 VAL，Web 实际计时 | GPU_WEB_VALIDATION_REPORT.md |
| GPU 单图 FPS（冷启动） | 1.96 | 单图一次性 | GPU_SINGLE_INFERENCE_REPORT.md |
| GPU Web 单图 FPS | 1.31 | Web 实际计时 | GPU_WEB_VALIDATION_REPORT.md |
| CPU 批量 FPS | 0.42 | 8 张 VAL，Windows Paddle3.3.0 | INFERENCE_VALIDATION_REPORT.md |
| CPU 演示 15 张 FPS | 0.59 | 15 张 VAL | WEB_DEMO_FINAL_REPORT.md / DEMO_CASES_REPORT.md |
| CPU 单图 FPS | 0.41~0.42 | Windows CPU | WEB_APP_VALIDATION / INFERENCE_VALIDATION |

> 说明：上述 FPS 为**不同口径**（冷启动 vs 预热稳态、单图 vs 批量、eval 循环 vs 推理流水线、CPU vs GPU），非同一实验的冲突值，但引用时必须注明口径。

### 3.6 各方向 VAL mAP@0.5:0.95 结论汇总（最终判定）

| 实验 | VAL（best） | 判定 | 证据 |
|---|---|---|---|
| Baseline v1（PP-YOLOE+-s, clean 数据） | 0.408（final 0.388；TEST best 0.396/final 0.401） | 基准 | BASELINE_SUMMARY.md |
| Arm A（clean, seed0） | 0.386 | 对照 | ABLATION_VAL_SUMMARY.md |
| **Arm B / A0（exiffix）** | 0.401（3 位）/ 0.4012（4 位） | ✅ EXIF 修复胜者 | ABLATION_VAL_SUMMARY.md / direction4 报告 |
| A1（CGPM guided） | 0.397 | ❌ | DIRECTION2_ABLATION_REPORT.md |
| A2（CGPM fixed） | 0.405 | ❌ 未过门槛 | 同上 |
| B1（class-weighted VFL） | 0.405 | ❌ +0.004 差 0.001 | DIRECTION3_B1_REPORT.md |
| D1（density-aware VFL） | 0.4061 | ❌ 差门槛 0.0001 | D1_REPORT.md |
| D2（推理侧重打分） | 0.4008 | ❌ <A0 | D2_REPORT.md |
| S1（CG-ACS 辅助监督） | 0.3668 | ❌ | S1_CG_ACS_FINAL_REPORT.md |
| B1-PVPRETRAIN（PV 预训练） | 0.211 | ❌ 负迁移 | B1_PVPRETRAIN_FINAL_REPORT.md |
| **M_SCALEUP（s→m）** | **0.428** | ✅ 最终模型候选 | M_SCALEUP_FINAL_REPORT.md |

---

## 四、待确认数据（同一实验出现多个数字，未裁决正误）

以下为交叉核对发现的冲突/不一致项，每个数字均标明来源；**本文件不判断哪个正确**。

### 冲突 1：A0（PP-YOLOE+-s，EXIF-fix）VAL best 出现两组数值

| 数值 | 出现位置 |
|---|---|
| **0.401**（3 位）/ **0.4012**（4 位），final 0.386 | `ablation_compare/ABLATION_VAL_SUMMARY.md`、`direction2/DIRECTION2_ABLATION_REPORT.md`、`ablation_dir3_B1/DIRECTION3_B1_REPORT.md`、`direction4/*` 系列 |
| **0.410**（@epoch49~65），final 0.389 | `direction5/B1_PVPRETRAIN_100e/B1_PVPRETRAIN_FINAL_REPORT.md`、`direction5/S1_CG_ACS_FINAL_REPORT.md`、`direction_m/M_SCALEUP_100e/M_SCALEUP_FINAL_REPORT.md` |

> 注：0.401/0.4012 为消融/方向2/3/4 口径；0.410 为方向5/M 报告中对「A0 参照」的复算值。两者来源不同、成因报告未解释，属待确认。

### 冲突 2：Baseline 的 VAL mAP 数值

| 数值 | 出现位置 |
|---|---|
| Baseline VAL **0.401**（TEST 0.401） | `PROJECT_FINAL_STATUS.md` §2.1（「基线 Baseline v1…0.401（TEST 0.401）」） |
| Baseline VAL best **0.408** / final **0.388**（TEST best 0.396 / final 0.401） | `baseline_v1/BASELINE_SUMMARY.md` |

> 注：PROJECT_FINAL_STATUS 的「0.401」实为 exiffix A0 与 model_final TEST 的数值，与 BASELINE_SUMMARY 的 best 0.408 口径不同，属待确认。

### 冲突 3：Baseline（PP-YOLOE+-s, clean）VAL 与消融 Arm A（clean）VAL 不一致

| 数值 | 出现位置 |
|---|---|
| clean 数据 VAL best **0.408** | `baseline_v1/BASELINE_SUMMARY.md` |
| clean 数据（Arm A, seed=0）VAL best **0.386** | `ablation_compare/ABLATION_VAL_SUMMARY.md` |

### 冲突 4：A0（s）模型大小两个数值

| 数值 | 出现位置 |
|---|---|
| **29.42 MB** | `baseline_v1/BASELINE_SUMMARY.md` |
| **30.8 MB** | `PROJECT_FINAL_STATUS.md` §6 |

### 冲突 5：A0（s）VAL eval FPS 两个数值

| 数值 | 出现位置 |
|---|---|
| eval.py 实测 **17.3** | `baseline_v1/BASELINE_SUMMARY.md` |
| VAL eval FPS **19.8** | `PROJECT_FINAL_STATUS.md` §6 |

### 冲突 6：数据集每类目标数（object_count）在不同报告间不一致

| 类别 | DATASET_FINAL_REPORT.md / label_list.txt（TRAIN 目标数） | FINAL_VISUAL_AUDIT.md |
|---|---|---|
| Tomato Early blight leaf | 213 | 252 |
| Tomato Septoria leaf spot | 430 | 431 |
| Tomato leaf | 396 | 444 |
| Tomato leaf bacterial spot | 278 | 281 |
| Tomato leaf late blight | 219 | 228 |
| Tomato leaf mosaic virus | 261 | 262 |

> 注：FINAL_VISUAL_AUDIT.md 的每类目标数偏大，与 label_list.txt（合计恰 3861）口径不一致，成因报告未解释，属待确认。

### 冲突 7（记录类，非数值）：direction2 诊断报告「TEST 未触碰」声明与其 §5 表格出现 `TEST_000024_…Rust-2017…jpg` 图片名存在字面矛盾

- 来源：`experiments/direction2/DIRECTION2_DIAGNOSTIC.md`（全文声明「TEST 未触碰」，但 §5 第 9 行引用了 TEST_ 前缀图片名）。仅作为历史记录如实指出，未做任何访问/重算。

### 冲突 8（记录类，非数值）：GPU_WEB_VALIDATION_REPORT.md 自述路径与实际不一致

- 该报告 §12 自述「`experiments/final_evaluation/GPU_WEB_VALIDATION_REPORT.md`（本文件）」，但实际文件位于项目根目录 `GPU_WEB_VALIDATION_REPORT.md`（`experiments/final_evaluation/` 下无此文件）。属自引用路径笔误，待确认。

### 冲突 9（记录类）：S1 最终报告实际路径

- 报告实际位于 `experiments/direction5/S1_CG_ACS_FINAL_REPORT.md`（另有 `S1_CG_ACS_100e/archive/S1_CG_ACS_FINAL_REPORT.md` 归档副本）；不存在 `S1_CG_ACS_100e/S1_CG_ACS_FINAL_REPORT.md`。

---

## 五、暂无实测数据（无原始报告支撑，禁止补充/虚构）

| 项目 | 状态 |
|---|---|
| 数据集官方下载 URL / license 原文 / 论文链接（PlantVillage、IP102） | 暂无实测记录（仅 PlantDoc 检测版记录了 GitHub 与 CC-BY-4.0） |
| 中文类名官方映射 | 暂无（报告多处注明「项目当前无官方中文类名映射」） |
| RK3588 / RKNN 实测 FPS、量化结果 | 暂无（接口已预留，未部署，未转换） |
| TensorRT 加速实测 | 暂无（BASELINE_SUMMARY 注明「TensorRT 未启用」） |
| 训练过程完整曲线 / loss 曲线图 | 未在报告中整理（仅有原始 train.log / vdlrecords） |
| 系统架构图 / 技术架构图 | 暂无（报告中为文字描述） |
| 系统界面截图 | 暂无（报告为文字验收） |
| 应用场景量化收益（节省成本/准确率提升幅度等） | 暂无（报告明确「未加入任何虚假准确率/节省成本指标」） |

---

*本索引由只读盘点生成，不修改任何原始报告；所有事实均指向原始证据文件。*
