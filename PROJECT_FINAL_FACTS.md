# PROJECT_FINAL_FACTS — 最终事实库（可直接用于申报书 / 技术文档 / PPT / 答辩稿）

> 生成时间：2026-08-18
> 本文件只收录**能被原始实验/验证报告直接支持**的事实，每项均注明证据来源文件。凡无实测证据的，一律标注「待补充 / 暂无实测数据」，**不按常识补写、不虚构结果**。
> 与 `PROJECT_EXPERIMENT_INDEX.md`（总清单与冲突清单）配合使用；冲突数据以该文件「待确认数据」为准。

---

## 1. 项目与模型名称

| 事实 | 内容 | 证据 |
|---|---|---|
| 竞赛/任务 | AIC2026 农业视觉（AgriVision）目标检测 | PROJECT_FINAL_STATUS.md §1 |
| 项目名 | AIC2026_AgriVision | 目录 / PROJECT_FINAL_STATUS.md |
| 系统展示名 | 🌾 AI 农业病害智能检测系统 / Agricultural Disease Intelligent Detection System | WEB_DEMO_FINAL_REPORT.md §1 |
| **最终模型名称** | **PP-YOLOE+-m**（PP-YOLOE Plus，medium） | PROJECT_FINAL_STATUS.md §1/§3.1 |

---

## 2. 数据集（13 类）

| 事实 | 内容 | 证据 |
|---|---|---|
| 数据集名称 | PlantDoc 目标检测数据集 | PROJECT_FINAL_STATUS.md §1 |
| 规模 | 3 作物（番茄/苹果/葡萄）/ 13 类 / 1144 图 / 3861 目标 | 同上 / DATASET_FINAL_REPORT.md |
| 官方来源 | GitHub `pratikkayal/PlantDoc-Object-Detection-Dataset`；论文 "PlantDoc: A Dataset for Visual Plant Disease Detection (CoDS-COMAD 2020)"；许可证 CC-BY-4.0 | `D:\Fruit\PLANTDOC_DETECTION_AUDIT.md` |
| 官方下载 URL / license 原文 | **待补充** | 暂无实测记录 |
| 最终数据划分 | train 915 / val 113 / test 114（EXIF-fix 版）；原始版 train 916 / val 114 / test 114 | PROJECT_FINAL_STATUS.md §1 |

**13 类（英文原名，全项目一致）**：
Tomato Early blight leaf、Tomato Septoria leaf spot、Tomato leaf、Tomato leaf bacterial spot、Tomato leaf late blight、Tomato leaf mosaic virus、Tomato leaf yellow virus、Tomato mold leaf、Apple Scab Leaf、Apple leaf、Apple rust leaf、grape leaf、grape leaf black rot。
（证据：`dataset/processed_detection/label_list.txt`、PROJECT_FINAL_STATUS.md §4）

---

## 3. 模型结构

| 事实 | 内容 | 证据 |
|---|---|---|
| 架构 | PP-YOLOE+-m：官方 CSPResNet 骨干 + CustomCSPPAN 颈部 + PPYOLOEHead，**零自定义模块** | PROJECT_FINAL_STATUS.md §3.1 |
| 规模参数 | depth_mult=0.67, width_mult=0.75 | 同上 / config |
| 预训练 | Objects365-m（`ppyoloe_crn_m_obj365_pretrained.pdparams`） | config / PROJECT_FINAL_STATUS.md |
| 参数量 | 23,568,416（23.57M） | PROJECT_FINAL_STATUS.md §6 |
| 模型大小 | 94.3 MB（fp32 pdparams） | 同上 |
| 输入尺寸 | 640×640（训练多尺度 320~768） | config / 多处 |

---

## 4. Checkpoint 路径

| 事实 | 内容 | 证据 |
|---|---|---|
| **最终 checkpoint** | `experiments/direction_m/M_SCALEUP_100e/checkpoints/best_model.pdparams` | PROJECT_FINAL_STATUS.md §3.1 |
| md5 | `18bd0e99329be2113f57280188c227b6` | FREEZE_CHECK.json |
| 备选（非最终） | `…/checkpoints/model_final.pdparams`（VAL 0.408，md5 `b4e92405…`） | PROJECT_FINAL_STATUS.md §3.1 |
| 最终配置 | `configs/ppyoloe_plus_crn_m_100e_agrivision.yml` | 同上 |

---

## 5. 训练环境（AutoDL）

| 事实 | 内容 | 证据 |
|---|---|---|
| 平台 | AutoDL Linux | PROJECT_FINAL_STATUS.md §1 |
| **GPU 型号** | **NVIDIA GeForce RTX 4090，24 GB** | AUTODL_ENV_CHECK.md / GPU_SINGLE_INFERENCE_REPORT.md |
| PaddlePaddle | 2.6.2（cu118） | AUTODL_ENV_CHECK.md |
| Python | 3.12.3 | AUTODL_ENV_CHECK.md |
| CUDA / cuDNN | 11.8 / 编译 8.6，运行 9.1 | AUTODL_ENV_CHECK.md / GPU_SINGLE |
| PaddleDetection | v2.9.0（git HEAD `b25522a0`） | FREEZE_CHECK.json |

---

## 6. 训练参数（最终 M 模型）

| 参数 | 值 | 证据 |
|---|---|---|
| epoch | 100 | config / PROJECT_FINAL_STATUS.md §3.2 |
| batch_size | 16（实际，`-o TrainReader.batch_size=16`） | PROJECT_FINAL_STATUS.md §3.2 / FREEZE_CHECK.json |
| base_lr | 0.002（实际，线性缩放 0.001×16/8） | 同上 |
| lr 调度 | CosineDecay(100) + LinearWarmup(5) | config |
| EMA | 开启（0.9998） | PROJECT_FINAL_STATUS.md |
| 训练耗时 | ≈2.5 h | PROJECT_FINAL_STATUS.md §3.2 |
| 训练显存峰值 | ~20.3 GB / 25.9 GB（78%） | PROJECT_FINAL_STATUS.md §3.2 |

> 注意：磁盘上的 `.yml` 只写默认 `bs=8/lr=0.001`；实际训练用命令行 `-o` 覆盖为 `bs=16/lr=0.002`，复现需携带覆盖参数。

---

## 7. 验证结果（最终模型）

| 指标 | 值 | 证据 |
|---|---|---|
| VAL mAP@0.5:0.95（选模型用） | **0.428** | PROJECT_FINAL_STATUS.md / M_FINAL_TEST_REPORT.md |
| **TEST mAP@0.5:0.95（独立评估）** | **0.417** | M_FINAL_TEST_REPORT.md |
| TEST mAP@0.5 / mAP@0.75 | 0.601 / 0.475 | 同上 |
| TEST AP small / medium / large | 0.408 / 0.294 / 0.444 | 同上 |
| TEST AR@1 / AR@10 / AR@100 | 0.264 / 0.598 / 0.686 | 同上 |
| TEST Precision / Recall / F1（conf=0.5） | 0.685 / 0.452 / 0.544 | 同上 |
| 密集场景 Recall（10 图/137 GT） | 0.372 | 同上 |

> TEST 仅最终一次独立评估（114 图/423 GT），未参与任何选择/调参/重训。

---

## 8. 推理性能结果

### 8.1 GPU 推理（AutoDL RTX 4090 + Paddle 2.6.2）

| 口径 | 结果 | 证据 |
|---|---|---|
| TEST 评估 FPS | 16.6（114 图，tools/eval.py） | M_FINAL_TEST_REPORT.md |
| 批量推理 FPS | 12.84（8 张 VAL，预热排除，run_batch） | GPU_BATCH_INFERENCE_REPORT.md |
| Web 批量 FPS | 12.00（8 张 VAL，Web 实际计时） | GPU_WEB_VALIDATION_REPORT.md |
| 单图（冷启动） | 509.8 ms / FPS 1.96 | GPU_SINGLE_INFERENCE_REPORT.md |
| 单图检测样例 | Apple Scab Leaf，conf 0.569851，bbox (166.68,105.84,104.96,87.61) | GPU_SINGLE / GPU_BATCH / GPU_WEB 一致 |
| 批量成功/失败 | 8/8 成功，0 失败，11 目标 | GPU_BATCH_INFERENCE_REPORT.md |
| 显存峰值 | 741 MiB（远低于 24 GB） | GPU_BATCH_INFERENCE_REPORT.md |

### 8.2 CPU 推理（本地 Windows，Paddle 3.3.0，仅验证用）

| 口径 | 结果 | 证据 |
|---|---|---|
| 批量推理 FPS | 0.42（8 张 VAL，18.84 s） | INFERENCE_VALIDATION_REPORT.md |
| 演示 15 张 FPS | 0.59（25.46 s） | WEB_DEMO_FINAL_REPORT.md |
| 单图耗时 | 约 2.3~2.4 s | INFERENCE_VALIDATION / WEB_APP_VALIDATION |

### 8.3 CPU / GPU 对比（同 8 张 VAL 图、同引擎、同阈值 0.5）

| 对比项 | 结果 | 证据 |
|---|---|---|
| 批量 FPS | CPU 0.42 → GPU 12.84（**约 30×**） | GPU_BATCH_INFERENCE_REPORT.md §四/§十一 |
| 检测目标一致性 | CPU 11 = GPU 11，逐目标 bbox 一致（max_conf_delta 0.000175、max_bbox_delta 0.04px） | 同上 §七 |

---

## 9. Web 系统结果

| 事实 | 内容 | 证据 |
|---|---|---|
| 框架 | Gradio 6.24.0，默认端口 7860 | GPU_WEB_VALIDATION / WEB_APP_VALIDATION |
| 设备自动选择 | `AGRI_DEVICE=auto` → 有 GPU 用 paddle_gpu/gpu:0，无 GPU 回退 paddle_cpu（均验证） | GPU_WEB_VALIDATION_REPORT.md §二 |
| 模型加载 | 单例，全程仅 open best_model 1 次 | GPU_WEB_VALIDATION_REPORT.md |
| 核心功能 | 单图/批量/文件夹/拖拽上传、置信度滑块（真实过滤）、可视化、结果表格、类别统计柱状图、批次汇总、结果下载、农业场景卡片、比赛演示模式、损坏图失败隔离、无目标空状态 | GPU_WEB_VALIDATION_REPORT.md §十（14 项） |
| 农业结果卡 | 含免责声明「以上为 AI 视觉检测结果，不等同于专业植保诊断」 | WEB_DEMO_FINAL_REPORT.md §12 |
| 15 张演示 | 15/15 正常；番茄8/苹果4/葡萄3；目标 25；演示模式沿用预生成 VAL 结果 | GPU_WEB_VALIDATION_REPORT.md §八 |

---

## 10. 15 张演示案例结果

| 事实 | 内容 | 证据 |
|---|---|---|
| 数据来源 | `dataset/processed_detection_exiffix/images/val/`（仅 VAL，未用 TEST） | DEMO_CASES_REPORT.md |
| 规模 | 15 张 / 成功 15 / 失败 0 / 目标 25 | 同上 |
| 性能 | 总耗时 25.46 s，平均 1.697 s/张，FPS 0.59 | 同上 |
| 覆盖 | 单目标 9 / 多目标 5 / 密集 1；番茄 8 / 苹果 4 / 葡萄 3；类别 12/13 | 同上 |

（逐案例检测明细见 `web/DEMO_CASES_REPORT.md` 案例表，此处不重复。）

---

## 11. Backend 设计（已实现）

| 事实 | 内容 | 证据 |
|---|---|---|
| 统一接口 | `DetectorBackend`（load / predict / predict_batch / info / close） | inference/detector.py 描述于 WEB_DEMO_FINAL_REPORT.md §6 |
| 已注册后端 | `paddle_cpu` / `paddle_gpu`（PaddleBackend）+ `rknn_reserved`（占位，NotImplemented） | 同上 |
| 解耦 | Web/UI 仅依赖 DetectorBackend 接口与 Detection 结构，不依赖 Paddle 私有实现 | 同上 |

---

## 12. RK3588 后续规划（未实施）

| 事实 | 内容 | 证据 |
|---|---|---|
| 当前状态 | **未部署、未转换、未量化**；仅预留接口 | WEB_DEMO_FINAL_REPORT.md §7 / INFERENCE_VALIDATION_REPORT.md §10 |
| 规划路径 | 实现 `RKNNBackend(DetectorBackend)`；Paddle→ONNX→RKNN-Toolkit2 INT8；`export_model`→Paddle2ONNX→RKNN 量化（校准集可用 VAL，非 TEST）→板端集成 | 同上 |
| 预估（报告内） | RK3588 NPU INT8 预估 7–11 FPS@640（LOCAL_DEPLOYMENT_AUDIT.md，**为预估非实测**） | LOCAL_DEPLOYMENT_AUDIT.md |
| 官方参考（报告内） | RKNN 官方 ppyoloe_s INT8 ≈32.9 FPS（**引用，非本模型实测**） | LOCAL_DEPLOYMENT_AUDIT.md |

---

## 13. 关键产品清单（路径）

| 产物 | 路径 | 证据 |
|---|---|---|
| 最终模型 | `experiments/direction_m/M_SCALEUP_100e/checkpoints/best_model.pdparams` | PROJECT_FINAL_STATUS.md §9 |
| 最终配置 | `configs/ppyoloe_plus_crn_m_100e_agrivision.yml` | 同上 |
| M 实验报告 | `experiments/direction_m/M_SCALEUP_100e/M_SCALEUP_FINAL_REPORT.md` | 同上 |
| 最终 TEST 报告 | `experiments/final_evaluation/M_FINAL_TEST_REPORT.md` | 同上 |
| 冻结检查 | `experiments/final_evaluation/FREEZE_CHECK.json`（PASS） | 同上 |
| 每类 PR 曲线 | 顶层 `bbox_pr_curve/`（13 类 jpg）+ `bbox.json` | 同上 |
| 推理程序 | `inference/`（config/detector/postprocess/visualize/batch_infer/infer） | INFERENCE_VALIDATION_REPORT.md |
| Web 系统 | `web/`（config/backend/app.py + make_demo_data.py + demo_data/） | WEB_DEMO_FINAL_REPORT.md |

---

## 14. 直接可用于答辩/申报的「一句话结论」

> 最终模型 = PP-YOLOE+-m（EXIF-fix 数据，100 epoch，obj365-m 预训练），参数量 23.57M、模型 94.3 MB；VAL mAP@0.5:0.95 = **0.428**，独立 TEST mAP@0.5:0.95 = **0.417**；在 AutoDL RTX 4090 上批量推理真实 FPS ≈12.84（Web 端 12.00），GPU 约为本地 CPU（0.42 FPS）的 30 倍。已交付 Gradio Web 检测系统与 15 张比赛演示案例，预留 RK3588 后端接口。
> （证据：PROJECT_FINAL_STATUS.md、M_FINAL_TEST_REPORT.md、GPU_BATCH_INFERENCE_REPORT.md、GPU_WEB_VALIDATION_REPORT.md、WEB_DEMO_FINAL_REPORT.md）

---

## 15. 待补充 / 暂无实测数据（引用前需先补齐）

| 项目 | 状态 |
|---|---|
| 数据集官方下载 URL、license 原文、论文链接 | 待补充（仅 PlantDoc 检测版有 GitHub/CC-BY-4.0 记录） |
| 中文类名官方映射 | 暂无 |
| RK3588 / TensorRT 实测 FPS | 暂无实测（RK3588 仅接口预留 + 预估） |
| 系统架构图、技术架构图、界面截图 | 暂无（需另行制作） |
| 应用场景量化收益 / 创新点定量对比 | 待补充 |
| 训练 loss/mAP 曲线图 | 待从 train.log / vdlrecords 整理 |

---

*本事实库由只读盘点生成，每一项均标注证据来源文件；未修改任何原始报告、模型、数据集、代码与 Web 系统。*
