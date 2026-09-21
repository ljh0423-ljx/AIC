# PRD · 赛道四（AI+农业主题赛）选题文档

> **赛道**：第八届全球校园人工智能算法精英大赛 · 算法主题赛 · AI+农业（全智赛组委会〔2026〕28号）
> **文档性质**：选题决策 + 产品需求文档（PRD）
> **生成日期**：2026-08-31
> **项目底座**：AIC2026_AgriVision（本地已有，代码/模型/数据全部就绪）
> **硬性时间**：报名 + 初赛作品提交截止 **2026-10-15 20:00**；总决赛 11 月底线下答辩

---

## 1. 选题结论 + 淘汰逻辑

### 1.1 结论

**选赛题一：农业视觉目标检测与识别挑战赛。**

选题理由一句话：本项目就是一道已完成工程闭环的 RGB 图像农作物病虫害目标检测赛题——13 类叶部病害检测模型已训练锁定、VAL/TEST 指标已实测归档、CPU/GPU 双后端推理与 Web 演示平台已交付，赛题考察的全部指标（识别准确率、召回率、模型大小、推理效率）均有真实数据支撑，**初赛材料 80% 已就绪**。

### 1.2 五个命题逐一评估

| 命题 | 匹配度 | 判定 | 核心理由 |
|---|---|---|---|
| **一、农业视觉目标检测与识别挑战赛** | ★★★★★ | **✅ 选定** | 赛题要求"运用深度学习、目标检测……基于 RGB 或多光谱影像，精准识别农作物病虫害"，与项目能力逐项对齐：PP-YOLOE+-m 已实现 3 作物 13 类叶部病害检测，TEST mAP@0.5:0.95=0.417、Precision=0.685、Recall=0.452、参数量 23.57M、FPS=16.6 全部有实测归档；复杂光照/遮挡/尺度变化正是 PlantDoc 数据与消融实验覆盖的难点（TEST AP small/medium/large = 0.408/0.294/0.444）。 |
| 二、农业数据智能分析与精准预测赛 | ★☆☆☆☆ | ❌ 排除 | 赛题要求"基于传感器时序数据、基因型（SNP）、气象土壤、水质参数等多源结构化数据，构建……产量预估、病害程度分级等模型，重点考察多源异构数据融合、时间序列分析"。项目**纯视觉输入、纯检测输出**，无任何时序/结构化数据管线，无 RMSE/R² 类预测能力，等于从零重做。 |
| 三、农业数字孪生与三维建模创新赛 | ★☆☆☆☆ | ❌ 排除 | 赛题要求"基于激光雷达、深度相机或多视角重建技术，构建作物、果树或农田系统的三维数字模型"。项目无点云/深度相机/三维重建任何积累，PP-YOLOE+-m 是 2D 检测模型，完全无法复用。 |
| 四、农业智能装备自主作业挑战赛 | ★☆☆☆☆ | ❌ 排除 | 赛题要求"设计并制作具备自主作业能力的农业机器人……融合机械设计、运动控制"。项目无硬件本体、无机械/控制能力；RK3588 仅为软件侧预留接口（`inference/detector.py:521` RKNNBackend 占位），不可混同于装备赛道。 |
| 五、农业智能体与大模型决策服务赛 | ★☆☆☆☆ | ❌ 排除 | 赛题核心技术栈为"大语言模型、知识图谱、检索增强生成"。项目**明确没有** RAG/知识库/大模型功能（事实库 §15 与代码均可证实），从零搭建智能体与大模型赛题工作量大、且无法复用现有检测优势，违背"复用同一套项目底座"原则。 |

**淘汰逻辑总结**：五个命题中只有赛题一是纯视觉检测命题，其余四个分别需要结构化数据建模、三维重建、机器人硬件、大模型技术栈——均为项目零积累方向，全部排除。

---

## 2. 项目名称 + 一句话简介

- **项目名称**：AI 农业病害智能检测系统（AgriVision · Agricultural Disease Intelligent Detection System）
- **一句话简介**：基于 PP-YOLOE+-m 的多作物叶部病害 RGB 图像目标检测系统，覆盖番茄/苹果/葡萄 3 类作物 13 类病害与健康状态，交付 CPU/GPU 双后端统一推理引擎与 Gradio Web 检测平台，并以可插拔后端接口预留 RK3588 边缘部署路径。

---

## 3. 背景与痛点（引用赛题通知原文）

赛题通知原文指出："传统的农作物病虫害识别、杂草分辨、生长态势判断等依赖人眼和经验，效率低、覆盖有限。"——这正是本项目要解决的核心痛点：

1. **人工识别效率低**：植保专家经验依赖强、覆盖有限，基层种植者面对病害"看不准、说不清"；
2. **类间相似难辨**：番茄类内多种病害（早疫病、斑枯病、晚疫病、细菌性斑点病）病灶形态相近，肉眼极易混淆（项目 TEST 误差分析显示主要混淆即发生在这几类之间，见 `experiments/final_evaluation/M_FINAL_TEST_REPORT.md` §3.5）；
3. **数据质量被忽视**：公开农业图像数据集普遍存在 EXIF 旋转元信息与实际像素不一致的"隐形错位"，直接训练会损伤精度——本项目用 EXIF-fix 数据修复解决了这一问题；
4. **落地工具缺场景适配**：通知所述"技术场景适配不足、普惠性工具短缺"，对应本项目交付的免安装、浏览器直接可用的 Web 检测平台与一键式 CLI 批量推理。

赛题同时强调"重点考察模型在复杂光照、遮挡、尺度变化条件下的检测精度、推理速度与轻量化部署能力"，"鼓励针对农业边缘计算场景的模型优化与工程化落地"——本项目分别以多尺度训练（320~768）、小/中/大目标分级 AP 实测、统一 FPS 口径计时与 DetectorBackend 可插拔架构响应（详见 §7 创新点）。

---

## 4. 目标用户与使用场景

| 用户 | 场景 | 对应能力 |
|---|---|---|
| 种植户 / 家庭农场主 | 手机拍摄叶片照片，通过 Web 页面上传，几秒内获知病害类别与位置 | Gradio Web 上传检测（`web/app.py`） |
| 植保技术员 / 农技推广人员 | 田间巡检后批量导入照片，一键生成整批检测统计与报告 | 批量推理引擎 + JSON/CSV 导出（`inference/batch_infer.py`） |
| 合作社 / 农业企业 | 周期性对示范基地做病害普查，按类别统计发生率 | 类别统计柱状图 + 批次汇总（`web/app.py` 类别统计 Tab） |
| 农业院校师生 / 科研人员 | 复现模型、验证指标、做消融对比 | 完整配置/脚本/实验报告（`configs/`、`scripts/`、`experiments/`） |
| 比赛评委 / 演示观众 | 观看 15 张案例自动播放演示，验证真实检测能力 | 比赛演示模式（`web/app.py` demo mode） |

> 边界声明（诚实性约束）：系统输出为 AI 视觉检测结果，**不等同于专业植保诊断**；Web 界面每次检测均附带该免责声明（`web/app.py` `_agri_result_card`）。

---

## 5. 核心功能清单

| # | 功能 | 状态 | 实现位置（证据路径） |
|---|---|---|---|
| 1 | 13 类叶部病害目标检测（番茄 8 / 苹果 3 / 葡萄 2） | ✓ 已实现 | `inference/detector.py`（PaddleBackend）；模型 `experiments/direction_m/M_SCALEUP_100e/checkpoints/best_model.pdparams` |
| 2 | CPU/GPU 双后端推理 + 设备自动选择（auto/cpu/gpu） | ✓ 已实现 | `inference/detector.py`（resolve_device/PaddleBackend）、`web/backend.py` |
| 3 | 后端统一抽象 + 注册表（可插拔换后端） | ✓ 已实现 | `inference/detector.py`（DetectorBackend ABC / register_backend / create_backend） |
| 4 | 单图 / 批量 / 文件夹递归 / 限额 CLI 推理 | ✓ 已实现 | `inference/infer.py`、`inference/batch_infer.py` |
| 5 | 失败隔离（单图损坏不中断批次）+ 时间戳输出目录 | ✓ 已实现 | `inference/batch_infer.py`（run_batch） |
| 6 | 置信度阈值真实过滤（Web 滑块 + CLI --conf） | ✓ 已实现 | `inference/postprocess.py`（filter_by_conf）、`web/app.py` |
| 7 | 结果可视化（bbox+类别+置信度）与原图对比 | ✓ 已实现 | `inference/visualize.py`、`web/app.py` 可视化结果 Tab |
| 8 | 检测结果表格 / 类别统计柱状图 / 批次汇总（真实 FPS） | ✓ 已实现 | `web/app.py`（检测结果表格 / 类别统计 / 批次汇总 Tab） |
| 9 | 结果导出 JSON / CSV / TXT / 检测图下载 | ✓ 已实现 | `inference/postprocess.py`、`inference/batch_infer.py`、`web/app.py` 结果下载 Tab |
| 10 | Gradio 6.24.0 Web 检测平台（模型单例加载 + 启动自检） | ✓ 已实现 | `web/app.py`、`web/backend.py`（get_detector/self_check） |
| 11 | 15 张比赛演示案例 + 自动播放演示模式（仅 VAL 图片） | ✓ 已实现 | `web/make_demo_data.py`、`web/demo_data/`、`web/app.py` 比赛演示模式 Tab |
| 12 | 模型训练 / 评估 / 环境一键脚本 | ✓ 已实现 | `scripts/train_baseline.sh`、`scripts/eval_baseline.sh`、`scripts/install_env.sh` |
| 13 | 多方向消融实验证据链（EXIF 修复、容量放大等） | ✓ 已实现 | `experiments/`（21 份报告，索引 `PROJECT_EXPERIMENT_INDEX.md`） |
| 14 | RK3588 边缘部署接口（后端占位） | ⚠️ 需改造 | `inference/detector.py:521` RKNNBackend——**仅注册占位，未实现**；赛期内可作为"工程化落地规划"展示架构，不声称已部署 |
| 15 | 模型轻量化（INT8 量化 / 剪枝 / 蒸馏） | ✗ 需补 | 无任何量化产物；23.57M/94.3 MB 为 fp32 权重。初赛阶段以"轻量化部署路径规划"表述，不做虚假声明 |
| 16 | 复杂光照/遮挡专项增强（对比测试报告） | ✗ 需补 | 可基于已有 `test_analyze.json` 错误分析整理出复杂场景专项分析，属文档级工作 |
| 17 | 中文类名展示 | ✗ 需补 | `inference/config.py` `CLASS_NAMES_CN` 留空（项目原则：无官方中文映射不自造）；若赛题材料需中文，须团队自行定义并标注"非官方译名" |
| 18 | 演示视频 / 答辩 PPT / 技术方案文档 | ✗ 需补 | 初赛提交物，赛期内完成（排期见 §11） |

---

## 6. 技术方案

### 6.1 总体架构

```
RGB 图片 -> Gradio Web（web/app.py）或 CLI（inference/infer.py）
        -> DetectorBackend 统一接口（inference/detector.py）
        -> PaddleBackend（paddle_cpu / paddle_gpu）
        -> 预处理 Resize/Normalize/Permute（与官方 reader 逐位一致）
        -> PP-YOLOE+-m 前向（PaddleDetection v2.9.0）
        -> NMS 解码 + clsid2catid 对齐（COCO 1-13 id）
        -> Detection 统一结构（inference/postprocess.py）
        -> 可视化 / 统计 / JSON/CSV 导出 / Web 展示
```

### 6.2 模型与 Paddle 官方组件映射

| 组件 | 采用的官方实现 | 论文出处（arXiv） |
|---|---|---|
| 检测模型 | **PP-YOLOE+-m**（PaddleDetection 官方结构：CSPResNet 骨干 + CustomCSPPAN 颈部 + PPYOLOEHead，**零自定义算法模块**） | PP-YOLOE: An Evolved Version of YOLO — arXiv:2203.16250 |
| 训练框架 | PaddleDetection v2.9.0（git HEAD b25522a0，源码零修改） | — |
| 深度学习框架 | PaddlePaddle 2.6.2（cu118，AutoDL RTX 4090 24GB） | — |
| 分类损失 | Varifocal Loss（VFL，官方 PP-YOLOE+ 默认） | VarifocalNet — arXiv:2008.13367 |
| 回归/质量损失 | Generalized Focal Loss（QFL + 分布式 bbox 表示） | Generalized Focal Loss — arXiv:2006.04388 |
| 标签分配 | Task Alignment Learning（TAL） | TOOD — arXiv:2108.07755 |
| 预训练权重 | Objects365-m 官方预训练（ppyoloe_crn_m_obj365_pretrained.pdparams） | Objects365 — arXiv:1908.00723 |
| Web 展示 | Gradio 6.24.0 | — |
| 边缘部署（规划） | Paddle2ONNX -> RKNN-Toolkit2 INT8（**未实施**，仅路径规划） | — |

### 6.3 训练配置要点（与 `configs/ppyoloe_plus_crn_m_100e_agrivision.yml` 一致）

- 配置用 `_BASE_` 组合项目数据集配置 + PaddleDetection 官方 runtime/optimizer/结构/reader 配置；
- 与 baseline（PP-YOLOE+-s）唯一差异：depth_mult 0.33→0.67、width_mult 0.50→0.75、Objects365-m 预训练；
- 实际训练参数：100 epoch、bs=16（`-o TrainReader.batch_size=16`）、base_lr=0.002（线性缩放）、CosineDecay + LinearWarmup(5)、EMA 0.9998、输入 640（多尺度 320~768）；
- 训练耗时 ≈2.5 h，显存峰值 ~20.3 GB / 25.9 GB。

### 6.4 验证协议（数据纪律）

- **模型选择唯一依据 = VAL（113 张）**；TEST（114 张）仅做一次最终独立评估，全程封闭，不参与任何训练/调参/选择（`experiments/final_evaluation/FREEZE_CHECK.json` PASS）；
- 推理/自检/演示只用 VAL 图片与 val.json，禁止读取 test.json（代码层已落实：`inference/config.py` ANNO_REL_PATH 指向 val.json；`web/app.py` `_load_demo_cases` 校验演示图来源必须在 images/val 下）。

---

## 7. 创新点（紧扣赛题评审要点）

赛题评审要点原文："比赛成绩由**识别准确率、召回率、模型大小及推理效率**等指标综合评定，鼓励针对**农业边缘计算场景的模型优化与工程化落地**。"

| # | 创新点 | 对应评审要点 | 证据 |
|---|---|---|---|
| 1 | **数据质量驱动的精度提升**：识别并修复公开数据集 EXIF 旋转像素错位（11 张，像素烘焙修复），消融验证有效后锁定数据基础 | 识别准确率 | `PROJECT_FINAL_STATUS.md` §2（Arm B 胜出，关键经验"EXIF 修复有效（+0.0148）"） |
| 2 | **受控消融驱动的模型选型**：多方向单变量消融（数据清洗/EXIF/损失加权/密度校准/辅助监督/预训练/容量放大），以固定判据筛选出容量放大（s→m）为唯一稳定增益方向，VAL +0.018 且 13 类中 10 升 2 平 2 微降、强类无回退 | 识别准确率 | `experiments/direction_m/M_SCALEUP_100e/M_SCALEUP_FINAL_REPORT.md` §2 |
| 3 | **复杂尺度条件的分级实测**：多尺度训练（320~768）+ TEST AP small/medium/large = 0.408/0.294/0.444、AR@100=0.686，直面"尺度变化条件下的检测精度" | 检测精度（尺度变化） | `M_FINAL_TEST_REPORT.md` §3.1 |
| 4 | **全口径真实推理效率**：区分评估循环/批量/Web/单图四种口径分别实测（16.6 / 12.84 / 12.00 / 1.96 FPS），且 CPU↔GPU 逐目标一致性验证（11=11 目标，最大 bbox 差 0.04 px、置信度差 0.000175），GPU 约为 CPU 的 30 倍 | 推理效率 | `GPU_BATCH_INFERENCE_REPORT.md`、`PROJECT_FINAL_FACTS.md` §8 |
| 5 | **面向边缘场景的工程化架构**：DetectorBackend 统一接口 + 后端注册表，Web/后处理/可视化零依赖 Paddle 私有实现，新增 RKNN 后端只需实现同类接口并注册——为农业边缘计算场景落地预留标准化路径（**当前为预留接口，未部署，如实表述**） | 轻量化部署与工程化落地 | `inference/detector.py` §后端注册机制 |
| 6 | **可运行、可演示的完整闭环**：数据→训练→独立验证→双后端推理→Web 平台→15 张演示案例，非离线评测脚本；含失败隔离、空状态、中文友好错误、免责声明等生产级细节 | 工程化落地 | `web/app.py`、`inference/batch_infer.py` |

---

## 8. 评测指标（全部为项目实测值）

来源：`PROJECT_FINAL_FACTS.md` §7/§8、`experiments/final_evaluation/M_FINAL_TEST_REPORT.md` §3。

### 8.1 精度（评审要点：识别准确率、召回率）

| 指标 | VAL（选模型用） | TEST（独立评估，114 图/423 GT） |
|---|---|---|
| mAP@0.5:0.95 | **0.428** | **0.417** |
| mAP@0.5 | 0.590 | 0.601 |
| mAP@0.75 | 0.492 | 0.475 |
| AP small / medium / large | — | 0.408 / 0.294 / 0.444 |
| AR@1 / AR@10 / AR@100 | — | 0.264 / 0.598 / 0.686 |
| **Precision / Recall / F1（conf=0.5）** | 0.576 / 0.400 / 0.472 | **0.685 / 0.452 / 0.544** |
| 密集场景 Recall（GT≥10，10 图/137 GT） | 0.210（VAL 5 图/62 GT） | 0.372 |

### 8.2 模型大小（评审要点：模型大小）

- 参数量：**23,568,416（23.57M）**
- 模型大小：**94.3 MB**（fp32 pdparams）

### 8.3 推理效率（评审要点：推理效率；口径不可混用）

| 场景 | 设备 | FPS |
|---|---|---|
| TEST 评估循环（114 图，tools/eval.py） | RTX 4090 | **16.6** |
| 批量推理（8 张 VAL，run_batch 引擎） | RTX 4090 | 12.84 |
| Web 批量检测（8 张 VAL） | RTX 4090 | 12.00 |
| 批量推理（8 张 VAL，18.84 s） | 本地 Windows CPU | 0.42 |
| 15 张演示批量（25.46 s） | 本地 Windows CPU | 0.59 |

### 8.4 13 类 TEST AP（可作 classwise 证据）

Apple Scab Leaf 0.592、grape leaf black rot 0.658、grape leaf 0.594、Apple rust leaf 0.550、Tomato leaf late blight 0.565、Tomato Septoria leaf spot 0.515 等强类 AP≥0.55；弱类（mosaic 0.095、bacterial spot 0.154）为 16–35 GT 小样本波动，已在报告中如实归因（`M_FINAL_TEST_REPORT.md` §4）。

---

## 9. 数据来源

| 项 | 内容 | 证据 |
|---|---|---|
| 数据集 | **PlantDoc** 目标检测数据集（GitHub `pratikkayal/PlantDoc-Object-Detection-Detection-Dataset`，论文 "PlantDoc: A Dataset for Visual Plant Disease Detection"，CoDS-COMAD 2020，CC-BY-4.0） | `PROJECT_FINAL_FACTS.md` §2 |
| 规模 | 3 作物（番茄/苹果/葡萄）/ 13 类 / 1144 图 / 3861 目标 | 同上 |
| 最终划分（EXIF-fix 锁定版） | train 915 图/3079 标注、val 113 图/350 标注、test 114 图/423 标注 | `PROJECT_FINAL_STATUS.md` §1 |
| EXIF-fix 处理流程 | 原始数据存在 11 张 EXIF 旋转方向（Orientation 6/8）与实际像素不一致的图片，在独立副本上做**像素烘焙修复**（像素旋转到显示方向），消融验证有效后作为所有后续实验数据基础；TEST 分片全程未改动 | `docs/PROJECT_TECHNICAL_SPEC.md` §4.4 |
| 13 类清单（英文原名，模型输出顺序 0–12） | Tomato Early blight leaf、Tomato Septoria leaf spot、Tomato leaf、Tomato leaf bacterial spot、Tomato leaf late blight、Tomato leaf mosaic virus、Tomato leaf yellow virus、Tomato mold leaf、Apple Scab Leaf、Apple leaf、Apple rust leaf、grape leaf、grape leaf black rot | `dataset/processed_detection_exiffix/annotations/val.json` / `label_list.txt` |
| 类别映射纪律 | COCO id 1–13（1-based），模型 0–12 输出经 clsid2catid 对齐，**不重新编号**；运行时从 val.json 读取（不使用 test.json） | `inference/config.py` |
| 外部数据声明 | PlantVillage、IP102 仅做过迁移可行性审计，**最终模型未使用外部数据训练** | `docs/PROJECT_TECHNICAL_SPEC.md` §4.1 |

---

## 10. 可行性分析

### 10.1 项目就绪度（最大优势）

| 交付物 | 状态 |
|---|---|
| 训练好的最终模型 + 配置 + checkpoint（94.3 MB） | ✅ 已锁定（FREEZE_CHECK PASS，md5 钉死） |
| VAL/TEST 全套指标 + 每类 AP + PR 曲线 + 错误分析 | ✅ 已归档（`experiments/final_evaluation/`、`bbox_pr_curve/`） |
| 推理引擎 + Web 平台 + 15 张演示案例 | ✅ 已交付并实测 |
| 唯一缺口 | 技术方案文档适配、答辩 PPT、演示视频（纯文档/视频工作，无技术风险） |

**核心判断：本选题是"材料制作型"任务而非"技术攻关型"任务**，与其他需要从零建模的命题相比风险最低。

### 10.2 团队能力匹配

- **算法功底**：蓝桥杯国奖经验——扎实的算法与工程基础，支撑过本项目的多方向消融实验设计与受控变量判据制定（提升/回退判定、固定决策规则等）；
- **限时开发节奏感**：项目在数周内完成了"数据审计→消融矩阵→最终训练→独立评估→Web 交付"完整闭环（时间线见 `PROJECT_FINAL_STATUS.md` §2），与赛期 6.5 周的文档冲刺节奏同量级；
- **答辩经验**：蓝桥杯国奖答辩训练出的临场表达与质询应对能力，直接复用于总决赛线下答辩；项目已自带 15 张自动播放演示模式（`web/app.py` 比赛演示模式 Tab）为答辩演示兜底；
- **技术栈完全对口**：Python / PaddlePaddle / PaddleDetection v2.9.0 / Gradio 6.24.0 / OpenCV，全部为项目日常使用技术，无新学习成本。

### 10.3 环境可行性

- 指标类材料全部来自 AutoDL RTX 4090 + Paddle 2.6.2 环境的既有实测报告，无需重跑；
- 本地 Windows（Paddle 3.3.0 CPU，无 GPU）已验证 CPU 推理路径与官方工具链逐位一致，可承担演示视频录制（注意：录制时须 `web\run_web.bat` 启动以规避 GBK 编码问题；CPU 演示口径 FPS=0.59，演示视频按 15 张案例说明真实耗时即可，不混用 GPU 数字）；
- 本地 Paddle 3.3.0 与训练环境 2.6.2 版本不匹配为已知事项：模型结构加载 0 缺失/0 意外键已实测验证，仅用于本地演示，正式指标一律引用 AutoDL 实测值。

---

## 11. 周排期（对齐初赛截止 2026-10-15 20:00）

> 起算日：2026-08-31（今日）。共约 6.5 周，含国庆缓冲。作品形式：技术方案 + 答辩 PPT + 演示视频。

| 周次 | 日期 | 任务 | 产出 |
|---|---|---|---|
| W1 | 09/01–09/07 | 完成报名（缴费 500 元/队、确认 ≤3 人 + ≤2 指导老师名单）；确定提交材料清单与评分对照表 | 报名回执、材料清单 |
| W2 | 09/08–09/14 | 技术方案文档初稿：以 `docs/PROJECT_TECHNICAL_SPEC.md` 为底本，按赛题一评审要点（准确率/召回率/模型大小/推理效率）重排章节；迁移 §8 全部实测指标并逐项标注证据来源 | 技术方案 v1 |
| W3 | 09/15–09/21 | 补齐"待补充"素材：系统界面截图（启动 Web 后实拍）、补充复杂光照/遮挡/尺度专项分析（整理 `test_analyze.json`/`test_errors.json` 已有数据，纯文档工作）；补齐架构图 | 技术方案 v2 + 截图/图集 |
| W4 | 09/22–09/28 | 演示视频脚本与录制：Web 上传检测全流程 + 类别统计 + 15 张演示模式；配音解说指标口径（VAL/TEST、CPU/GPU 分开说） | 演示视频 v1 |
| W5 | 09/29–10/05 | 答辩 PPT 制作（含国庆缓冲）；内部走查数据一致性（每个数字回溯 PROJECT_FINAL_FACTS.md） | PPT v1 |
| W6 | 10/06–10/12 | 模拟答辩 + 材料终审（技术方案/PPT/视频三件套互检；检查无"已部署 RK3588"类虚假表述）；平台上传演练 | 终版三件套 |
| 缓冲 | 10/13–10/15 | 平台正式提交，**10/15 20:00 前完成**，预留 ≥48h 应对平台拥堵 | 提交回执 |

> 复赛/总决赛：初赛通过后，复赛仍为"技术方案 + PPT + 视频"形式，主要增量是把边缘部署（RKNN INT8 转换链路）从"规划"推进为"小规模验证"（校准集仅用 VAL，遵守 TEST 封闭纪律）；总决赛 11 月底线下答辩，以现有 15 张演示模式为现场兜底。

---

## 12. 风险与对策

| # | 风险 | 等级 | 对策 |
|---|---|---|---|
| 1 | **绝对指标不算顶尖**（TEST mAP=0.417） | 中 | 评委综合评定"准确率、召回率、模型大小及推理效率"——以口径完整性与过程严谨性取胜：小样本（1144 图/13 类）设定下的 mAP@0.5=0.601、P=0.685，且每类 AP/混淆对/漏检分析完整归档；绝不虚报数字 |
| 2 | 密集场景 Recall 偏低（TEST 0.372） | 中 | 如实呈现为"已知局限 + 已归因"（漏检集中在单图 18–23 GT 的密集图与低对比度弱类），体现误差分析深度而非回避；不虚构严重程度或防治建议 |
| 3 | 无中文类名映射 | 低 | 材料中使用英文类名 + 团队自译对照表并标注"非官方译名"；不改代码即不引入不一致风险 |
| 4 | RK3588 仅接口预留，评委追问"部署" | 中 | 严格口径："预留接口 + 明确的转换链路规划（export_model→Paddle2ONNX→RKNN-Toolkit2 INT8，校准集用 VAL）"，以 `inference/detector.py` 架构证明工程可行性；复赛阶段再推进小规模验证 |
| 5 | 本地 Windows GBK 编码 / Paddle 版本不匹配 | 低 | 演示一律走 `web\run_web.bat`（内置 chcp 65001 + PYTHONUTF8=1）；正式指标只引用 AutoDL 2.6.2 实测报告，本地仅做演示 |
| 6 | 时间冲突（与其他赛道/课业撞期） | 中 | 本选题为"材料制作型"，W5 含国庆缓冲；W6 起所有材料进入终审冻结，只改错不加新 |
| 7 | 平台提交拥堵 | 低 | 排期要求 10/13–10/15 提交，预留 48h；提前演练上传格式 |
| 8 | 团队分工风险（≤3 人） | 低 | 文档/视频/PPT 三线并行：技术方案（算法主力）+ PPT/答辩（有答辩经验成员）+ 视频剪辑，互为备份 |

---

## 附：与赛道二 PRD 的底座复用说明

本文档与 `docs/PRD_赛道二_算法创新赛道_选题文档.md` **复用同一套项目底座**，零重复开发：

| 共用底座 | 路径 | 赛道四用法 | 赛道二用法 |
|---|---|---|---|
| PP-YOLOE+-m 检测模型 + checkpoint | `experiments/direction_m/M_SCALEUP_100e/checkpoints/best_model.pdparams` | 精度/召回/模型大小指标主体 | 软件系统的 AI 引擎 |
| PaddleDetection v2.9.0 推理框架 | `PaddleDetection/`（本地副本，不改源码） | 训练/评估证据链 | 同左 |
| 推理引擎（后端抽象/批量/后处理） | `inference/` | 检测能力与效率证据 | 软件核心模块 |
| Gradio 6.24.0 Web 系统 | `web/` | 演示平台 | 软件产品本体 |

差异仅在**叙事与材料侧重**：赛道四讲"算法指标 + 边缘部署路径 + 工程化落地"；赛道二讲"软件产品完整性 + 架构创新"。两份赛题的全部增量工作均为文档/PPT/视频，可由同一团队在一套素材库上分叉产出（技术方案 ↔ 软件说明书共享 §8 指标表与图集）。

---

*本 PRD 中所有指标与事实均取自 `PROJECT_FINAL_FACTS.md` / `PROJECT_FINAL_STATUS.md` 及其标注的原始实验报告，未虚构任何数字；未实现能力（RKNN 部署、量化、中文类名）均如实标注。*
