# PROJECT_FINAL_STATUS — AIC2026_AgriVision 项目最终状态审计报告

- **审计日期**: 2026-08-17
- **审计性质**: 只读审计（未运行任何训练/评估，未修改任何模型、数据、配置或实验结果）
- **项目状态**: 全部训练与消融实验已结束，模型已锁定，进入最终归档/交付阶段
- **一句话结论**: 最终模型 = **PP-YOLOE+-m**（EXIF-fix 数据，100 epoch，obj365-m 预训练）`best_model` checkpoint；最终独立 TEST mAP@0.5:0.95 = **0.417**；参数 23.57M / 模型 94.3 MB / 推理 FPS 16.6（TEST，RTX 4090）。

---

## 1. 项目概况

| 项 | 内容 |
|---|---|
| 竞赛/任务 | AIC2026 农业视觉（AgriVision）目标检测 |
| 数据源 | PlantDoc 数据集（3 作物 / 13 类 / 1144 图 / 3861 目标） |
| 框架 | PaddlePaddle 2.6.2 + PaddleDetection v2.9.0（commit `b25522a0`） |
| 训练环境 | AutoDL Linux，RTX 4090 24GB（本地 Windows 无 GPU，只做审计/整理） |
| 数据基础 | **EXIF-fix 副本** `dataset/processed_detection_exiffix`（已锁定） |
| 最终模型 | **PP-YOLOE+-m**（depth_mult=0.67, width_mult=0.75） |

**数据划分（exiffix，最终锁定）**: train 915 图 / 3079 标注，val 113 图 / 350 标注，test 114 图 / 423 标注，13 类。
原始 `processed_detection`：train 916 / 3084，val 114 / 354，test 114 / 423（与 exiffix 仅 train/val 不同，test 标注 md5 三副本一致）。

---

## 2. 实验总览（全部方向已终结，禁止继续探索）

### 2.1 时间线 / 方向矩阵

| 阶段 | 实验 | 变量 | VAL mAP@0.5:0.95 | 判定 |
|---|---|---|---|---|
| 基线 | Baseline v1（A0 前身） | 官方 PP-YOLOE+-s baseline | 0.401（TEST 0.401） | 基准 |
| 数据消融 | Arm A `ablation_data_clean` | 清洗副本 | 0.386（best） | 负 |
| **数据消融** | **Arm B `ablation_data_exiffix`（A0）** | **EXIF 旋转像素烘焙修复（10 张 train/val）** | **0.4012（best）** | **✅ 数据消融胜者，作为后续数据基础** |
| 方向2 | A1 / A2（CGPM 组内判别 / 固定 margin） | 分类损失机制 | 0.375 / 0.385（见 §2.2） | ❌ 无效 |
| 方向3 | B1（class-weighted VFL） | VFL 分类损失乘类权重 | ≈0.38（见 §2.2） | ❌ 无效 |
| 方向4 | D1（density-aware VFL 校准） | 损失目标按密度校准 | 0.4061 | ❌ FAIL（差判据 0.0001） |
| 方向4 | D2（推理侧密邻增强） | 推理后处理 | 0.4008 | ❌ 未达判据，停止 |
| 方向5 | S1（PV_CORE CG-ACS 辅助监督） | 图像级分类辅助损失 λ=0.1 | 0.3668 | ❌ FAIL |
| 方向5 | B1-PVPRETRAIN（PV 预训练 backbone） | 两阶段分类预训练→检测微调 | 0.211 | ❌ FAIL（负迁移） |
| **方向M** | **M_SCALEUP_100e（A0→M 容量放大）** | **s→m（depth 0.33→0.67, width 0.50→0.75）** | **0.428（best）** | **✅ 最终模型候选，用户批准进入 TEST** |

### 2.2 已判定失败的探索（归档即可，禁止重启）

- **A1/A2（CGPM）**: 混淆引导 pair-margin 分类损失，A1≈0.375 / A2≈0.385，均低于 A0，无效。产物：`experiments/direction2/`、`experiments/ablation_dir2_A1|A2/`。
- **B1（class-weighted VFL）**: 类别加权 VFL，无效。产物：`experiments/ablation_dir3_B1/`。
- **D1（density-aware VFL）**: 密集场景 Recall 显著提升（≥10目标/图 0.210→0.290），但 mAP 0.4061 < 判据 0.4062（差 0.0001），且强类 3 类回退 >0.03 → **FAIL，禁止调参二轮**。产物：`experiments/direction4/d1_formal/`。
- **D2（推理侧）**: 簇投票重打分+保守重标，0.4008 < A0，未达判据即停止。产物：`experiments/direction4/d2_results/`。
- **S1（CG-ACS 辅助监督）**: PV_CORE 图像级分类辅助监督，0.3668，Apple Scab→Apple rust 混淆 3→15 恶化，FAIL。产物：`experiments/direction5/S1_CG_ACS_100e/`。
- **B1-PVPRETRAIN**: PV_CORE 分类预训练 backbone 两阶段，0.211（13/13 类 AP 全降，密集场景 0 命中），确认 **PV_CORE 方向整体失败，立即停止**。产物：`experiments/direction5/B1_PVPRETRAIN_100e/`。

> 关键经验（已在实验中确认，不必重试）：EXIF 修复有效（+0.0148）；模型容量放大（s→m）是唯一获得稳定增益的方向（+0.018）。

---

## 3. 最终模型锁定信息

### 3.1 模型与 checkpoint

| 项 | 值 |
|---|---|
| 架构 | PP-YOLOE+-m（官方 CSPResNet + CustomCSPPAN + PPYOLOEHead，零自定义模块） |
| 规模参数 | depth_mult=0.67, width_mult=0.75 |
| 预训练 | 官方 `ppyoloe_crn_m_obj365_pretrained.pdparams`（obj365-m） |
| **最终 checkpoint** | **`experiments/direction_m/M_SCALEUP_100e/checkpoints/best_model.pdparams`** |
| best_model md5 | `18bd0e99329be2113f57280188c227b6`（DRIFT_GUARD / FREEZE_CHECK 钉死） |
| best_model 大小 | 94,337,197 B = **94.3 MB** |
| 参数量 | **23,568,416（23.57M）** |
| VAL mAP@0.5:0.95（选模型用） | **0.428** |
| 备选 final checkpoint | `checkpoints/model_final.pdparams`（VAL 0.408，md5 `b4e92405…`，非最终模型） |

### 3.2 训练条件（FREEZE_CHECK 确认，实际运行时用 `-o` 覆盖）

| 项 | 值 |
|---|---|
| 配置 | `configs/ppyoloe_plus_crn_m_100e_agrivision.yml`（md5 `c107b879…`） |
| epoch / snapshot | 100 / 5 |
| **batch_size（实际）** | **16**（配置默认 8，命令行 `-o TrainReader.batch_size=16`） |
| **base_lr（实际）** | **0.002**（配置默认 0.001，`-o LearningRate.base_lr=0.002`；线性缩放：0.001×16/8） |
| lr 调度 | CosineDecay(max_epochs=100) + LinearWarmup(5) |
| EMA | 开启（0.9998 默认） |
| 输入 | 640（多尺度 320~768） |
| 数据 | EXIF-fix（软链 `PaddleDetection/dataset/processed_detection -> dataset/processed_detection_exiffix`） |
| 显存（M 正式训练） | 峰值 ~20.3 GB / 25.9 GB（78%），bs=16 安全 |
| 训练耗时 | ≈2.5 h（100 epoch） |
| git HEAD | `b25522a0`（PaddleDetection v2.9.0；M 期间 ppdet .py 零改动） |

> **注意**：磁盘上的 `.yml` 只体现默认 `bs=8/lr=0.001`；实际训练为 `bs=16/lr=0.002`（有训练日志 57 iter/epoch = 915/16、lr=0.002 为证）。复现训练/推理时需携带 `-o` 覆盖。

---

## 4. 13 类映射（最终锁定）

COCO JSON 中 category_id 为 **1–13**（1-based），ppdet COCODataSet 内部减 1 后与模型 0–12 输出对齐。以下为顺序即模型输出顺序（`label_list.txt`）：

| id(0-based) | id(COCO 1-based) | 类别 | 作物 | 性质 | TRAIN 图数/目标数 |
|---|---|---|---|---|---|
| 0 | 1 | Tomato Early blight leaf | Tomato | 病害 | 89 / 213 |
| 1 | 2 | Tomato Septoria leaf spot | Tomato | 病害 | 149 / 430 |
| 2 | 3 | Tomato leaf | Tomato | 健康 | 72 / 396 |
| 3 | 4 | Tomato leaf bacterial spot | Tomato | 病害 | 113 / 278 |
| 4 | 5 | Tomato leaf late blight | Tomato | 病害 | 110 / 219 |
| 5 | 6 | Tomato leaf mosaic virus | Tomato | 病害 | 55 / 261 |
| 6 | 7 | Tomato leaf yellow virus | Tomato | 病害（多目标） | 75 / 824 |
| 7 | 8 | Tomato mold leaf | Tomato | 病害 | 90 / 291 |
| 8 | 9 | Apple Scab Leaf | Apple | 病害 | 93 / 171 |
| 9 | 10 | Apple leaf | Apple | 健康 | 91 / 247 |
| 10 | 11 | Apple rust leaf | Apple | 病害 | 88 / 178 |
| 11 | 12 | grape leaf | Grape | 健康 | 69 / 220 |
| 12 | 13 | grape leaf black rot | Grape | 病害（小样本） | 64 / 133 |

---

## 5. 最终 TEST 指标（独立报告，仅一次，全程封闭）

评估对象 = M `best_model.pdparams`（VAL 0.428）。TEST（114 图 / 423 GT）**未参与任何选择/调参**。

### 5.1 COCO 指标（tools/eval.py --classwise）

| 指标 | VAL（选模型用） | **TEST（独立报告）** |
|---|---|---|
| **mAP@0.5:0.95** | 0.428 | **0.417** |
| mAP@0.5 | 0.590 | 0.601 |
| mAP@0.75 | 0.492 | 0.475 |
| AP small / medium / large | — | 0.408 / 0.294 / 0.444 |
| AR@1 / AR@10 / **AR@100** | — | 0.264 / 0.598 / **0.686** |
| AR small / medium / large | — | 0.567 / 0.676 / 0.693 |
| **FPS** | 17.8 | **16.6** |

### 5.2 P/R/F1 + 密集场景（conf=0.5）

| 指标 | TEST |
|---|---|
| Precision | **0.685**（191/279） |
| Recall | **0.452**（191/423） |
| F1 | **0.544** |
| 密集场景 Recall（10 图 / 137 GT，GT≥10） | **0.372**（51/137） |

### 5.3 13 类 AP（TEST）与 VAL 对比

| 类别 | VAL | TEST | 类别 | VAL | TEST |
|---|---|---|---|---|---|
| Tomato Early blight leaf | 0.282 | 0.268 | Tomato mold leaf | 0.199 | 0.302 |
| Tomato Septoria leaf spot | 0.468 | 0.515 | Apple Scab Leaf | 0.547 | 0.592 |
| Tomato leaf | 0.241 | 0.256 | Apple leaf | 0.850 | 0.620 |
| Tomato leaf bacterial spot | 0.296 | 0.154 | Apple rust leaf | 0.658 | 0.550 |
| Tomato leaf late blight | 0.461 | 0.565 | grape leaf | 0.712 | 0.594 |
| Tomato leaf mosaic virus | 0.231 | 0.095 | grape leaf black rot | 0.414 | 0.658 |
| Tomato leaf yellow virus | 0.208 | 0.248 | | | |

主要差异（bacterial spot 0.296→0.154、mosaic 0.231→0.095、Apple leaf 0.850→0.620）为小样本（该类 16–35 GT）波动，无强类系统性回退。
主要混淆：番茄类内相近病灶互判（yellow virus→Tomato leaf×5、bacterial spot→Septoria×4、late blight→Septoria×3 等）。

---

## 6. 模型参数量 / 大小 / FPS（统一协议实测）

| 项 | A0（s） | **M（m，最终）** | 变化 |
|---|---|---|---|
| 参数量 | 7,700,106 | **23,568,416（23.57M）** | 3.06× |
| 模型大小（pdparams） | 30.8 MB | **94.3 MB** | 3.06× |
| 推理 FPS（VAL eval，4090） | 19.8 | 17.8 | −10% |
| 推理 FPS（TEST eval，4090） | — | **16.6** | — |
| 训练显存峰值（bs=16） | — | ~20.3 GB / 25.9 GB | — |

---

## 7. 推理命令（最终模型）

### 7.1 单图推理（tools/infer.py，AutoDL 环境）

```bash
cd /root/autodl-tmp/AIC2026_AgriVision/PaddleDetection
python tools/infer.py \
  -c ../configs/ppyoloe_plus_crn_m_100e_agrivision.yml \
  -w ../experiments/direction_m/M_SCALEUP_100e/checkpoints/best_model.pdparams \
  --infer_img=../dataset/processed_detection_exiffix/images/test/某图.jpg \
  --output_dir=../experiments/direction_m/M_SCALEUP_100e/predictions
```

### 7.2 TEST/VAL 评估命令（复现 0.417）

```bash
cd /root/autodl-tmp/AIC2026_AgriVision/PaddleDetection
python tools/eval.py \
  -c ../configs/ppyoloe_plus_crn_m_100e_agrivision.yml \
  -o weights=../experiments/direction_m/M_SCALEUP_100e/checkpoints/best_model.pdparams \
     EvalDataset.image_dir=images/test \
     EvalDataset.anno_path=annotations/test.json \
  --classwise
```

> 依赖软链 `PaddleDetection/dataset/processed_detection -> dataset/processed_detection_exiffix`（AutoDL 上已建；本地缺失，见 §8）。

---

## 8. 本地推理条件评估（本机 = Windows，无 NVIDIA GPU）

结论：**当前本机不满足"开箱即用"的推理条件，但 CPU 推理路径基本具备可行性（未端到端验证）**。审计实测如下：

| 检查项 | 状态 | 说明 |
|---|---|---|
| 本地 NVIDIA GPU / nvidia-smi | ❌ 无 | `nvidia-smi` not found；README 亦声明本机无 GPU。**无 GPU 推理** |
| PaddlePaddle 版本 | ⚠️ **3.3.0（不匹配）** | 训练/评估环境为 **paddle 2.6.2**；本地装的是 3.3.0（GPU wheel）。README 明确警告 ppdet 2.9 不建议用 3.x。实测：模型结构构建 + checkpoint 加载在 3.3.0 下 **0 缺失 / 0 意外键**，前向可走到检测头，兼容性比预期好，但**未做端到端完整推理验证** |
| PaddleDetection 源码 | ✅ 完整 | `PaddleDetection/` 完整（v2.9.0 b25522a0），`ppdet` 可导入 |
| 最终 checkpoint | ✅ 存在 | `experiments/direction_m/M_SCALEUP_100e/checkpoints/best_model.pdparams`（94.3 MB，md5 与冻结值一致） |
| 数据集 | ✅ 存在 | `dataset/processed_detection_exiffix/`（train 915 / val 113 / test 114 图片 + 标注） |
| **数据集软链** | ❌ **缺失** | `PaddleDetection/dataset/processed_detection` **本地不存在**（AutoDL 训练时由脚本创建）。本地推理需改为 `-o TrainDataset.dataset_dir=…` 覆盖路径，或在 PaddleDetection/dataset 下建立 junction |
| 配置文件读取 | ⚠️ 需 UTF-8 | ppdet `load_config` 用系统默认编码打开 yml，Windows GBK 会 UnicodeDecodeError；需 `PYTHONUTF8=1`（Linux 无此问题） |
| 预训练权重 | — | `configs/*.yml` 中 `pretrain_weights` 指向 `/root/.cache/paddle/weights/…`，仅训练需要；**本地推理不需要** |

**本地 CPU 推理的最小步骤（如需要，须在明确指令下执行）**：
1. 修复软链/路径：`-o TrainDataset.dataset_dir=…`（或建 junction `PaddleDetection/dataset/processed_detection -> ../../dataset/processed_detection_exiffix`）；
2. 设置 `PYTHONUTF8=1`；
3. 用 §7.1 的 `tools/infer.py` 命令（CPU 上 FPS 会远低于 16.6，仅作验证/展示用）。
> 本审计未执行上述推理（按要求只做审计）。**建议以 AutoDL RTX 4090 + paddle 2.6.2 环境作为正式推理环境**（与 TEST 完全一致的软硬件）。

---

## 9. 关键产物清单

### 9.1 最终交付物
| 产物 | 路径 |
|---|---|
| 最终模型 | `experiments/direction_m/M_SCALEUP_100e/checkpoints/best_model.pdparams`（94.3 MB） |
| 最终配置 | `configs/ppyoloe_plus_crn_m_100e_agrivision.yml` |
| M 实验最终报告 | `experiments/direction_m/M_SCALEUP_100e/M_SCALEUP_FINAL_REPORT.md` |
| 最终 TEST 报告 | `experiments/final_evaluation/M_FINAL_TEST_REPORT.md` |
| TEST 冻结检查 | `experiments/final_evaluation/FREEZE_CHECK.json`（PASS） |
| TEST 评估日志/预测 | `experiments/final_evaluation/test_eval.log`、`test_predictions.json`（4.7 MB）、`test_analyze.json`、`test_errors.json` |
| 环境/命令快照 | `experiments/final_evaluation/ENV_GIT.txt` |
| TEST 归档副本 | `experiments/final_evaluation/archive/`（与当前逐字节一致） |
| 每类 PR 曲线 | 顶层 `bbox_pr_curve/`（13 类 jpg）+ 顶层 `bbox.json`（4.95 MB） |
| 数据锁定 | `dataset/processed_detection_exiffix/`（EXIF-fix 最终数据基础） |

### 9.2 实验仓库（全部实验，仅归档）
- `experiments/` 下 861 文件 ≈15.0 GB：167 个 pdparams checkpoint、21 份报告、27 份 CSV、17 份 PNG、44 份日志等。
- 报告总索引：`PROJECT_ARCHIVE_STATUS.md`（2026-08-16 归档状态）。

---

## 10. 冻结声明与约束（本次审计确认）

1. **模型已锁定**：PP-YOLOE+-m `best_model.pdparams`（VAL 0.428 / TEST 0.417）。**禁止重新训练、禁止调参、禁止修改 TEST**。
2. **TEST 已封闭使用**：仅 2026-08-17 最终独立评估一次，结果 0.417 已归档，不得再用于选择/调参。
3. **数据已冻结**：原始 `processed_detection`、`_clean`、`_exiffix` 三副本均未改动（test.json md5 三副本一致）。
4. **失败方向已封存**：A1/A2/CGPM、B1(class-w)、D1、D2、S1、B1-PVPRETRAIN 全部判定失败/无效，禁止继续探索。
5. **本次审计零修改**：仅新增本报告 `PROJECT_FINAL_STATUS.md`，未触碰任何模型/数据/配置/实验产物。

---

*审计生成：2026-08-17 | 数据来源：experiments/ 全部报告、FREEZE_CHECK.json、DRIFT_GUARD.json、M_FINAL_TEST_REPORT.md、PROJECT_ARCHIVE_STATUS.md、configs/、环境实测*
