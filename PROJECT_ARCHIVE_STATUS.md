# AIC2026_AgriVision — 项目最终归档状态

- **归档检查日期**: 2026-08-16
- **检查性质**: 只读审计(未运行任何训练/评估,未删除/压缩/修改任何文件、模型或数据)
- **项目定位**: 已进入最终归档状态。**TEST 完全封闭**,未用于训练/调参/模型选择;原始数据集零修改。

---

## 1. 实验目录完整性核对(全部 PASS)

| 实验目录 | 报告 | 配置 | 日志 | checkpoint | VAL 评估 | 状态 |
|---|---|---|---|---|---|---|
| `baseline_v1` | ✅ BASELINE_SUMMARY / BASELINE_DIAGNOSTIC_REPORT / DATASET_CONSISTENCY_REPORT | ✅ config.yml / config_dataset.yml | ✅ train.log + VDL | ✅ 22 pdparams(best + final + 每5epoch) | ✅ val_final_metrics / val_per_class / 混淆矩阵 | **完整**(TEST 最终评估已完成) |
| `ablation_data_clean` | ✅(归并入 ablation_compare/ABLATION_VAL_SUMMARY) | ✅ config.yml / config_dataset.yml | ✅ train.log + VDL | ✅ 22 pdparams | ✅ val_eval_best/final | 完整 |
| `ablation_data_exiffix` | ✅(同上;best=0.4012,即 A0) | ✅ config.yml / config_dataset.yml | ✅ train.log + VDL | ✅ 22 pdparams | ✅ val_eval_best/final | 完整 |
| `direction2`(A1 CGPM / A2 固定margin) | ✅ DIRECTION2_ABLATION_REPORT / DIRECTION2_DESIGN / DIRECTION2_DIAGNOSTIC | ✅(A1/A2 配置在 `configs/`) | ✅ run_A1_wrapper.log / run_A2_wrapper.log | ✅ A1、A2 各 22 pdparams | ✅ val_eval + A0_A1_A2_metrics.csv | 完整(A1/A2 无效) |
| `ablation_dir3_B1`(class-weighted VFL) | ✅ DIRECTION3_B1_REPORT | ✅ config.yml / config_base.yml | ✅ train.log + VDL | ✅ 22 pdparams | ✅ val_eval A0/B1 best+final | 完整(B1 无效) |
| `direction4/d2_results`(推理侧) | ✅ D2_REPORT | ✅(仅推理,无训练) | ✅ train_infer.log | —(不训练) | ✅ A0_D2_metrics / density_d2_comparison / params_locked | 完整(D2 未达判据,停止) |
| `direction4/d1_formal`(density-aware VFL) | ✅ D1_REPORT | ✅ config.yml / config_base.yml | ✅ train.log + VDL | ✅ 22 pdparams | ✅ val_eval/D1_best + 4 份对比 CSV + 2 PNG | **完整(D1 无效,FAIL)** |

> checkpoint 完整性: **全部正式训练实验各 22 个 pdparams**(best_model + model_final + 每 5 epoch 共 20 个中间档),smoke 实验各 4 个(best + final + epoch0/1),无缺失。

## 2. 原始数据集 `processed_detection` 未被修改 ✅

- 全部文件 mtime **≤ 2026-08-14 18:03**(即数据准备窗口内),此后(8/14 晚 ~ 8/16 归档日)零改动。
- `test.json` md5 **`2008eddc2aa4fcd63ba720c33cc678da` 在原始 / clean / exiffix 三份副本中完全一致** → TEST 标注从未被任何下游改动。
- clean/exiffix 仅 train/val 与原始不同(文档化的清洗/EXIF-Fix 派生):train 图 916→915(去 1 重复),val 114→113(EXIF 旋转图烘焙),test 114→114 且标注 md5 不变。
- 原始划分: train 916 图 / 3084 标注,val 114 图 / 354 标注,test 114 图 / 423 标注,13 类。
- 所有训练均通过软链使用副本(`PaddleDetection/dataset/processed_detection -> processed_detection_exiffix|_clean`),原始目录未被训练读取/写入。

## 3. TEST 封闭性验证 ✅

- 唯一数据集配置 `configs/datasets/agrivision_detection.yml`:
  - `TrainDataset → annotations/train.json`、`EvalDataset → annotations/val.json`(全部训练/验证读取路径)
  - `TestDataset → annotations/test.json` **仅被 `scripts/eval_baseline.sh` 最终评估引用**
- 全部 train.log(9 个实验)**grep 无任何 `test.json` 引用**。
- 训练配置仅含 `TrainReader`/`EvalReader`,无 TestReader。
- **TEST 唯一一次读取 = baseline_v1 最终评估**(8/14 `eval_baseline.sh`,产物 `metrics/test_eval.json/bbox.json` 4.86MB + `test_eval.log`);顶层 `bbox.json` + `bbox_pr_curve/` 为其同级产物。此后 TEST 未被再访问。

## 4. 进程检查 ✅

- `ps aux | grep tools/train.py | tools/eval.py | tools/infer.py` → **无任何运行中进程**(已全部结束)。

## 5. 文件统计(experiments/ 全量)

| 类型 | 数量 | 类型 | 数量 |
|---|---|---|---|
| checkpoint .pdparams | **167** | checkpoint .pdema | 157 |
| checkpoint .pdopt | 157 | checkpoint .pdstates | 156 |
| 日志 .log | 44 | JSON | 34 |
| 文本 .txt | 32 | JPG | 31 |
| **CSV** | **27** | **报告 .md** | **21** |
| 配置 .yml | 18 | **PNG** | **17** |

- 文件总数 **861** 个,体积 **≈15.01 GB**(主要为 checkpoint)。
- 报告 21 份(见附录),CSV 27 份(全部对比/每类/密度/低置信分析),PNG 17 份(全部可视化),覆盖全部 7 个方向。

## 6. 各实验最终结论(项目总览)

| 实验 | 变量 | VAL mAP@.5:.95 | 判定 |
|---|---|---|---|
| Baseline v1 | 官方 baseline | 0.401(TEST 0.401) | ✅ 基准 |
| A0 = exiffix | EXIF 修复数据 | **0.4012** | ✅ 数据消融胜者 |
| A1 / A2 | CGPM 组内判别 | — | ❌ 无效 |
| B1 | class-weighted VFL | — | ❌ 无效 |
| D2 | 推理侧密邻增强 | 0.4008(< A0) | ❌ 未达判据,停止 |
| D1 | density-aware VFL 校准 | 0.4061(< 门槛 0.4062) | ❌ FAIL,不调参不二轮 |

## 7. 归档注意事项 / 已知 caveats

- **D1 未做 TEST 评估**(按指令"暂时不要做 TEST 最终评估";且 D1 判定无效)。
- **TEST 仅应在获得明确指令后进行最终一次独立评估**,不得再用于任何训练/调参/模型选择。
- 原始数据集、clean/exiffix 副本均未改动;PaddleDetection 源码仅 `ppyoloe_head.py` 被各方向修改(默认关闭,均有机制校验)。
- 如需恢复任何实验:训练/评估命令已存档于各实验 `train_command.txt` / `train_dir*.sh`;环境快照存于各实验 `environment.txt`;git commit 存于 `git_commit.txt`。

---

### 附录 A:报告清单(21 份)

`BASELINE_REPORT.md` · `ablation_compare/ABLATION_VAL_SUMMARY.md` · `ablation_dir3_B1/DIRECTION3_B1_REPORT.md` · `baseline_v1/{ABLATION_PLAN, BASELINE_DIAGNOSTIC_REPORT, BASELINE_SUMMARY, DATASET_CONSISTENCY_REPORT, EXIF_FIX_PLAN}.md` · `direction2/{ABLATION_DIRECTION2_PLAN, DIRECTION2_ABLATION_REPORT, DIRECTION2_DESIGN, DIRECTION2_DIAGNOSTIC}.md` · `direction3/{ABLATION_DIRECTION3_PLAN, DIRECTION3_DESIGN, DIRECTION3_DIAGNOSTIC}.md` · `direction4/{ABLATION_DIRECTION4_PLAN, DIRECTION4_DESIGN, DIRECTION4_DIAGNOSTIC}.md` · `direction4/d1_formal/D1_REPORT.md` · `direction4/d1_smoke/D1_SMOKE_REPORT.md` · `direction4/d2_results/D2_REPORT.md`
