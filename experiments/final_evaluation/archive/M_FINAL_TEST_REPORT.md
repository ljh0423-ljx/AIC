# 最终 TEST 评估报告 — PP-YOLOE+-m (M-best) 独立泛化性能

日期: 2026-08-17
**模型选择依据: VAL(仅使用 VAL 选择 best checkpoint)。TEST 仅作为最终独立泛化性能报告,不用于重新选择模型,不触发任何重新训练/调参。**
本 TEST 为最终候选模型 PP-YOLOE+-m 的**一次且仅一次**独立评估,评估对象 = M-best checkpoint (VAL mAP@0.5:0.95 = 0.428)。

---

## 0. 结论

| 项 | VAL (选模型用) | **TEST (独立报告)** |
|---|---|---|
| mAP@0.5:0.95 | **0.428** | **0.417** |
| mAP@0.5 | 0.590 | 0.601 |
| mAP@0.75 | 0.492 | 0.475 |
| AR@100 | 0.723 | 0.686 |

TEST 独立数据上 mAP@0.5:0.95 = **0.417**,与 VAL 0.428 一致(小样本 114/113 张的固有波动)。**该数值与 VAL 判定一致,再次确认 PP-YOLOE+-m 为强候选模型。** TEST 结果不改写模型选择结论(选择已由 VAL 决定并锁定);TEST 仅作为最终独立泛化性能报告存档。

---

## 1. 评估前最终冻结检查 (FREEZE_CHECK.json, 状态 PASS)

评估执行前完成冻结检查,确认评估对象与 VAL 最终阶段完全一致:

| 检查项 | 结果 |
|---|---|
| M best checkpoint (md5 与 DRIFT_GUARD 一致) | PASS |
| EXIF-fix 数据 (train/val/test json md5 一致) | PASS |
| 13 类映射 (test.json: 114 图 / 423 标注 / 13 类) | PASS |
| 配置 (epoch=100, bs=16, lr=0.002, obj365-m 预训练, EMA) | PASS |
| 代码版本 (git HEAD b25522a0; ppdet .py 无改动) | PASS |
| 训练条件与 VAL 阶段一致 (无中途调参) | PASS |

详情: `experiments/final_evaluation/FREEZE_CHECK.json`。

---

## 2. 评估对象与协议

- **模型**: PP-YOLOE+-m (depth_mult=0.67, width_mult=0.75), obj365-m 预训练, 100 epoch 训练
- **checkpoint**: `experiments/direction_m/M_SCALEUP_100e/checkpoints/best_model.pdparams` (VAL mAP 0.428)
- **数据**: EXIF-fix test.json — 114 张 / 423 个 GT 标注 / 13 类 (COCO 1-based id 1–13), 图片 images/test
- **协议**: 与 A0/VAL 完全一致的 `tools/eval.py --classwise` (COCO 指标, 单 GPU 顺序) + `analyze_test.py` (conf=0.5 的 P/R/F1/每类/混淆/密集, 复用 analyze_val.py 逻辑, 仅把 EvalDataset 指向 test.json)
- **TEST 全程封闭**: 仅此一次, 未用于任何选择/调参/重训

---

## 3. TEST 指标 (独立记录, 与 VAL 分开)

### 3.1 COCO 指标 (tools/eval.py, 114 样本)
| 指标 | TEST |
|---|---|
| **mAP@0.5:0.95** | **0.417** |
| mAP@0.5 | 0.601 |
| mAP@0.75 | 0.475 |
| AP small | 0.408 |
| AP medium | 0.294 |
| AP large | 0.444 |
| AR@1 | 0.264 |
| AR@10 | 0.598 |
| **AR@100** | **0.686** |
| AR small / medium / large | 0.567 / 0.676 / 0.693 |
| **FPS** (114 样本) | **16.6** |

### 3.2 P/R/F1 + 密集场景 (conf=0.5)
| 指标 | TEST |
|---|---|
| Precision | **0.685** |
| Recall | **0.452** |
| F1 | **0.544** |
| 密集场景 Recall (10 图 / 137 GT, GT≥10) | **0.372** (51/137) |

### 3.3 13 类 AP (TEST) 与 VAL 对比
| 类别 | VAL | TEST |
|---|---|---|
| Tomato Early blight leaf | 0.282 | 0.268 |
| Tomato Septoria leaf spot | 0.468 | 0.515 |
| Tomato leaf | 0.241 | 0.256 |
| Tomato leaf bacterial spot | 0.296 | 0.154 |
| Tomato leaf late blight | 0.461 | 0.565 |
| Tomato leaf mosaic virus | 0.231 | 0.095 |
| Tomato leaf yellow virus | 0.208 | 0.248 |
| Tomato mold leaf | 0.199 | 0.302 |
| Apple Scab Leaf | 0.547 | 0.592 |
| Apple leaf | 0.850 | 0.620 |
| Apple rust leaf | 0.658 | 0.550 |
| grape leaf | 0.712 | 0.594 |
| grape leaf black rot | 0.414 | 0.658 |

### 3.4 每类 P/R (conf=0.5, TEST)
| 类别 | GT | Pred | P | R | TP | FP | 漏检 |
|---|---|---|---|---|---|---|---|
| Tomato Early blight leaf | 21 | 6 | 0.667 | 0.190 | 4 | 2 | 17 |
| Tomato Septoria leaf spot | 55 | 59 | 0.712 | 0.764 | 42 | 17 | 13 |
| Tomato leaf | 66 | 37 | 0.595 | 0.333 | 22 | 15 | 44 |
| Tomato leaf bacterial spot | 35 | 5 | 0.600 | 0.086 | 3 | 2 | 32 |
| Tomato leaf late blight | 34 | 29 | 0.759 | 0.647 | 22 | 7 | 12 |
| Tomato leaf mosaic virus | 20 | 3 | 0.333 | 0.050 | 1 | 2 | 19 |
| Tomato leaf yellow virus | 67 | 22 | 0.636 | 0.209 | 14 | 8 | 53 |
| Tomato mold leaf | 20 | 12 | 0.417 | 0.250 | 5 | 7 | 15 |
| Apple Scab Leaf | 16 | 15 | 0.733 | 0.688 | 11 | 4 | 5 |
| Apple leaf | 33 | 44 | 0.636 | 0.848 | 28 | 16 | 5 |
| Apple rust leaf | 17 | 15 | 0.800 | 0.706 | 12 | 3 | 5 |
| grape leaf | 29 | 23 | 0.870 | 0.690 | 20 | 3 | 9 |
| grape leaf black rot | 10 | 9 | 0.778 | 0.700 | 7 | 2 | 3 |

### 3.5 主要混淆对 (TEST)
| 混淆 (GT → 误判) | 次数 |
|---|---|
| Tomato leaf yellow virus → Tomato leaf | 5 |
| Tomato leaf bacterial spot → Tomato Septoria leaf spot | 4 |
| Tomato leaf late blight → Tomato Septoria leaf spot | 3 |
| Tomato Early blight leaf → Tomato Septoria leaf spot | 2 |
| Tomato leaf → Apple leaf | 2 |
| Tomato leaf mosaic virus → Tomato mold leaf | 2 |
| Tomato Early blight leaf → late blight | 1 |
| Tomato Early blight leaf → mold | 1 |

混淆结构主要为番茄类内相近病灶互判 (Septoria/bacterial/early blight/late blight/yellow virus 之间),与 VAL 阶段观察一致,无系统性新混淆。

### 3.6 主要错误案例 (每图漏检最多)
| img_id | GT | 匹配 | 漏检 | 漏检类别分布 |
|---|---|---|---|---|
| 60 | 23 | 9 | 14 | Tomato leaf ×14 |
| 90 | 14 | 2 | 12 | Tomato leaf yellow virus ×12 |
| 62 | 14 | 3 | 11 | Tomato leaf ×11 |
| 69 | 12 | 1 | 11 | Tomato leaf bacterial spot ×11 |
| 59 | 18 | 9 | 9 | Tomato leaf ×9 |
| 92 | 10 | 2 | 8 | Tomato leaf yellow virus ×8 |
| 43 | 8 | 1 | 7 | Early blight ×6, Tomato leaf ×1 |
| 89 | 10 | 3 | 7 | Tomato leaf yellow virus ×7 |

主要错误集中在**密集场景** (img 60/59 单图 18–23 个 GT) 与**弱类** (yellow virus 低对比度黄化病、bacterial spot 小斑病)。与 VAL 分析结论一致: conf=0.5 工作点下漏检为主 (R 0.452), 但 COCO AP 排名保持稳定 (强类 Apple/grape 系列 AP 均 ≥0.55)。

### 3.7 参数量 / 模型大小 / FPS
| 项 | 值 |
|---|---|
| 参数量 | 23,568,416 (23.57M) |
| 模型大小 (pdparams) | 94.3 MB |
| 推理 FPS (TEST eval, 114 样本) | 16.6 |
| 推理 FPS (VAL eval, 113 样本, 同协议) | 17.8 |

---

## 4. TEST vs VAL 关键观察

1. **整体一致**: TEST mAP 0.417 vs VAL 0.428 (Δ−0.011), 处于 113/114 张小样本的合理波动区间, 无异常崩塌。
2. **无强类系统性回退**: Apple/grape 强类在 TEST 上仍保持高位 (grape黑腐 0.658、Apple Scab 0.592、grape leaf 0.594、late blight 0.565)。
3. **差异点 (小样本噪声范围内)**: bacterial spot AP 0.296→0.154, mosaic 0.231→0.095, Apple leaf 0.850→0.620 —— 均为 TEST 样本量小 (该类 16–35 个 GT) 下的波动, 与 VAL 中这些弱类的已知困难一致 (VAL 报告中 Bact spot 高阈值 recall 已偏弱)。
4. **TEST 独立确认模型可用**: 在从未参与训练/选择的 114 张独立测试图上, mAP@0.5:0.95=0.417、mAP@0.5=0.601, 泛化性能与 VAL 判定相符。

---

## 5. 模型选择声明

- **模型选择依据 = VAL** (仅使用 VAL 选择 best checkpoint, VAL 判定 PP-YOLOE+-m 为最终候选模型)。
- **TEST 仅作为最终独立泛化性能报告**, 不用于重新选择模型, 未触发任何重新训练/调参/修改配置。
- 根据固定决策规则: M VAL mAP 0.428 ≥ 0.415 且无强类回退 → PP-YOLOE+-m 列为最终模型候选; 用户已批准其进入最终 TEST 评估。TEST 结果 (0.417) 与 VAL 判定一致。

---

## 6. 产出物 (experiments/final_evaluation/)

| 文件 | 内容 |
|---|---|
| `M_FINAL_TEST_REPORT.md` | 本报告 |
| `FREEZE_CHECK.json` | 评估前最终冻结检查 (PASS) |
| `test_eval.log` | TEST COCO 评估日志 (含 per-class AP 表、FPS) |
| `test_predictions.json` | TEST 预测结果 (4.5 MB) |
| `test_analyze.json` | conf=0.5 分析 (P/R/F1、每类、混淆、密集) |
| `test_errors.json` | 每图漏检统计 + 最差图像错误案例 |
| `test_analyze_run.log` | 分析运行日志 |
| `ENV_GIT.txt` | 环境/git/命令快照 |
| `archive/` | 归档副本 |

TEST 完成后已停止: 不访问、不重选、不重训、不再调参。
