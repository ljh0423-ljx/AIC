# 方向2 消融实验报告 — CGPM (Confusion-Guided Pair-Margin) A0/A1/A2

- **日期**: 2026-08-15
- **数据基础**: EXIF-Fix (Arm B) 副本 `dataset/processed_detection_exiffix`(唯一实验数据)
- **数据分割**: 仅 VAL 评估 (113 图 / 350 目标); **TEST 完全封闭, 全程未使用**
- **混淆先验**: `confusion_prior_train_only.json`(仅 TRAIN 生成, 冻结; A0/A1/A2 共用)
- **唯一变量**: 分类损失机制 (VFL vs VFL+CGPM-guided vs VFL+CGPM-fixed)
- 模型结构 / epoch=100 / bs=16 / lr=0.002 / seed=0 / 增强 / 训练策略 三臂完全一致

## 1. 三臂定义

| 臂 | 数据 | 分类损失 | margin | 说明 |
|---|---|---|---|---|
| A0 | exiffix | 原始 VFL (PP-YOLOE 默认) | — | 对照 = Arm B (复用已有 checkpoint, 不重训) |
| A1 | exiffix | VFL + **CGPM guided** | `0.2 × prior_weight` | 混淆引导成对 margin (主方案) |
| A2 | exiffix | VFL + **CGPM fixed** | 统一 `0.2` | 与 A1 相同类别对, 普通 margin 对照 |

- A1/A2 推理零额外开销: CGPM 仅在训练时计入 loss, 无新增模块/FLOPs/后处理 (推理结构与 A0 完全一致)。
- CGPM 公式 (score 空间): `L_cg = λ · Σ_i a_i·relu(p_{i,r} − p_{i,g} + m[g][r]) / (Σ_i a_i·|R(g_i)| + ε)`, λ=0.25。

## 2. 总体指标 (VAL, COCO metric)

### 2.1 best_model (训练期 val 最佳)

| 指标 | A0 | A1 | A2 | A1−A0 | A2−A0 |
|---|---|---|---|---|---|
| mAP@0.5:0.95 | **0.401** | 0.397 | **0.405** | **−0.004** | **+0.004** |
| mAP@0.5 | 0.554 | 0.542 | 0.570 | −0.012 | +0.016 |
| Precision@conf0.5 | 0.717 | 0.724 | 0.696 | +0.007 | −0.021 |
| Recall@conf0.5 | 0.471 | 0.503 | 0.537 | +0.032 | +0.066 |
| F1@conf0.5 | 0.569 | 0.594 | 0.606 | +0.025 | +0.037 |
| 定位正确 TP (loc_ok) | 165 | 176 | 188 | +11 | +23 |
| 定位对且类对 (cls_ok) | 124 | 127 | 143 | +3 | +19 |
| **定位对但类错** | **41** | **49** | **45** | **+8** | **+4** |

### 2.2 model_final (epoch 100)

| 指标 | A0 | A1 | A2 |
|---|---|---|---|
| mAP@0.5:0.95 | 0.386 | 0.373 | 0.383 |
| mAP@0.5 | 0.546 | 0.519 | 0.546 |
| F1@conf0.5 | 0.629 | 0.623 | 0.605 |
| 定位对但类错 | 66 | 60 | 55 |

## 3. 13 类 AP (best_model, IoU=0.5:0.95) — 重点番茄类

| 类别 | A0 | A1 | A2 | A1−A0 | A2−A0 |
|---|---|---|---|---|---|
| Tomato Early blight leaf | 0.210 | 0.194 | 0.236 | −0.016 | **+0.026** |
| Tomato Septoria leaf spot | 0.461 | 0.442 | 0.362 | −0.019 | −0.099 |
| Tomato leaf | 0.215 | 0.236 | 0.211 | +0.021 | −0.004 |
| Tomato leaf bacterial spot | **0.284** | **0.175** | 0.200 | **−0.109** | −0.084 |
| Tomato leaf late blight | 0.379 | 0.399 | **0.523** | +0.020 | **+0.144** |
| Tomato leaf mosaic virus | 0.236 | 0.225 | 0.232 | −0.011 | −0.004 |
| Tomato leaf yellow virus | 0.185 | 0.210 | 0.206 | **+0.025** | +0.021 |
| Tomato mold leaf | 0.160 | 0.157 | 0.152 | −0.003 | −0.008 |
| Apple Scab Leaf | 0.484 | 0.568 | 0.534 | +0.084 | +0.050 |
| Apple leaf | 0.811 | 0.726 | 0.729 | −0.085 | −0.082 |
| Apple rust leaf | 0.650 | 0.714 | 0.772 | +0.064 | +0.122 |
| grape leaf | 0.738 | 0.720 | 0.678 | −0.018 | −0.060 |
| grape leaf black rot | 0.403 | 0.397 | 0.425 | −0.006 | +0.022 |

## 4. 命名混淆对双向混淆率 (best_model, conf=0.5/IoU=0.5)

| 混淆对 | GT 总数 | A0 | A1 | A2 | A1−A0 | A2−A0 |
|---|---|---|---|---|---|---|
| **bact→Septoria** | 16 | 7 (43.8%) | **5 (31.3%)** | 6 (37.5%) | **−12.5pp** | −6.3pp |
| Septoria→bact | 37 | 0 | 1 (2.7%) | 1 (2.7%) | +2.7pp | +2.7pp |
| mosaic→yellow | 33 | 3 (9.1%) | 2 (6.1%) | 3 (9.1%) | −3.0pp | 0 |
| yellow→mosaic | 57 | 0 | 0 | 2 (3.5%) | 0 | +3.5pp |
| EB→bact | 21 | 1 (4.8%) | 2 (9.5%) | 1 (4.8%) | +4.8pp | 0 |
| bact→EB | 16 | 0 | 1 (6.3%) | 0 | +6.3pp | 0 |
| mold→EB | 32 | 0 | 0 | **4 (12.5%)** | 0 | **+12.5pp** |
| EB→mold | 21 | 1 (4.8%) | 0 | 0 | −4.8pp | −4.8pp |

## 5. 成功判据判定

| 判据 | 要求 | 实测 (best) | 结论 |
|---|---|---|---|
| 主判据 1 | A1 − A0 的 VAL mAP@0.5:0.95 ≥ +0.005 | **−0.004** | ❌ **未达标** |
| 副判据 2 | 主要目标混淆对至少一组明确下降 | bact→Septoria −12.5pp | ✅ 达标 |
| 判据 3 | A1 应优于 A2 (支持 "guided > fixed") | A1 0.397 < A2 0.405 | ❌ **不成立 (反而 A2 更优)** |

> **总体判定: A1 消融失败 (主判据未过)。** 按既定规范, 不得人为调整参数后继续训练; 以下如实汇报失败原因。

## 6. 失败原因分析 (假设, 未经调参验证)

**6.1 CGPM 机制确实生效, 但只在 top-1 混淆对上**
- bact→Septoria (诊断中最严重的混淆对, 43.8%) 被 A1 显著压低到 31.3% (−12.5pp), A2 压到 37.5% — 证明 margin 机制能在指定类别对上生效。
- 但 A2 (固定 margin=0.2 施加于所有类对) 的 cls_ok 提升 +19, 远大于 A1 的 +3。原因假设: **guided 权重归一化使次级对手的 margin 趋近于 0** (如 bact 的 EB 对手 weight=0.0864 → margin=0.0173), 导致 guided 退化为"仅 top-1 对手获得有效 push"; fixed 对所有类对施加同样 0.2 的 push, 正则更强。

**6.2 错检"位移"效应: 目标对降低, 但类错总数上升**
- A1: loc_ok +11 但 cls_ok 仅 +3, **定位对但类错 41→49 (+8)**。
- 具体: bact→Septoria 减 2 例, 但同时 Septoria→bact +1、EB→bact +1、bact→EB +1。margin 压低目标对手分数后, 分数竞争在 score 空间的"邻居"重新分配, 未被 prior 覆盖的类别对被推高 — **小样本下误差位移而非消除**。
- A2 更重地压所有类对 (mold→EB 0→4, +12.5pp 反向恶化), 同样体现位移。

**6.3 mAP@0.5:0.95 轻微下降的机制假设**
- CGPM 对已正确分类的正 anchor 施加 penalty (relu 在 p_r − p_g + m > 0 时非零), 压低了部分正确检测的置信度, 使 PR 曲线在高 IoU / 中置信段略受损。
- λ=0.25 下 A1 训练损失曲线始终有限且下降, 无数值问题; 属优化目标竞争, 非实现缺陷。

**6.4 结论 (创新论断不支持)**
- 实测 **A2 (统一固定 margin) > A1 (confusion-guided margin)**, "混淆引导成对 margin 优于统一固定 margin" 的论断在本设置下不成立 — 与设计假设相反。
- 小样本 (915 train 图, 弱类每类仅几十例) 下, 细粒度 margin 的收益被误差位移吞没; 先验 weight 驱动的可变 margin 在多数类对上退化为无效微调。

## 7. 诚实记录与后续建议 (不执行, 仅供决策)

- **有效部分**: CGPM 可在不增推理开销的前提下定向压低指定混淆对 (bact→Septoria 实证 −12.5pp)。若比赛指标含 F1/召回, A1/A2 的 Recall@conf0.5 (+0.032/+0.066) 与 F1 (+0.025/+0.037) 有提升。
- **无效部分**: 主指标 mAP@0.5:0.95 无提升; "guided > fixed" 反证。
- **若未来重启 (需用户决定)**: 可考虑的假设方向 —— (a) 仅对 top-1 混淆对施加固定 margin (等价于 A2 的子集); (b) 降低 λ 或增大 margin 再测; (c) 将 margin 移到 NMS 前/得分融合而非训练损失。**以上均未执行, 不构成对结果的事后修饰。**
- A3 参数敏感性实验: **未执行** (按批准范围)。
- TEST: 未接触。

## 8. 产物清单

```
experiments/direction2/
├── DIRECTION2_ABLATION_REPORT.md      本报告
├── A0_A1_A2_metrics.csv               三臂总体指标 (best+final)
├── per_class_comparison.csv           13 类 AP 对比
├── confusion_pair_comparison.csv      命名混淆对双向对比
├── confusion_matrix_{A0,A1,A2}.csv    各臂 VAL 混淆矩阵
├── confusion_matrix_A0_A1_A2.png      三臂混淆矩阵可视化
├── per_class_ap_comparison.png        每类 AP 对比柱状图
├── confusion_pair_rates.png           混淆对混淆率折线
└── val_eval/{A0,A1,A2}_{best,final}/  6 组 VAL 评估原始输出
```
