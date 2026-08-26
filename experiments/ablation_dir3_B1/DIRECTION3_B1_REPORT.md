# 方向3 B1 消融实验报告 — Class-Weighted VFL

- **日期**: 2026-08-15
- **唯一变量**: VFL 分类损失乘类权重 `w_c`(`PPYOLOEHead.use_cls_weight=True`);其余与 A0 逐字段一致
- **数据基础**: EXIF-Fix 副本 `dataset/processed_detection_exiffix`(固定实验数据,原始数据集未改)
- **权重**: `class_weight_train.json`(仅 TRAIN 统计生成,冻结,Σw=13,均值 1.0,`w_bg=1.0`);严禁用 VAL/TEST 生成,已合规
- **评估**: 仅 VAL(113 图/350 目标);TEST 完全封闭
- **对照**: A0 = 方向2 EXIF-Fix best checkpoint,**用同一评估脚本重算**(`scripts/eval_dir3_b1.sh`),避免脚本差异

## 1. 实验设置

| 项 | A0 (对照) | B1 (实验) |
|---|---|---|
| 分类损失 | 原始 VFL | VFL × class weight |
| 模型 | PP-YOLOE+-s | 同(结构零改动) |
| epoch / bs / lr / seed | 100 / 16 / 0.002 / 0 | 同 |
| 增强 / 采样 / NMS / 推理阈值 | 官方默认 | 同(未动) |
| 回归损失 (iou/dfl) | 默认 | 同(未动) |
| 权重文件 | — | `class_weight_train.json` (TRAIN-only, 冻结) |

- B1 训练:100 epoch,seed=0(`--enable_ce True`),训练期 VAL best mAP@0.5:0.95 = **0.405**(epoch ~89),final = 0.404
- 实现检查(`scripts/verify_b1_implementation.py`)4 项全 PASS:默认关闭 A0 一致、加权仅乘 cls loss、class_id 0~12 ↔ category_id 1~13 严格对齐、class weight 进入梯度(逐类 ∝ w_c, r=1.0)
- 2-epoch Smoke Test 通过(0 ERROR/OOM/NaN)

## 2. 总体指标 (VAL, COCO metric)

| 指标 (best) | A0 | B1 | Δ | 说明 |
|---|---|---|---|---|
| **mAP@0.5:0.95** | 0.401 | **0.405** | **+0.004** | 主判据门槛 +0.005 → **差 0.001 未过** |
| mAP@0.5 | 0.554 | 0.566 | +0.012 | |
| Precision@conf0.5 | 0.717 | 0.716 | −0.001 | |
| Recall@conf0.5 | 0.471 | 0.563 | **+0.092** | |
| F1@conf0.5 | 0.569 | 0.630 | **+0.061** | |
| 定位正确 (loc_ok) | 165 | 197 | +32 | |
| 定位且类对 (cls_ok) | 124 | 145 | +21 | |
| **定位对但类错** | **41** | **52** | **+11** | 错误位移(同方向2 CGPM 模式) |

model_final 对比:A0 0.386 vs B1 0.404(+0.018),B1 在最终权重上更稳健。

## 3. 13 类 AP 与 Recall (best)

| 类别 | w_c | A0 AP | B1 AP | ΔAP | A0 Rec | B1 Rec | ΔRec |
|---|---|---|---|---|---|---|---|
| Early blight | 1.10 ↑ | 0.210 | 0.267 | **+0.057** | 0.571 | 0.619 | +0.048 |
| Septoria | 0.56 ↓ | 0.461 | 0.440 | **−0.021** ⚠ | 0.568 | 0.649 | **+0.081** |
| leaf | 0.64 ↓ | 0.215 | 0.246 | +0.031 | 0.417 | 0.583 | **+0.167** |
| **bacterial spot** | 0.83 ↓ | 0.284 | 0.159 | **−0.125** ⚠ | 0.625 | 0.625 | 0.000 |
| late blight | 1.19 ↑ | 0.379 | 0.490 | **+0.111** | 0.692 | 0.769 | +0.077 |
| mosaic virus | 0.91 ↓ | 0.236 | 0.237 | +0.001 | 0.273 | 0.424 | **+0.152** |
| yellow virus | 0.27 ↓ | 0.185 | 0.168 | −0.017 | 0.123 | 0.158 | +0.035 |
| mold leaf | 0.79 ↓ | 0.160 | 0.189 | +0.029 | 0.250 | 0.406 | **+0.156** |
| Scab | 1.51 ↑ | 0.484 | 0.500 | +0.016 | 0.600 | 0.767 | **+0.167** |
| **Apple leaf** | 0.97 ≈ | 0.811 | 0.780 | **−0.031** ⚠ | 1.000 | 1.000 | 0.000 |
| **rust** | 1.27 ↑ | 0.650 | 0.601 | **−0.049** ⚠ | 0.833 | 0.833 | 0.000 |
| **grape leaf** | 1.09 ↑ | 0.738 | 0.713 | **−0.025** ⚠ | 0.786 | 0.857 | +0.071 |
| black rot | 1.87 ↑ | 0.403 | 0.470 | **+0.067** | 0.545 | 0.636 | +0.091 |

⚠ = 强类回退告警 (ΔAP < −0.02)。

## 4. 成功判据判定

| 判据 | 要求 | 实测 | 结论 |
|---|---|---|---|
| 主判据 | B1 − A0 VAL mAP@0.5:0.95 ≥ +0.005 | **+0.004** | ❌ **未达标 (差 0.001)** |
| 副判据 | 总体 Recall 或 F1 提升 | Recall +0.092, F1 +0.061 | ✅ 达标 |
| 负向红线 | 无强类回退 (ΔAP < −0.02) | **5 类告警** (Septoria/bact/Apple/rust/grape) | ❌ **触发** |

> **总体判定: B1 消融失败。** 主判据差 0.001 未过,且负向红线独立触发(5 类强类回退,其中 bacterial spot −0.125 为大幅回退)。按既定规范,**不得调整权重或参数后重试**;以下如实记录机制分析。

## 5. 机制分析 (为何 class-weighted VFL 未提升 mAP)

**5.1 机制方向被证实有效** —— 权重与效果正相关:
- 大幅上调类全部提升: black rot (w=1.87) AP +0.067 / Rec +0.091, late blight (1.19) +0.111, Scab (1.51) +0.016 & Rec +0.167
- 点名弱势类 Recall 大幅提升: mosaic +0.152, mold +0.156, yellow +0.035, Septoria +0.081 —— 正是用户要求关注的"重点弱势类别 Recall"
- 总体 Recall +0.092 / F1 +0.061 是**机制确实改变了类间梯度强调**的直接证据

**5.2 但代价 = 强类回退,净 mAP 几乎不动 (+0.004)**:
- 被下调的强类回退: bacterial spot −0.125(最大), Septoria −0.021; 部分上调类也回退 (Apple leaf −0.031, rust −0.049, grape −0.025)—— 后三者 w>1 但 VAL 上 AP 下降, 说明类间权重重分配带来的"此消彼长"在 mAP 上相互抵消
- **loc_but_wrong_cls 41 → 52 (+11)**: 提升弱类 Recall 时引入了更多"定位对但类别错",与方向2 CGPM 的误差位移模式一致

**5.3 与诊断先验一致 (诚实结论)**:
- 方向3 诊断已证明:点名漏检类 (yellow/mosaic/mold) 的漏检主因是密集场景置信度压低, **而非欠训练/不平衡**;yellow 甚至是 TRAIN 实例数第一的类
- B1 真实作用面是 TRAIN 稀少类(blackrot/Scab/rust/lateblight)—— 这些类的 Recall/AP 确实受益
- 因此 "B1 提升弱类召回但不提升 mAP" 是**设计阶段就已预告的预期结果**, 不是实现缺陷

**5.4 直接结论**: 在 mAP@0.5:0.95 为主指标的判据下, class-balanced VFL **不是有效方案**。其价值仅体现在 Recall/F1 维度(若比赛另有 Recall/F1 指标可作参考,但本实验主判据未过)。

## 6. 诚实记录与后续

- **不调参**: 按规则, B1 失败即记录, 不调整 w_c 缩放、不换权重重训。
- **B2 (density-aware inference)**: 被门控于 "B1 完成且有效"。**B1 判定失败 → B2 不进入设计/执行**, 方向3 就此收尾。
- **A0 仍是方向3 基准**: B1 未构成有效提升, 不取代 A0。
- **TEST**: 全程未接触, 保持封闭。

## 7. 产物清单

```
experiments/ablation_dir3_B1/
├── DIRECTION3_B1_REPORT.md           本报告
├── A0_B1_metrics.csv                 总体指标 (A0/B1 × best/final)
├── per_class_B1_comparison.csv       13 类 AP + Recall 对比
├── class_weight_effect.csv           w_c × ΔAP × ΔRecall
├── strong_class_regression.csv       强类回退检查
├── per_class_ap_A0_vs_B1.png         每类 AP 对比
├── per_class_recall_A0_vs_B1.png     每类 Recall 对比
├── class_weight_effect.png           权重-效果散点
├── val_eval/{A0,B1}_{best,final}/    4 组 VAL 评估原始输出
├── checkpoints/                      best_model(epoch~89) + model_final
├── train.log / config.yml / config_base.yml / class_weight_train.json
├── environment.txt / git_commit.txt / train_command.txt
└── (smoke: experiments/ablation_dir3_B1_smoke/ 含 mechanism_check.txt)
```

附:实现与机制校验脚本 `scripts/verify_b1_implementation.py`(4 项全 PASS),评估脚本 `scripts/eval_dir3_b1.sh`,分析脚本 `scripts/dir3_b1_analysis.py`,训练脚本 `scripts/train_dir3_b1.sh`。
