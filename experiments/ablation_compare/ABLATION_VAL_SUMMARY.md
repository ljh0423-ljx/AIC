# ABLATION_VAL_SUMMARY — Arm A (clean) vs Arm B (EXIF-fix), VAL 对比

- 日期: 2026-08-14; 仅 VAL 分析; TEST 全程封闭 (未触碰)
- 唯一自变量: 10 张 train/val EXIF 图像素烘焙修复; 模型/配置/epoch=100/bs=16/lr=0.002/seed=0 完全一致

## 1. 总体指标 (VAL, COCO metric)

| checkpoint | 臂 | mAP@0.5:0.95 | mAP@0.5 | P (conf0.5) | R (conf0.5) | F1 | TP/FP/FN |
|---|---|---|---|---|---|---|---|
| best | A | 0.386 | 0.542 | 0.493 | 0.386 | 0.433 | 135/139/215 |
| final | A | 0.380 | 0.536 | 0.470 | 0.420 | 0.443 | 147/166/203 |
| best | B | 0.401 | 0.554 | 0.539 | 0.354 | 0.428 | 124/106/226 |
| final | B | 0.386 | 0.546 | 0.467 | 0.386 | 0.423 | 135/154/215 |

- **mAP@0.5:0.95 差异 (best)**: Arm B - Arm A = **+0.0148**  |  差异 (final): +0.0065
- per-class AP50:95 赢/输/平 (best): **7 胜 / 6 负 / 0 平**

## 2. EXIF 图 (img65 = TRAIN_000054) recall 变化

> 注: 11 张 EXIF 图中仅 img65 位于 VAL (可直接测 recall); 9 张在 train (影响训练), 1 张在 test (封闭不动)。

| 臂 | tag | GT | 检测数 | recall | TP | FN | 定位对类错 |
|---|---|---|---|---|---|---|---|
| A | best | 7 | 4 | 0.000 | 0 | 7 | 0 |
| A | final | 7 | 4 | 0.000 | 0 | 7 | 0 |
| B | best | 7 | 0 | 0.000 | 0 | 7 | 0 |
| B | final | 7 | 2 | 0.000 | 0 | 7 | 0 |

## 3. per-class AP50:95 差异 (Arm B - Arm A, best_model)

| 类 | GT | ArmA | ArmB | Δ |
|---|---|---|---|---|
| Tomato leaf bacterial spot | 16 | 0.163 | 0.284 | +0.120 |
| Tomato leaf late blight | 26 | 0.447 | 0.379 | -0.067 |
| grape leaf | 14 | 0.683 | 0.738 | +0.056 |
| Apple Scab Leaf | 30 | 0.528 | 0.484 | -0.045 |
| Tomato leaf mosaic virus | 33 | 0.192 | 0.236 | +0.044 |
| Apple leaf | 14 | 0.768 | 0.811 | +0.042 |
| Tomato Septoria leaf spot | 37 | 0.427 | 0.461 | +0.034 |
| Tomato leaf yellow virus | 57 | 0.161 | 0.185 | +0.024 |
| Apple rust leaf | 12 | 0.633 | 0.650 | +0.017 |
| Tomato Early blight leaf | 21 | 0.226 | 0.210 | -0.016 |
| Tomato leaf | 36 | 0.228 | 0.215 | -0.013 |
| Tomato mold leaf | 32 | 0.162 | 0.160 | -0.002 |
| grape leaf black rot | 22 | 0.405 | 0.403 | -0.002 |

## 4. 结论与建议
- **结论: EXIF 修复有效** (VAL mAP@0.5:0.95 提升 +0.0148 > 0.005) → **建议将 EXIF 修复作为后续实验的数据基础**。
- **TEST 仍封闭**: 未用任何 VAL 结果查看/选择 TEST; 是否上 TEST 由后续决策决定。

可视化: `ablation_perclass_ap.png`; 明细: `ablation_exif_perclass.csv`, `ablation_exif_img65.csv`