# BASELINE_DIAGNOSTIC_REPORT — PP-YOLOE+-s Baseline 深度诊断 (VAL 集)

- 日期: 2026-08-14
- 阶段: Baseline 深度诊断 (不改模型 / 不重训 / TEST 完全封闭, 仅 VAL 分析)
- 分析模型: `model_final.pdparams` (Baseline 训练产物之一; checkpoint 选择仅依据 VAL, 详见 BASELINE_SUMMARY.md §7)
- 评估集: VAL 113 张 / 350 目标 (训练时每 epoch 评估同源)
- 配套产物: `metrics/val_per_class_metrics.csv`, `metrics/val_confusion_analysis.csv`,
  `metrics/val_crop_summary.csv`, `metrics/val_bbox_size_dist.csv`, `metrics/val_image_errors.json`,
  `error_cases/` (9 张标注图), `diagnostic_visualizations/` (5 张图表)

---

## 0. 结论速览

1. **作物级差距是最大结构性信号**: Tomato mAP@0.5:0.95=**0.248** vs Apple=**0.646** vs Grape=**0.560** (VAL)。番茄 8 类占 VAL 74% 的目标, 却贡献了绝大部分错误。
2. **核心失败模式 = 同类内疾病混淆 + 漏检**, 不是定位能力不足, 更不是小目标问题 (VAL 仅 4 个小目标)。
3. **混淆高度集中在作物内部**: 组内混淆 89 次 vs 跨作物混淆仅 9 次 (conf=0.3); 最大混淆边 `Tomato bacterial spot→Septoria` 占其 GT 的 **31.2%**。
4. **存在可直接定位的数据缺陷**: 11 张 EXIF 旋转图 (如 img65) bbox 与像素错位, 单图可造成 7 GT 全漏 + 大量误检。
5. 基于真实错误案例提出 **3 个有数据证据的候选改进方向** (见 §6), 不做泛泛建议。

---

## 1. VAL 集 per-class 指标 (model_final)

P/R/F1 为 conf=0.5, IoU=0.5 操作点; AP 为 COCO metric。

| class | 目标 | 图片 | AP50 | AP50:95 | P | R | F1 | TP/FP/FN |
|---|---|---|---|---|---|---|---|---|
| Tomato Early blight leaf | 21 | 8 | 0.406 | 0.232 | 0.429 | 0.286 | 0.343 | 6/8/15 |
| Tomato Septoria leaf spot | 37 | 15 | 0.605 | 0.448 | 0.613 | 0.514 | 0.559 | 19/12/18 |
| Tomato leaf | 36 | 6 | 0.410 | 0.220 | 0.349 | 0.417 | 0.380 | 15/28/21 |
| Tomato leaf bacterial spot | 16 | 11 | 0.358 | 0.240 | 0.333 | 0.312 | 0.323 | 5/10/11 |
| Tomato leaf late blight | 26 | 12 | 0.657 | 0.429 | 0.682 | 0.577 | 0.625 | 15/7/11 |
| Tomato leaf mosaic virus | 33 | 6 | 0.211 | **0.151** | **0.154** | **0.061** | **0.087** | 2/11/31 |
| Tomato leaf yellow virus | 57 | 8 | 0.328 | **0.146** | 0.429 | **0.263** | 0.326 | 15/20/42 |
| Tomato mold leaf | 32 | 9 | 0.217 | **0.121** | 0.261 | 0.188 | 0.218 | 6/17/26 |
| Apple Scab Leaf | 30 | 9 | 0.773 | 0.541 | 0.600 | 0.500 | 0.545 | 15/10/15 |
| Apple leaf | 14 | 9 | 0.933 | 0.758 | 0.480 | 0.857 | 0.615 | 12/13/2 |
| Apple rust leaf | 12 | 9 | 0.779 | 0.640 | 0.600 | 0.750 | 0.667 | 9/6/3 |
| grape leaf | 14 | 6 | 0.831 | 0.674 | 0.846 | 0.786 | 0.815 | 11/2/3 |
| grape leaf black rot | 22 | 6 | 0.622 | 0.445 | 0.467 | 0.636 | 0.538 | 14/16/8 |

## 2. 按作物汇总 (VAL)

| 作物 | 类数 | 目标数 | 图片数 | mAP50 | mAP50:95 | mP | mR | mF1 | 小目标占比 |
|---|---|---|---|---|---|---|---|---|---|
| Tomato | 8 | 258 (73.7%) | 74 | 0.399 | **0.248** | 0.406 | **0.327** | **0.358** | 1.6% |
| Apple | 3 | 56 | 27 | 0.828 | 0.646 | 0.560 | 0.702 | 0.609 | 0% |
| Grape | 2 | 36 | 12 | 0.726 | 0.560 | 0.656 | 0.711 | 0.676 | 0% |

> 番茄 8 类共享相似叶片底色/斑点/黄化症状, 类间可分性差 → 整体最弱。Apple/Grape 类数少、症状形态差异大 → 显著更强。

## 3. bbox 尺寸分布与类别不平衡

### 3.1 尺寸分布 (VAL GT, 面积按 COCO 阈值)

| 档位 | 数量 | 占比 |
|---|---|---|
| small (<32² px) | 4 | 1.1% |
| medium (32²–96²) | 92 | 26.3% |
| large (≥96²) | 254 | 72.6% |

- 仅 mosaic(1) 与 yellow virus(3) 含小目标。**小目标不是本数据集的性能瓶颈** (VAL AP-small 0.055 只是 4 个样本的统计噪声)。

### 3.2 类别不平衡 (VAL; 训练集目标数见括号, 来自 label_list col5)

| 失衡表现 | 数据 |
|---|---|
| 训练集极端不平衡 | Tomato yellow virus 824 / Septoria 430 vs grape black rot 133 / Apple Scab 171 |
| 但 VAL 目标数相对均衡 | 12–57 之间, 无一类 <12 |

→ 训练侧不平衡显著, 但 VAL 弱类并非"目标最少"类 (yellow virus 57 个最多却最弱), 说明**弱因是视觉可分性而非单纯样本数**。

## 4. 混淆分析 (VAL, conf=0.5 / 0.3)

### 4.1 主要混淆边 (count≥3)

| GT → Pred | 次数 | 占 GT 比例 |
|---|---|---|
| Tomato bacterial spot → Septoria leaf spot | 5 | **31.2%** |
| Tomato mosaic virus → yellow virus | 5 | 15.2% |
| Apple Scab → Apple rust | 4 | 13.3% |
| Tomato Early blight → bacterial spot | 3 | 14.3% |
| Tomato mold → Early blight | 3 | 9.4% |
| Tomato mold → mosaic | 3 | 9.4% |

### 4.2 结构结论

- **组内混淆 89 次 vs 跨作物混淆仅 9 次** (conf=0.3): 模型在作物层面几乎完美 (无番茄↔苹果↔葡萄混淆), 错误全部发生在一作物内的疾病之间。
- 最大混淆边 `bacterial spot→Septoria` (31.2%): 两者均表现为叶面小褐色圆斑, 形态高度相似。
- `mosaic→yellow` (15.2%): 两种病毒病均导致叶片黄化/斑驳。
- `Scab→rust` (13.3%): 苹果病害中均含褐/橙色斑。
- 详细矩阵见 `diagnostic_visualizations/confusion_heatmap_val.png` 与 `metrics/val_confusion_analysis.csv`。

## 5. 错误案例分析 (error_cases/, 附真实图)

| 案例 | 图像 | 错误类型 | 证据 |
|---|---|---|---|
| `case_missed_dense_yellowvirus_img92.jpg` | TRAIN_000085_DSC01999.JPG | 漏检 (密集同类) | 19 GT 仅检出 1, 18 漏检; 1024×768 密集黄化叶片 |
| `case_missed_dense_mosaic_img7.jpg` | TEST_000079_9511.img.jpg | 漏检 (密集 mosaic) | 12 GT 漏 10, 小尺寸 540×400 |
| `case_missed_smallimg_mosaic_img88.jpg` | TRAIN_000080_... | 漏检 (极小图) | 205×216, 7 GT 全漏; 上采样后特征不足 |
| `case_fp_leaf_dense_img64.jpg` | TRAIN_000053_... | 误检 (过度检测) | 8 GT 却产生 22 检测 (17 FP), 密集叶场景框泛滥 |
| `case_fp_blackrot_overseg_img114.jpg` | TRAIN_000113_00gb.jpg | 误检 (过度分割) | grape black rot 7 GT / 16 检测, 10 FP 同类; 单个病灶簇被多框 |
| `case_conf_leaf_vs_blight_mold_img63.jpg` | TRAIN_000052_... | 类别混淆 | Tomato leaf GT 被判 late blight(2×, 高conf 0.70/0.59)与 mold(2×) |
| `case_conf_bactspot_vs_septoria_img5.jpg` | ... | 类别混淆 | bacterial spot→Septoria, conf 0.92 |
| `case_conf_bactspot_vs_septoria_img67.jpg` | ... | 类别混淆 | 同上, 第二例 |
| `case_exif_rotation_img65.jpg` | TRAIN_000054_happier-inside.jpg | 数据缺陷 | 标注 3888×2592, 实际像素 2592×3888 (EXIF 旋转未烘焙); 7 GT 全漏 + 大量误检 |

### 5.1 错误构成分解 (VAL, conf=0.5)

| 分量 | 数值 |
|---|---|
| GT 被任一类框定位 (类无关 loc recall) | 203/350 = 58.0% |
| 定位正确且类标签正确 (per-class recall) | 144/350 = 41.1% |
| **定位对但类错 (分类瓶颈)** | 59 (占定位成功的 29%) |
| 检测总数 / 背景误检 / 类错 / 类对 | 304 / 101 (33.2%) / 59 (19.4%) / 144 (47.4%) |

按作物分类瓶颈占比 (定位对但类错 ÷ GT):
- Tomato: loc=0.519, cls=0.322 → **瓶颈 0.198**
- Apple: loc=0.768, cls=0.643 → 瓶颈 0.125
- Grape: loc=0.722, cls=0.694 → 瓶颈 0.028

→ 番茄的错误是"漏" (48%) 与"类错" (20%) 并存; 苹果/葡萄主要是漏 (低 recall), 类错很少。

## 6. 为什么弱类弱 / 为什么 grape black rot 较强 (数据证据)

### 6.1 弱类成因 (mosaic 0.151 / yellow 0.146 / mold 0.121 / bacterial spot 0.240 / early blight 0.232)

1. **视觉可分性差 (主因)**: 番茄 8 类共享同一叶片底色, 病害仅表现为斑点/斑驳/黄化/霉层的细微差异。混淆矩阵证明: bacterial spot↔Septoria 31.2%, mosaic↔yellow 15.2%, mold↔early blight 9.4% — 全是"看起来像另一种病"。
2. **类定义歧义**: "Tomato leaf" (健康叶) 与各病害叶视觉重叠 → 健康叶被判为具体病害 (img63: leaf GT→late blight), 具体病害又被压回 leaf (mosaic→leaf 2×)。leaf 类本身 AP=0.220 且 FP=28 为全表最高之一。
3. **漏检而非定位**: mosaic/yellow/mold 的 R 仅 0.06–0.28。密集同场景 (img92) 与极小图 (img88) 下模型干脆不出框 (conf=0.5 时 42% 番茄 GT 无任何框覆盖)。
4. **训练侧不平衡放大**: yellow virus 训练目标 824 (7× 于 black rot 的 133), 但形态单一; 模型对多形态的 yellow 特征拟合不足, 表现为低 recall 而非低 precision。

### 6.2 grape black rot 为什么 AP 较高 (0.445, 强于番茄所有弱类)

- 葡萄仅 2 类, 且黑腐病斑 = 黑褐色同心圆环斑, 与葡萄健康叶及番茄/苹果症状**形态差异大**, 特征可分。
- loc 强 (0.722)、类错极少 (瓶颈 0.028): 定位对且类对的比例高。
- 但其 F1 仍受 **过度分割** 拖累 (img114: 7 GT→16 框, 10 同类 FP), P=0.467。说明病斑密集处 NMS 抑制不足 → 多框重复。

## 7. 候选算法创新方向 (数据支撑, 非泛泛建议)

### 方向 1 — 数据管线修正: EXIF 旋转对齐 + 标注几何校验
- **针对问题**: 11 张 EXIF 图 bbox 与像素错位 (img65 实证: 7 GT 全漏 + 大面积误检), 是"单图多错误"的集中来源。
- **数据证据**: img65 标注 3888×2592 vs 实际 2592×3888; 该批图像错误密度远高于正常图。
- **方案**: 预处理按 EXIF Orientation 旋转像素并**同步变换 bbox** 至统一朝向; 之后用 cv2 尺寸 vs 标注尺寸自动校验全部图像几何。
- **预期改善**: 消除 0.96% 图像的灾难性错误; 涉及类以番茄为主, 直接小幅提升 tomato recall。属于确定性的负熵修复。
- **难度**: ★☆☆☆☆ (数据管线, 不改模型结构, Paddle 数据算子即可)
- **AIC2026 适用性**: ★★★★★ (纯正确性修复, 与任何后续创新叠加, 零风险)

### 方向 2 — 组内判别增强: 针对同作物疾病混淆的分类头/损失 (方向性最强)
- **针对问题**: 组内混淆 89 次 vs 跨作物 9 次; 番茄分类瓶颈 19.8pp; 最大混淆边 bacterial spot↔Septoria (31.2%)、mosaic↔yellow (15.2%)。
- **数据证据**: 混淆矩阵 + 错误分解共同指向"定位后标签错"——模型已把叶子框对 (作物层面零混淆) 却分不清相似病害。
- **方案** (选一或组合, 均在 PaddleDetection 生态内):
  - 检测头分类子空间按作物分组 (crop-conditioned heads / 每组独立 classifier), 直接复用"跨作物已可分"这一先验;
  - 分类损失引入**难分对判别约束** (AM-Softmax/ArcFace 式 margin, 或对 confusion pair 加原型间距), 专门拉大 bacterial spot↔Septoria、mosaic↔yellow 的特征间距;
  - 可选**两阶段解耦**: 类无关定位器 + 独立强分类器 (减轻 33% 背景误检对分类共享头的干扰)。
- **预期改善**: 直接降低最大混淆边 → 提升 bacterial spot / mosaic / early blight / mold 的 per-class AP 与 Precision; 对整体 mAP@0.5:0.95 影响显著 (番茄占 74% 目标)。
- **难度**: ★★★☆☆ (改损失/头; 若做两阶段为 ★★★★☆)
- **AIC2026 适用性**: ★★★★★ (赛道正是作物病害分类+检测, 类间相似是核心难点)

### 方向 3 — 低召回与密集场景: 类平衡训练 + 密度感知后处理
- **针对问题**: Tomato 漏检 48% (conf=0.5), mosaic R=0.061 / yellow R=0.263 / mold R=0.188; 密集同类场景 recall 崩塌 (img92: 19→18 漏); 黑腐过度分割 (img114: 10 同类 FP)。
- **数据证据**: per-class FN 统计 (mosaic 31 / yellow 42 / mold 26 / Septoria 18) + img92/img114 两例对比 (同类密集→漏检 vs 病斑密集→多框)。
- **方案** (组合):
  - 训练: 类加权 / repeat factor sampling, 提高 low-recall 类在 loss 中的占比 (yellow/mosaic/mold); 或 focal 式损失抑制背景误检 (33% 检测为背景);
  - 推理: soft-NMS / 多尺度 TTA 缓解"密集同类互相抑制导致漏检"与"病灶多框重复", 二者在同一 NMS 问题上呈现两极端。
- **预期改善**: 弱类 Recall 与 per-class AP 提升; 密集场景漏检与重复框同步下降 → mAP@0.5:0.95 直接收益。
- **难度**: ★★☆☆☆ (损失+后处理, 无 backbone/head 结构改动)
- **AIC2026 适用性**: ★★★★☆ (小样本+类不平衡赛道, 低成本高回报)

### 方向对比 (一句话)
方向 1 是**必做的数据负熵修复** (低成本确定性收益); 方向 2 直击**赛道最核心的疾病类间混淆** (最大提升空间); 方向 3 解决**召回与密集场景** (最贴近工程落地, 改动小)。

> 建议实施顺序: 1 → 3 → 2。三个方向互不冲突, 可在同一 Baseline 上分别做消融, 用 VAL 验证后再决定是否上 TEST。

---

## 8. 附: 交付物清单

```
experiments/baseline_v1/
├── BASELINE_DIAGNOSTIC_REPORT.md      ← 本文件
├── DATASET_CONSISTENCY_REPORT.md      数据一致性核对 (916→915 说明等)
├── metrics/
│   ├── val_per_class_metrics.csv       每类 AP50/AP50:95/P/R/F1/TP/FP/FN/尺寸
│   ├── val_crop_summary.csv            按作物汇总
│   ├── val_confusion_analysis.csv      混淆边 (gt→pred, count, 占GT比例)
│   ├── val_bbox_size_dist.csv          small/medium/large 分布
│   └── val_image_errors.json           逐图错误分解 (供复现筛选)
├── error_cases/                        9 张标注错误案例图
└── diagnostic_visualizations/          per-class AP / 混淆热图 / 尺寸分布 / 作物对比 / 错误构成
```

注: TEST 集在本阶段**完全未触碰**; 所有指标仅基于 VAL。未修改任何数据、未修改模型、未开始任何训练。
