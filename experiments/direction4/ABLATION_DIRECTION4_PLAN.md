# 方向4 消融计划 — 密集感知置信度校准

- **日期**: 2026-08-15
- **状态**: 待批准后执行;当前阶段不训练、不修改正式模型、不碰 TEST
- **对照基准**: A0(方向2 EXIF-Fix best checkpoint),使用同一评估脚本重算
- **数据**: TRAIN(参数锁定/机制验证)→ VAL(唯一最终评估)→ TEST(完全封闭)
- **原始数据集**: 绝不修改;沿用 EXIF-Fix 副本

---

## 1. 目标与共享判据

**目标**:验证"密集场景分类置信度校准"能否提升 VAL mAP@0.5:0.95,并确认不损伤强类。

**共享成功判据**(主判据+副判据+负向红线,沿用方向3纪律):

| 判据 | 要求 |
|---|---|
| 主判据 | 相对 A0 best 的 VAL mAP@0.5:0.95 ≥ **+0.005** |
| 副判据 | 密集桶(≥5 目标/图)Recall 提升 且 总体 F1 不降 |
| 负向红线 | 无强类回退(ΔAP < −0.02 告警) |

**失败即记录,不调参重试。**

---

## 2. 实验顺序与门控

```
Step 0  ─ 基线复核: A0 best 在 VAL 上的 mAP@0.5:0.95 (用方向4同一脚本, 确认≈0.4012)
Step 1  ─ D2 推理侧消融 (零训练, 当天出结果)  → 廉价信号
Step 2  ─ D1 训练侧消融 (100 epoch, 主路径)
Step 3  ─ 汇总对比 + 报告 (D1 判据为主; D2 作为补充数据)
(D3 TTA 仅当 Step 2 后仍有余量且用户要求时才考虑)
```

---

## 3. Step 1 — D2 推理侧消融(无训练)

**目的**:量化"推理侧保守重标"能否在零训练成本下捕获 oracle·重标61(+0.019)的一部分;同时实证"推理侧天花板",为 D1 决策提供依据。

**唯一变量**:对 A0 推理候选框(score≥0.1)施加后处理算子;其余(模型/阈值/NMS 上游)不变。

**流程**:
1. **TRAIN 推理**:用 A0 best 对 TRAIN 集推理,得到 TRAIN 预测候选框(`dataset/processed_detection_exiffix` 的 train 标注)。
2. **TRAIN 参数锁定(网格,仅在 TRAIN)**:
   - 参数域:α∈{0.5,0.6,0.7,0.8}、straggler∈{0.2,0.3,0.5}、τ∈{0.15,0.2,0.25}(若启用重打分)。
   - 选择规则:**只接受在 TRAIN 上 mAP 不降且 Recall 提升的组合**;取 TRAIN mAP 最高的一个锁定。全程用 TRAIN,VAL 不参与。
3. **锁前机制验证**:
   - 重标仅触碰 0.3–0.5 带内、类≠图像主导类、原类弱支持的框;统计被改框数与"改后变 TP/FP"的 TRAIN 计数。
   - 若重打分(TRAIN 验证下)TP/FP 区分度不达标,按设计规则**弃用重打分,仅保留重标**。
4. **VAL 单次评估**:用锁定参数对 VAL 推理结果后处理,同一 eval 脚本输出 mAP@0.5:0.95 / mAP@0.5 / P/R/F1@conf0.5 / 13 类 AP / 密集桶 Recall。
5. **记录**:如实报告;若 Δ≈0 或为负,记录为"D2 未达判据,推理侧天花板确认"。

**产物**:`experiments/ablation_dir4_D2/`(参数表、TRAIN 锁定日志、VAL 评估、对比 CSV/PNG)。

---

## 4. Step 2 — D1 训练侧消融(主路径)

**唯一变量**:PPYOLOEHead 的 VFL 分类损失开启"密集感知 gt_score 地板"(`use_dense_calib=True`,`dense_thr=5`,`score_floor=0.65`);其余与 A0 逐字段一致。

| 项 | A0(对照) | D1(实验) |
|---|---|---|
| 分类损失 | 原始 VFL | VFL,密集图正锚 gt_score' = max(IoU, 0.65) |
| 模型/检测头 | PP-YOLOE+-s | 同(结构零改动) |
| 回归损失(iou/dfl) | 默认 | 同 |
| 数据增强/采样/NMS/推理阈值 | 官方默认 | 同 |
| epoch / bs / lr / seed | 100 / 16 / 0.002 / 0 | 同 |
| 权重 | — | 同 A0 初始化(Objects365 预训练,seed=0) |

**流程**:
1. **实现检查**(同 verify_b1 纪律):`use_dense_calib=False` 时与原始 VFL 逐位一致;开启后仅密集图正锚目标分被地板(逐图密度计算正确、不影响稀疏图);gt_score 进入梯度;TRAIN 图密度统计与标注一致。4 项全 PASS 才继续。
2. **2-epoch Smoke Test**:forward/backward/loss/checkpoint/VAL 全正常,0 ERROR/0 NaN;打印密集图 gt_score 地板生效样例。
3. **正式 100 epoch 训练**:`seed=0`(`--enable_ce True`)、全部配置同 A0;保存完整命令/配置/env/git commit/train.log/checkpoint。
4. **VAL 评估**:同一 eval 脚本重算 A0 与 D1(best+final),输出 mAP@0.5:0.95 / mAP@0.5 / P/R/F1@conf0.5 / 13 类 AP+Recall / 强类回退表 / **密集桶(≥5)Recall 对比**(该桶是 D1 的作用面)。
5. **判据判定**:主/副/红线三表,结论如实(达/未达)。
6. **机制核对**:对比 A0 vs D1 在 0.3–0.5 带候选分布(是否整体上移)、密集桶 TP 分数分布;确认提升来源于密集置信度而非其它。

**D1b(可选独立实验,不与 D1 主判定混跑)**:在 D1 基础上叠加"拥挤背景负项降权"(β=0.5)。单独训练/评估,单变量=D1 基础上再加这一项;作为 D1 未达判据时的次选,或 D1 达标后是否叠加的附加数据。

**产物**:`experiments/ablation_dir4_D1/`(mechanism_check.txt、smoke 日志、config/env/git/命令、train.log、checkpoints、val_eval、D1_vs_A0_metrics.csv、per_class_D1_comparison.csv、dense_bucket_comparison.csv、PNG、DIRECTION4_D1_REPORT.md)。

---

## 5. Step 3 — 汇总报告

- **DIRECTION4_D1_REPORT.md**:D1 单变量对照、判据三表、机制核对、诚实结论(与方向3 B1 的失败教训并列讨论:D1 与 B1 机制差异)。
- **汇总表**:A0 / B1(方向3,已失败)/ D2 / D1 的 VAL 指标横排对比。
- 若 D1 未达判据 → 记录并停止,不调参重训;TEST 继续封闭。

---

## 6. 风险与门控

| 风险 | 缓解 |
|---|---|
| D1 地板把坏框推高伤 precision | smoke + 训练期 P/R 监控;地板 0.65 远低于实测框 IoU 中位 0.78 |
| D1 触发强类回退 | 判据红线监控;回退即记录失败 |
| D2 算子误伤正确检测 | TRAIN 锁定阶段把关,不达标则弃用对应算子 |
| VAL 被多次使用(污染) | VAL 仅用于 D1/D2 各一次最终评估;参数全部在 TRAIN 锁定 |
| 推理侧天花板(诊断已证) | D2 预期低,如实报告,主路径坚定 D1 |

## 7. 约束重申

- TEST 全程不参与训练/调参/参数锁定,仅最终评估(本计划不含 TEST)。
- 原始数据集与 `processed_detection` 原文件不修改。
- 每步失败立即停止并如实报告。
