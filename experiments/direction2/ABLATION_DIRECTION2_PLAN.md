# ABLATION_DIRECTION2_PLAN — 组内判别增强消融实验计划

- 日期: 2026-08-15 | **计划阶段: 未开始训练, TEST 全程封闭**
- 数据基础: **Arm B (exiffix)** (方向1 结论"有效", 已过 validate_exiffix ALL PASS)
- 基线: Arm B `best_model` (VAL mAP@0.5:0.95=0.401) 与 `model_final` (0.386)
- 唯一变量原则: 每个方案对照只新增"该方案组件", 其余 (数据=exiffix / seed=0 / bs=16 / lr=0.002 / epoch=100 / 增强 / assigner / 评估=仅VAL) 逐字段与 Arm B 一致。

---

## 0. 阶段门槛 (Gates)

| 门 | 条件 | 不满足则 |
|---|---|---|
| G0 | Arm B 数据 (exiffix) 校验报告仍 ALL PASS | 先重跑 validate_exiffix.py |
| G1 | CGPM 的混淆查表来自 **train 集** (不得用 VAL 混淆调 margin) | 先对 Arm B 在 train 集评估导出 |
| G2 | 每臂训练日志完整 + checkpoint + env + git commit + config 存档 | 补齐后再评估 |
| G3 | 只用 VAL 评估; 不因任何结果查看/选择 TEST | 违反即停止 |

---

## 1. 数据与共享产物

- 数据: `dataset/processed_detection_exiffix` (软链到 `PaddleDetection/dataset/processed_detection`)。
- 混淆查表 (CGPM 输入, 冻结一次): `experiments/direction2/confusion_train_baseline.json` —— 由 Arm B 在 train 集评估得到 13×13 混淆矩阵, 提取 `R(g)` (top-K 混淆对手) 与 margin 权重 (混淆率/最大混淆率归一化)。
- 训练脚本: 复刻 `scripts/train_ablation.sh` (DATA_ARM 固定 exiffix), 输出到 `experiments/ablation_dir2_<tag>/`。
- 评估/对比脚本: 复用 `scripts/eval_val_ablation.sh` + `scripts/val_ablation_compare.py` (或扩展 per-pair recall 输出)。

## 2. 方案 A: CGPM (主方案) 消融表

| 臂 | 说明 | 目的 |
|---|---|---|
| A0 | Arm B (已有, 不重训) | 基线 |
| **A1** | Arm B + CGPM (混淆引导 margin, λ=0.25, s=16, margin_base=0.15, top_k=2) | 主效果 |
| A2 | Arm B + 固定 margin (m 常数, 不按混淆加权) | 创新点对照: "混淆引导" vs "固定 margin" |
| A3-λ | A1 最优超参灵敏度 λ∈{0.1, 0.5} | 稳健性 |
| A3-s | s∈{8, 32} | 稳健性 |

**流程**: G1 先导出 train 混淆查表 → 训练 A1 (seed=0, bs=16, lr=0.002, epoch=100, 仅 VAL 评估) → VAL eval (best+final) → 与 A0 对比。A2/A3 同构。

## 3. 方案 B: CCDH (备用) 消融表

| 臂 | 说明 | 目的 |
|---|---|---|
| B0 | Arm B (已有) | 基线 |
| B1 | Arm B + CCDH 两级头 (crop 门控 + 作物内细粒度) | 主效果 |
| B2 | B1 去掉 crop_loss (仅结构解耦, 无作物监督) | 分离结构/监督贡献 |

**流程**: 仅当 A1-A0 < 0.005 或需要第二独立证据时执行。改动限于 head 末层, 其余不变。

## 4. 评估指标与成功判据

**每臂 (best+final) 全量指标** (复用 val_ablation_compare.py 口径):
- mAP@0.5:0.95, mAP@0.5, AR, P/R/F1 (类感知, conf=0.5/IoU=0.5), TP/FP/FN。
- 13 类 AP50/AP50:95 + per-class P/R/F1。
- 类无关 loc_ok / cls_ok / **loc_but_wrong_cls (41→?)**; 高置信类错 (9→?); 低置信正确 (38→?)。
- **命名混淆对 per-pair 指标**: bact↔Septoria、mosaic↔yellow、EB↔bact、mold↔EB (及 Apple Scab↔rust) 的错分次数与得分分布重叠 (0.889→?)。

**成功判据 (预注册, 避免事后挑选):**
1. 主判据: VAL mAP@0.5:0.95 较 A0 提升 **> 0.005**。
2. 次判据: 命名混淆对 per-class AP 多数 (≥5/7) 提升, 且无任何类 AP 下降 > 0.02 (或下降类数 ≤2)。
3. 计算量: 推理 FPS 下降 ≤2% (A) / ≤3% (B); 参数量增量 ≤3%。
4. 若主判据不满足: 如实记录"方向2 分类边界非主瓶颈", 不强行套方案, 转向方向3 (召回/密度) 或方向1 数据复查。

## 5. 产出物 (每个通过 G2 的臂)

```
experiments/ablation_dir2_<tag>/
├── train.log / environment.txt / git_commit.txt / config.yml / train_command.txt
├── checkpoints/{best_model,model_final}.pdparams
├── metrics/val_eval_{best,final}.json/bbox.json   (VAL 评估)
└── metrics/val_eval_{best,final}.log
```
汇总: `experiments/direction2/ABLATION_DIRECTION2_SUMMARY.md` (A1-A0 对比表 + 命名对 per-pair + 结论与推荐)。

## 6. 安全边界 (硬约束)
- **TEST 全程封闭**: 不训练/不调参/不查看/不选择, 仅用于最终独立评估 (待用户批准)。
- **原始数据不修改**: 只在 exiffix 副本上训练; 无任何数据写入。
- **正式模型不修改**: 所有改动在 PaddleDetection 副本 (非正式 baseline 权重), 通过 git 追踪。
- **失败立即停止并报告**: 环境/安装/校验失败或指标异常, 不强行修复。

## 7. 预计资源与时间
- 每臂: 100 epoch 与 Arm B 相同 (数据加载受限, ~每 epoch 略快于 1.5 小时 → 约 3–4 天/臂)。A1 为必跑; A2/A3 视 A1 结果与资源决定; B1/B2 仅备用。
- 若单卡串行, 优先跑 A1; A2 是证明创新点的关键对照, 建议保留。

> 本计划未执行任何训练; 待用户批准后按 G0→G1→A1 顺序开始。
