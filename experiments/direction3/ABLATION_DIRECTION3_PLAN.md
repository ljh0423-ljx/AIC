# 方向3 消融实验计划 — A0 → B1 单变量对照 (Class-Weighted VFL)

- 日期: 2026-08-15
- 状态: **计划定稿, 尚未开始训练** (按用户指示, 设计完成后停止并汇报)
- 设计文档: `DIRECTION3_DESIGN.md` (公式/实现/权重/监控)
- 诊断文档: `DIRECTION3_DIAGNOSTIC.md`

---

## 1. 实验定义

| | A0 (对照) | B1 (实验) |
|---|---|---|
| 数据 | exiffix (`processed_detection_exiffix`) | 同 |
| 分类损失 | 原始 VFL | VFL × class weight (w_c, 均值 1.0) |
| 模型 | PP-YOLOE+-s | 同 |
| epoch / bs / lr | 100 / 16 / 0.002 | 同 |
| seed | 0 (`--enable_ce True`) | 同 |
| 预训练 / 增强 | Objects365 / 官方默认 | 同 |
| Head 开关 | `use_cls_weight: False` (默认) | `use_cls_weight: True` |
| 权重文件 | — | `class_weight_train.json` (TRAIN-only, 冻结) |

**单变量性**: A0 与 B1 唯一差异 = 训练分类损失中的类权重乘法。A0 **不重训**, 复用方向2 A0 (exiffix) best checkpoint; 推理路径两臂完全一致 (权重仅训练期用)。

## 2. 目录与脚本

```
experiments/ablation_dir3_B1/
├── b1/                  # B1 训练输出 (save_dir), best_model / model_final
├── val_eval/A0_best/    # A0 重评估输出 (与 B1 同脚本, 复算保证无脚本偏差)
├── val_eval/B1_best/    # B1 best 评估输出
├── val_eval/B1_final/   # B1 model_final 评估输出
├── metrics.csv          # 两臂总体指标
├── per_class.csv        # 13 类 AP + Recall 对比
├── ablation_report.md   # 结果报告 (成功/失败如实记录)
└── *.png                # 可视化
```

- 配置文件: `configs/ablation_dir3_B1.yml` (新建, 见设计文档 §3.4)
- 权重文件: `experiments/direction3/class_weight_train.json` (已生成, 冻结)
- 训练脚本: `scripts/train_dir3_b1.sh` (新建, 仿 train_dir2_ablation.sh, 用 B1 配置 + seed=0)
- 评估脚本: `scripts/eval_dir3_b1.sh` (新建, 复用 eval_dir2_ablation.sh 的 VAL 评估模板, 含 `--classwise`)

## 3. 执行步骤

| 步骤 | 动作 | 产出 |
|---|---|---|
| 1 | `git` 无关代码变更: 实现 §3.2/§3.3 的 Head 修改 (flag 默认关闭) | 代码就绪, A0 行为不变 |
| 2 | 冒烟测试: 2 epoch 极小训练 (`configs/ppyoloe_plus_crn_s_2e_smoke.yml` 思路), 验证 `use_cls_weight=True` 下 forward/backward/loss 正常 | 冒烟通过记录 |
| 3 | 训练 B1 (100 epoch, seed=0) | `experiments/ablation_dir3_B1/b1/{best_model,model_final}` |
| 4 | 用**同一评估脚本**评估 A0_best / B1_best / B1_final | `val_eval/` 原始输出 |
| 5 | 指标对比 + 逐类 Recall 对比 + 强类回退检查 (设计 §6.1) | `metrics.csv` / `per_class.csv` |
| 6 | 写报告, 按 §4 判据判定 | `ablation_report.md` |

## 4. 成功判据 (同时满足)

| # | 判据 | 要求 |
|---|---|---|
| 1 (主) | VAL mAP@0.5:0.95 (best) | **B1 − A0 ≥ +0.005** |
| 2 (主) | 总体 Recall 或 F1 @conf0.5 | **至少一项 > A0** |
| 3 (副) | 弱势类 Recall | 被点名漏检类不降, 或 TRAIN 稀少类任一 Recall 提升 |
| 4 (红线) | 强类回退 | 无 A0 强类 ΔAP < −0.02 (若有, 报告中拆解是否吞没收益) |

判定口径: **全部用 best_model**; 评估用同一脚本、同一数据、同一 IoU/conf 协议; A0 数值由本次重评估给出, 不以方向2 报告直接引用 (防脚本偏差)。

## 5. 失败处理协议 (诚实报告, 不调参)

- 若 B1 未过主判据 (mAP Δ < +0.005 或负值): **接受失败**, 在 `ablation_report.md` 如实记录; 不进行任何权重缩放/学习率/epoch 调整后重试。
- 若 B1 通过主判据但触发强类回退红线: 报告收益与代价的拆解, 由用户决定是否视为有效。
- **B2 (density-aware inference) 仅在 B1 判定有效后才进入设计**; 无效则方向3 收尾, B2 不启动。
- TEST: 全程不接触。正式模型 (非 flag) 行为不变。

## 6. 风险与缓解

| 风险 | 说明 | 缓解 |
|---|---|---|
| yellow w=0.27 致其 AP/Recall 回落 | 设计先验已接受 (诊断证明其漏检非不平衡所致) | 判据 3 允许"点名类不降或稀少类提升"; 主判据仍为硬门槛 |
| 上调类过拟合 (black rot 1.87 / Scab 1.51) | 少样本大权重放大噪声 | 每 epoch VAL 轨迹 + per-class 轨迹监控; best vs final 对比 |
| loss 量级漂移 | w 均值 1.0, 期望近似不变 | 训练日志监控 loss_cls 量级 |
| 复算 A0 与方向2 数值不一致 | 评估协议/脚本差异 | 同脚本复算, 以复算值为准并记录偏差 |

## 7. 交付物清单

- `configs/ablation_dir3_B1.yml`
- `scripts/train_dir3_b1.sh`, `scripts/eval_dir3_b1.sh`
- Head 修改 (flag 默认关闭, A0 不重训不受影响)
- `experiments/ablation_dir3_B1/` 全部产物 + `ablation_report.md`
- 三份设计/诊断/计划文档 (方向3 目录)

**当前状态: 本计划已完成, 未执行任何训练/评估。等待用户批准后开始步骤 1-2。**
