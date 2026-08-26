# ABLATION_PLAN — Ablation-Data 实验: Baseline 清洁数据 vs EXIF 修复数据

- 日期: 2026-08-14
- 状态: **方案设计完成, 未启动任何训练**
- 关联: 方向1 数据质量修复 (EXIF_FIX_PLAN.md), Baseline 诊断 (BASELINE_DIAGNOSTIC_REPORT.md §7 方向1)
- 原则: 唯一自变量 = **数据修复因子**; 模型/配置/种子/epoch/bs/lr 全同; VAL 用于对比; TEST 全程封闭。

---

## 1. 目标与假设

**目标**: 用严格对照实验量化"EXIF 旋转烘焙修复"对检测性能的净影响, 决定是否将修复纳入正式数据管线。

**假设 H**: 修复后 VAL mAP@0.5:0.95 ≥ 未修复, 且受影响类别 (Tomato 类 1/2/3/4, Apple rust 11) 的 recall 提升。
理论依据 (EXIF_FIX_PLAN.md §2): 11 张 EXIF 图的 bbox 与像素错位, img65 (val TRAIN_000054) 实证 7 GT 全漏 + 大量误检 — 属于确定性的数据负熵, 修复为纯收益。

## 2. 实验设计 (两臂对照)

| | Arm A (对照, 复现 Baseline 数据) | Arm B (实验, EXIF 修复) |
|---|---|---|
| 数据源 | `dataset/processed_detection_clean` | `dataset/processed_detection_exiffix` |
| 与 Baseline 的差异 | 无 (同 clean) | 仅 10 张 train/val 图像烘焙 (EXIF_FIX_PLAN.md §3); **test 分片零改动** |
| 唯一自变量 | — | 数据修复因子 |
| 训练配置 | 与 Arm A **逐字段一致** (见 §4) | 同 |
| 随机种子 | 固定 (见 §5) | **同一种子** |
| epoch / bs / lr | 100 / 16 / 0.002 | 100 / 16 / 0.002 |

> **关键决策**: 两臂**重新训练** (不直接复用已完成的 Baseline 权重/指标)。
> 原因: Baseline 训练未固定随机种子 (见 §5), 若只重训 Arm B 而拿历史 Baseline 当 Arm A, 随机轨迹不同, 无法把差异干净归因于数据修复。重训两臂 (≈2×2.9h) 才能满足"唯一变量"的严格对照。历史 Baseline (VAL 0.408) 仅作参考坐标。

## 3. 数据副本构建 (设计)

1. 依据 EXIF_FIX_PLAN.md §3.1–3.2: `cp -al` 建 `processed_detection_exiffix`, 烘焙 10 张 (train 9 + val 1), test 不动。
2. **JSON / YOLO txt / label_list / dataset.yaml 与 clean 逐字节一致** (EXIF 修复不改标注)。
3. 入训前强制跑 `scripts/validate_exiffix.py` (EXIF_FIX_PLAN.md §4) 全部 PASS, 否则禁止 Arm B 训练。
4. 数据软链约定: 保持配置文件 `dataset_dir: dataset/processed_detection` 不变, 两臂运行时分别把 `PaddleDetection/dataset/processed_detection` 指向对应副本 (`clean` / `exiffix`)。软链在每臂启动前重建, 保证训练读取的就是该臂数据。

## 4. 模型与训练配置 (逐字段一致)

沿用 Baseline 配置文件 `configs/ppyoloe_plus_crn_s_100e_agrivision.yml` **原样** (不新增任何改动):

| 项 | 值 |
|---|---|
| 模型 | PP-YOLOE+-s (depth_mult=0.33, width_mult=0.50), 官方结构, 零结构/损失/增强改动 |
| epoch | 100 |
| batch_size | 16 (train) / 2 (eval) |
| base_lr | 0.002 (=0.001×16/8 线性缩放) |
| 调度 | LinearWarmup 5 epoch → CosineDecay max_epochs=100 |
| 预训练 | Objects365 骨干 (ppyoloe_crn_s_obj365_pretrained.pdparams, 官方默认) |
| 评估 | `--eval` 每 epoch val; snapshot_epoch=5 |
| 其他 | log_iter=20, use_gpu=true, 与 Baseline 完全一致 |

仅 `-o` 覆盖项与 Baseline 相同: `save_dir` / `TrainReader.batch_size=16` / `LearningRate.base_lr=0.002`, 外加 §5 的种子参数。

## 5. 随机种子方案 (已核实 PaddleDetection v2.9.0 实现)

- 已确认: `tools/train.py` L130–131 仅在 `FLAGS.enable_ce` 为真时调用 `set_random_seed(0)` (ppdet/engine/env.py: `paddle.seed(0) + random.seed(0) + np.random.seed(0)`)。config 中无 seed 字段。
- **Baseline 训练命令未传 `--enable_ce` → 未固定种子** (这正是 §2 要求重训两臂的依据)。
- **设计**: 两臂训练命令**都追加 `--enable_ce`**, 令两臂均以 seed=0 起训。
  - `enable_ce` 在 train.py 中仅有种子这一副作用 (已 grep 确认), 不改变训练逻辑, 亦不修改 PaddleDetection 代码。
  - 由于 Arm A/B 的 COCO JSON 与文件清单**逐字节相同**, 相同种子 → 相同采样顺序与数据增强序列; 两臂唯一差异 = 10 张修复图的像素内容。→ 差异可干净归因于数据修复。
- 若需更正式地固定种子, 可另写不带 enable_ce 语义的包装脚本 `python -c "import ppdet.engine.env as e; e.set_random_seed(0); ..."` 调 train.py 主流程 (设计备选, 默认走 `--enable_ce`)。

## 6. 运行流程 (设计, 未执行)

新增 `scripts/train_ablation.sh` (设计要点, 与 train_baseline.sh 同构):

```
# 用法: DATA_ARM=clean|exiffix bash scripts/train_ablation.sh
DATA_ARM=${DATA_ARM:?需指定 clean 或 exiffix}
DS=$PROJECT/dataset/processed_detection_${DATA_ARM}
EXP=$PROJECT/experiments/ablation_data_${DATA_ARM}
SAVE_DIR=$EXP/checkpoints   VDL_DIR=$EXP/logs
# 1) 校验数据 (仅 exiffix 臂): validate_exiffix.py 全 PASS
# 2) 重建软链: ln -sfn $DS $PD/dataset/processed_detection
# 3) 训练: python -u tools/train.py -c <原配置> --eval --use_vdl=true \
#         --enable_ce -o save_dir=$SAVE_DIR TrainReader.batch_size=16 LearningRate.base_lr=0.002
# 4) 日志: tee $EXP/train.log; 记录 train_command.txt
```

执行顺序: **先 Arm A (clean), 后 Arm B (exiffix)**; 两臂各自独立输出目录, 互不覆盖。

## 7. 评估与对比协议 (VAL 决定, TEST 封闭)

1. **主指标 (VAL, 同 Baseline 的 COCO eval)**: mAP@0.5:0.95, mAP@0.5, AR@100。以 **best_model (VAL-best)** 与 model_final 并列报告。
2. **per-class 指标 (VAL)**: 复用 `scripts/val_diagnosis.py` 对每臂计算 AP50 / AP50:95 / P / R / F1 / TP/FP/FN → `val_per_class_metrics.csv`。
3. **新增 `scripts/compare_ablation.py` (设计)**: 逐类并表两臂 AP50:95 / Recall, 标出 win/loss; 特殊关注点:
   - val 图 `TRAIN_000054` (7 个 Tomato leaf GT, 修复前 0 TP): 对比修复后该图 recall;
   - 受影响类 (Tomato 1/2/3/4 与 Apple rust 11) 的 recall 变化;
   - 输出 `ablation_data_*_VAL_comparison.csv` 与 `ABLATION_VAL_SUMMARY.md`。
4. **判定 (在 VAL 上进行)**:
   - 采纳: Arm B mAP@0.5:0.95 ≥ Arm A, 且受影响类 recall 不降 → 修复纳入正式管线;
   - 持平/不确定: 差异在种子/方差噪声内 (参考两臂 mAP 差 < 0.01) → 保持 Baseline, 记录结论;
   - 变差: 不采纳, 记录原因。
5. **TEST 封闭**: 两臂训练/调参全程不触碰 test。仅在 VAL 判定完成后, 若用户决定上 TEST, 用**唯一最终权重** (由 VAL 选出的胜者臂 best_model) 做**一次独立评估** (复用 eval_baseline.sh), 结果如实报告; **TEST 绝不用于选择数据臂或 checkpoint**。
   - 由于 test 分片两臂输入完全相同, TEST 对比仅反映训练侧数据差异, 不会引入评估集偏置。

## 8. 成功标准与最小可报告效果

| 指标 (VAL, best_model) | 阈值 |
|---|---|
| Arm B ≥ Arm A (mAP@0.5:0.95) | 主判据, 差值 > 0.005 视为显著 |
| 受影响类 recall 提升 | 期望正收益 (Tomato leaf / Septoria / bacterial spot / early blight / Apple rust) |
| img65 (TRAIN_000054) 单图 recall | 0 → ≥4/7 为强证据 |
| 全量几何校验 | validate_exiffix.py 全 PASS (数据前提) |

## 9. 风险与限制

| 项 | 说明 |
|---|---|
| 收益规模小 | 修复仅涉及 10 张 (0.88% train+val), 单图收益 (img65) 明显但整体 mAP 抬升可能有限 (预期 +0.0~0.01); 属正确性修复而非性能创新, 即使 mAP 持平, 消除错误图仍值得纳入管线。 |
| seed 非完全确定性 | cuDNN 等底层算子在相同 seed 下仍可能引入极小非确定性, 但两臂对称, 不影响归因。 |
| JPEG 重压缩 | 烘焙 quality=95, 内容仅重压缩差异 (EXIF_FIX_PLAN.md §4.4 校验), 对训练影响可忽略。 |
| test 共享 | test 分片两臂完全相同 → 无评估偏置; TRAIN_000064 的修复留待未来 (EXIF_FIX_PLAN.md §3.4)。 |
| 训练成本 | 两臂重训 ≈ 2×2.9h (RTX 4090); 如用户接受略降严格性, 可仅重训 Arm B 并以历史 Baseline 为 Arm A, 但结论需注明"未控种子"。 |

## 10. 产物清单 (完成后)

```
experiments/
├── ablation_data_clean/            Arm A (同 Baseline 数据, seed 固定)
│   ├── train.log / train_command.txt / checkpoints/ (best_model+model_final)
│   └── metrics/ (val eval + val_diagnosis 产物)
├── ablation_data_exiffix/          Arm B (EXIF 修复数据)
│   └── (同上)
├── baseline_v1/EXIF_FIX_PLAN.md    数据修复方案 (本计划的依赖)
├── baseline_v1/ABLATION_VAL_SUMMARY.md  两臂 VAL 对比结论 (由 compare_ablation.py 生成)
└── scripts/validate_exiffix.py / train_ablation.sh / compare_ablation.py
```

> **当前状态**: 本方案仅为设计; EXIF 修复、validate_exiffix.py、两臂训练均**未执行**, 待用户批准后按 EXIF_FIX_PLAN.md → ABLATION_PLAN.md 顺序推进。
