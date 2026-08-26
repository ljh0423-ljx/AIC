# D1 Density-aware VFL Target Calibration — Smoke Test 报告

- **日期**: 2026-08-16
- **阶段**: 仅「实现代码 + 2 epoch Smoke Test」;未访问 TEST;未修改原始数据集;**未启动 100 epoch 正式训练**。
- **结论**: **Smoke Test 全部通过 (35/35 机制检查 + 114 批次实时证据 + loss/梯度/数值/checkpoint/VAL 全正常)**。机制上已具备进入正式 100 epoch 训练的条件,但按用户指令**等待批准,不自动启动**。

---

## 1. 修改文件清单

| 文件 | 类型 | 说明 |
|---|---|---|
| `PaddleDetection/ppdet/modeling/heads/ppyoloe_head.py` | 修改(唯一源码改动) | D1 校准实现,默认关闭,关闭后与 A0 逐位一致 |
| `configs/ablation_dir4_D1.yml` | 新增(未运行) | 正式 100 epoch 配置,`use_density_calibration=True/density_thr=5/calib_floor=0.65`,其余逐字段 = A0 |
| `configs/ablation_dir4_D1_smoke.yml` | 新增(已运行) | 2 epoch / snapshot_epoch=1 / log_iter=5,基于 D1 配置派生 |
| `scripts/train_dir4_d1.sh` | 新增(已运行) | `MODE=smoke\|full` 启动器;smoke 先跑 35 项机制门禁(0 FAIL 才训练)+ `D1_DEBUG=1` 真实批次证据 |
| `scripts/d1_mechanism_check.py` | 新增(已运行) | 35 项静态/单元/全路径/反向机制检查 |

> 原始数据集、`processed_detection` 下所有文件均未改动;仅软链 exiffix 副本(与 A0 同一数据基础)。

---

## 2. 关键代码位置 (`ppyoloe_head.py`)

| 位置 | 内容 |
|---|---|
| L99-101 / L191-193 | 构造器新增参数 `use_density_calibration=False, density_thr=5, calib_floor=0.65`,存入实例属性 |
| **L355-369** | **`_density_calibrate`**: 校准核心公式 `paddle.where(pos & dense, maximum(assigned_scores, calib_floor), assigned_scores)`。仅正锚(`>0`)且密集图掩码=1 时抬升;负锚与 sparse 图锚点原样返回;纯函数,不改输入 |
| L371-399 | `_debug_density`: `D1_DEBUG=1` 时打印真实批次统计(全 detach,对数值/梯度零影响);未设置时完全不执行 |
| L503-510 | `get_loss`: 从 `pad_gt_mask` 逐图统计 `num_gt`(reshape+sum),`dense_img = num_gt >= 5`,构造锚点级 `dense_anchor_mask` 形状 `[B,1,1]`;仅 `use_density_calibration=True` 时构建 |
| L578 / L584 | `dense_anchor_mask` 传入两次 `get_loss_from_assign`(静态 ATSS 期 + 动态 TaskAligned 期) |
| **L595-613** | `get_loss_from_assign`: 校准只在**独立副本 `assigned_scores_cls`** 上进行;VFL 用校准后副本;`use_density_calibration=False` 时 `assigned_scores_cls is assigned_scores`(零改动) |
| L617 | 分类归一化保持 VFL 原有逻辑:`loss_cls /= assigned_scores_cls.sum()`(DDP all_reduce + clip min=1.) |
| **L640-648** | **回归分支完全隔离**: `assigned_scores_sum_reg = assigned_scores.sum()`(原始值),`_bbox_loss` 的逐锚权重与 IoU/DFL 分母仍用原始 `assigned_scores` → **bbox 回归不受 D1 影响** |

### 共享张量风险处理(关键设计决策)
`assigned_scores` 同时驱动:① 分类 VFL 目标、② 回归逐锚权重 `bbox_weight=assigned_scores.sum(-1)` 与归一化分母 `assigned_scores_sum`、③ CGPM。因此校准**绝不原位修改**该张量,而是产生 `assigned_scores_cls` 独立副本;回归路径用原始和 `assigned_scores_sum_reg`。已由机制检查 A5/A6/C3 逐位验证回归不受影响。

---

## 3. 校准公式

密集图(该图 GT 目标数 ≥ density_thr=5)的正锚:

```
gt_score' = max(gt_score, 0.65)
```

- 稀疏图(<5 targets): `gt_score' = gt_score`,与 A0 逐元素一致。
- 低 IoU 目标仅按上式抬到 0.65,**不额外加权、不乘系数**。
- 负锚(=0)不受影响,保持 0。
- 进入 VFL: `weight = alpha*pred_score.pow(gamma)*(1-label) + gt_score'*label`,正样本分支使用校准后目标;归一化 `loss_cls /= assigned_scores_cls.sum()` 不变。

**assigner 语义说明(如实记录)**: 前 30 epoch 用 ATSSAssigner,正样本目标 = 纯 IoU(`assigned_scores *= ious`),此时上式 = **字面 `max(IoU, 0.65)`**。epoch 31-100 用 TaskAlignedAssigner,正样本目标 = 重缩放对齐度量 `alignment/max_metrics × max_ious`(最优锚 ≈ GT 的最大 IoU),此时式为 `max(对齐度量, 0.65)`,对最优锚 ≈ `max(maxIoU, 0.65)`。两种路径 `assigned_scores>0` 均正确标识正锚,校准均正确生效。

---

## 4. Smoke Test 配置

| 项 | 值 |
|---|---|
| 配置 | `configs/ablation_dir4_D1_smoke.yml`(派生自 D1,再基继承 A0) |
| epoch / snapshot_epoch / log_iter | 2 / 1 / 5 |
| batch_size / base_lr | 16 / 0.002(CosineDecay max_epochs=2 + LinearWarmup 1 epoch) |
| seed | 0(`--enable_ce True`) |
| 数据 | exiffix(与 A0 一致),TEST 未参与 |
| D1 开关 | `use_density_calibration=True, density_thr=5, calib_floor=0.65`,`D1_DEBUG=1` |
| 环境 | RTX 4090 24GB, Paddle 2.6.2 cu118, Python 3.12.3, PaddleDetection @ b25522a0 |
| 产物 | `experiments/direction4/d1_smoke/`(train.log / mechanism_check.log / config 副本 / environment.txt / checkpoints/ / logs/) |

---

## 5. 机制验证结果

### 5.1 静态/单元机制检查 — **35 PASS / 0 FAIL**(`mechanism_check.log`)

- **A 静态检查 (8 项)**: 默认关闭;`density_thr=5`/`calib_floor=0.65` 默认值;校准用独立副本(A5);回归分支用原始 sum(A6);未新增独立 loss 模块(A7)。
- **B `_density_calibrate` 公式逐元素 (4 项)**: 低 IoU 0.3/0.5/0.1→0.65、高 IoU 0.8→0.8、sparse 原样、负锚=0(B1);仅正锚被抬升(B2);纯函数不改输入(B3);sparse 图逐元素一致(B4)。
- **C 全路径 `get_loss_from_assign` (11 项)**: `on+mask=None == off` 逐位一致(C1);`on+全稀疏 == off` 逐位一致(C2);含 dense 时 `loss_iou/dfl/l1` 与 off 逐位一致(回归不受影响, C3);`loss_cls` 差异恰等于「仅将 dense 正锚目标改为 max(·,0.65)」的重算值(C4)。
- **D 反向 (7 项)**: 校准进入梯度路径且改变梯度方向(D1);全稀疏时梯度与 off 逐位一致(D2);pred_scores 梯度有限(D3);`loss/loss_cls/loss_iou/loss_dfl/loss_l1` 全部有限(D4)。

### 5.2 Smoke 训练实时证据(D1_DEBUG, 114 个真实批次)

| 统计项 | 结果 |
|---|---|
| dense 图样本 / sparse 图样本 | **271 / 1553**(114 批中 93% 批次含 ≥1 dense 图) |
| 每批 dense_imgs / sparse_imgs | [0,5] / [11,16];7 个全 sparse 批次(校准空操作,短格式输出) |
| dense 正锚总数 | 21–342/批,合计 **12,458** |
| `below_floor == raised` 不匹配 | **0**(每个 IoU<0.65 的 dense 正锚恰好抬升一次) |
| `gt_cal_min != 0.6500` | **0**(校准后最小值恒为 0.65) |
| `sparse_same != True` | **0**(sparse 图 gt_score 逐元素与 A0 一致) |
| 校准前后 gt_score 均值 | raw mean ∈ [0.114, 0.814] → cal mean ∈ [0.650, 0.817](低 IoU 目标被显著抬升) |
| `cal == max(raw, 0.65)` 逐批校验 | **0 违例** |

抽查实录(真实批次):
```
[D1_DEBUG] dense_imgs=5 sparse_imgs=11 dense_pos=342 below_floor=263 raised=263 gt_raw_mean=0.4442 gt_cal_mean=0.6722 gt_cal_min=0.6500 sparse_same=True
[D1_DEBUG] dense_imgs=3 sparse_imgs=13 dense_pos=38 below_floor=32 raised=32 gt_raw_mean=0.5078 gt_cal_mean=0.6590 gt_cal_min=0.6500 sparse_same=True
```

### 5.3 Loss / 梯度 / 数值 / 内存

- **Loss 正常下降**: 首批 4.80(loss_cls 3.25)→ 末批 3.29(loss_cls 1.92);epoch 0 → 1 全程下降,无回升异常。
- **无 NaN/Inf**: 全日志扫描 0 命中。
- **无 Traceback/ERROR/OOM**: 全日志扫描 0 命中。
- **梯度有限**: 机制检查 D3/D4 逐项验证(off/dense/sparse 均有限)。
- **显存**: max_mem_reserved 10.7GB / 24GB,无 OOM 风险。

### 5.4 Checkpoint 与 VAL

- **Checkpoint 正常保存**: `0.pdparams/pdema/pdopt/pdstates`(epoch1)、`best_model.*`(最优 VAL)、`model_final.*`(最终),各 ~30MB,完整齐全。
- **VAL 正常评估**: 2 次 VAL eval 均完成(Best test bbox ap 0.009→0.026);2 epoch 尚在 warmup 尾,AP 数值无意义但评估管线正常;VAL FPS ~31-33。VAL 仅用官方 VAL 划分,**TEST 全程未访问**。

---

## 6. 结论 — 是否允许进入正式 100 epoch 训练

**Smoke Test 判定: PASS(允许进入正式训练的条件全部满足)**

| 用户要求验证项 | 结果 |
|---|---|
| ① dense≥5 才启用校准 | ✅ dense_imgs 阈值分桶正确(0 vs ≥1),机制 A/B + 实时统计 |
| ② sparse<5 逐元素与 A0 一致 | ✅ `sparse_same=True` 恒成立,机制 C2 + D2 逐位一致 |
| ③ 公式严格 `max(IoU,0.65)`,不改 0.65 | ✅ `gt_cal_min=0.6500` 恒成立,B1 逐元素 + C4 精确重算 |
| ④ 低 IoU 仅按公式,不加权不乘系数 | ✅ B1/B3 + 代码审查(`maximum` 单算子) |
| ⑤ `assigned_scores.sum()` 与 VFL 归一化不变 | ✅ L617 原逻辑保留;回归路径用原始 sum(L640-648) |
| ⑥ 默认关闭,关闭后与 A0 一致 | ✅ A1-A4,C1 逐位一致 |
| ⑦ 无新增 loss/attention/sampling/class weight | ✅ A7:无新 import、无新 nn.Layer |
| loss 下降 / 梯度正常 / 无 NaN/Inf/OOM | ✅ 见 5.3 |
| checkpoint 保存 / VAL 可评估 | ✅ 见 5.4 |
| dense/sparse 随机抽查 | ✅ 见 5.2(271 dense / 1553 sparse,114 批) |

**正式训练方式**: 待用户批准后执行 `MODE=full bash scripts/train_dir4_d1.sh`(100 epoch, bs=16, lr=0.002, seed=0, exiffix),输出 `experiments/direction4/d1/`,对照 A0 EXIF-Fix best checkpoint(mAP@0.5:0.95=0.4012)。

**按指令: 本阶段到此停止,等待用户下一条指令,不自动启动正式训练。**
