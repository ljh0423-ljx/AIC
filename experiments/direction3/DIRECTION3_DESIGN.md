# 方向3 设计文档 — B1: Class-Balanced Training (class-weighted VFL)

- 日期: 2026-08-15
- 状态: **设计完成, 未开始训练** (按用户指示, 完成设计后停止)
- 基准: **A0** (EXIF-Fix + 原始 VFL), mAP@0.5:0.95 = 0.401
- 约束: 只使用 TRAIN 统计生成固定权重; VAL/TEST 不参与权重计算; TEST 全程封闭; 不修改原始数据集
- 背景: 深度诊断结论 (`DIRECTION3_DIAGNOSTIC.md`) — 点名漏检类 (yellow/mold/mosaic) 的漏检是**置信度被密集场景压低**, 而非欠训练; yellow 是 TRAIN 实例数最多的类 (700)。本设计的预期收益与诊断结论的诚实评估见 §5。

---

## 1. 推荐方案: class-weighted VFL (损失侧加权)

**推荐理由 (为什么选它而不是 class-aware sampling):**

| 维度 | class-weighted VFL (推荐) | class-aware sampling |
|---|---|---|
| 机制数 | 1 (仅在分类损失乘类权重) | 1 (改 dataloader) |
| 可解释性 | 公式透明, 每类权重可直接检查 | 采样概率需还原为"等效倍数", 间接 |
| 是否引入重复数据 | **否** | 过采样会产生重复图像, 污染 epoch/数据分布 |
| 是否改变推理 | 否 (权重仅训练期用) | 否 |
| 与 A0 单变量性 | 只改 loss 内一个乘法 | 同时改了数据分布 + 隐含梯度分布 |
| 训练/验证统计独立性 | 权重仅来自 TRAIN 计数 | 同 |

**决定: 采用损失侧 class-weighted VFL。** 单一机制、无重复数据、推理与 A0 完全一致, 是最容易做严格 ablation 的方案。

## 2. 权重公式与生成 (冻结, 仅 TRAIN)

### 2.1 公式

```
w_c = (1 / N_c) / mean_{c'}(1 / N_{c'})      c = 1..13
w_bg = 1.0
```

- `N_c` = TRAIN 中类 c 的实例数 (annotation 计数)
- 归一化后 **mean(w_c)=1.0, Σw_c = 13** → 全局 loss 尺度不变, 只改变类间相对强调
- 背景通道权重 1.0 (VFL 中背景通过 `label[..., :-1]` 裁掉, 实际不参与, 保留字段仅为完整性)

### 2.2 生成方式

- 脚本: `scripts/gen_class_weight.py` (只读取 `dataset/processed_detection_exiffix/annotations/train.json`)
- 输出: `experiments/direction3/class_weight_train.json` — **冻结 JSON**, 训练前生成一次, 全程不改
- 严禁: 用 VAL/TEST 计算或调整任何权重

### 2.3 已生成的固定权重 (Σw=13, 均值 1.0)

| 类 | N_train | w_c | 相对 A0 强调 |
|---|---|---|---|
| Early blight | 171 | 1.104 | ↑ |
| Septoria | 338 | 0.559 | ↓ |
| leaf | 294 | 0.642 | ↓ |
| bacterial spot | 227 | 0.832 | ↓ |
| late blight | 159 | 1.188 | ↑ |
| mosaic virus | 208 | 0.908 | ↓ |
| **yellow virus** | **700** | **0.270** | **↓↓ 最强下调** |
| mold leaf | 239 | 0.790 | ↓ |
| Scab | 125 | 1.511 | ↑ |
| Apple leaf | 198 | 0.968 | ≈ |
| rust | 149 | 1.267 | ↑ |
| grape leaf | 173 | 1.092 | ↑ |
| black rot | 101 | 1.870 | ↑↑ 最强上调 |

## 3. 实现位置与精确代码 (B1)

### 3.1 数据流与维度

- `pred_scores`: `[B, N, 13]`, 类维度在**最后一维**, 顺序 = category_id 1..13 (= 训练 0-based 0..12)
- `label` = `one_hot(assigned_labels, 14)[..., :-1]` → `[B, N, 13]` (背景通道裁掉, `get_loss_from_assign` 行 513-514)
- `assigned_scores`: `[B, N, 13]`, 正样本对应类为质量分数, 其余 0

### 3.2 修改 1 — `__init__` 增加开关 (仿 CGPM 接入模式, 默认关闭)

文件: `PaddleDetection/ppdet/modeling/heads/ppyoloe_head.py`
位置: `__init__` 参数表 (行 91-95 CGPM 参数之后) 与初始化块 (行 154-167 CGPM 块之后)

```python
# __init__ 参数追加:
use_cls_weight=False,
cls_weight_path='',
```

```python
# __init__ 初始化块追加 (CGPM 块之后):
self.use_cls_weight = bool(use_cls_weight)
self.cls_weight = None
if self.use_cls_weight:
    assert cls_weight_path, 'use_cls_weight=True 时须提供 cls_weight_path (TRAIN-only, 冻结)'
    import json
    with open(cls_weight_path) as f:
        cw = json.load(f)['weight']
    self.cls_weight = paddle.to_tensor(
        [float(cw[str(c)]) for c in range(1, self.num_classes + 1)],
        dtype='float32')  # [13], 顺序对齐 category_id 1..13
```

### 3.3 修改 2 — `_varifocal_loss` 加权 (行 320-325)

`_varifocal_loss` 当前是 `@staticmethod`。B1 需访问 `self.cls_weight`, 故**去掉 `@staticmethod` 装饰器** (调用方式 `self._varifocal_loss(...)` 不受影响), 并在 `weight` 上乘类权重:

```python
def _varifocal_loss(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
    weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
    if self.use_cls_weight:
        # B1: 类加权 VFL, [B,N,13] × [1,1,13]; 仅训练期, 推理路径不变
        weight = weight * self.cls_weight[None, None, :]
    loss = F.binary_cross_entropy(
        pred_score, gt_score, weight=weight, reduction='sum')
    return loss
```

**归一化 (不改动)**: 保持 `loss_cls /= assigned_scores_sum`, 其中 `assigned_scores_sum = assigned_scores.sum()` (行 520)。理由: `w` 均值为 1.0, 期望全局 loss 尺度近似不变, 使 B1 与 A0 的差异**严格只在类间相对权重**, 单变量性最强。`self.cls_weight` 的梯度是恒定的缩放因子, 不影响参数可学习性。

### 3.4 修改 3 — B1 配置文件 (新建)

文件: `configs/ablation_dir3_B1.yml`

```yaml
# B1 — A0(exiffix) + class-weighted VFL
# 唯一变量: Head.use_cls_weight=True 带来的类加权 VFL。其余与 A0 逐字段一致。
_BASE_: [
  './ppyoloe_plus_crn_s_100e_agrivision.yml',
]
epoch: 100
LearningRate:
  base_lr: 0.002
TrainReader:
  batch_size: 16
EvalReader:
  batch_size: 2
PPYOLOEHead:
  use_cls_weight: True
  cls_weight_path: /root/autodl-tmp/AIC2026_AgriVision/experiments/direction3/class_weight_train.json
```

> A0 对照组: **不重训**。直接复用方向2 A0 (exiffix) best checkpoint 作为 A0 臂 (已有 `direction2/val_eval/A0_best`), 保证 A0→B1 严格单变量 (唯一差异 = 训练损失里的类权重)。

## 4. 训练数据与实验目录

| 项 | 值 |
|---|---|
| 训练数据 | `dataset/processed_detection_exiffix` (与 A0 完全一致; 原始数据集不动) |
| 权重来源 | `experiments/direction3/class_weight_train.json` (仅 TRAIN, 冻结) |
| 实验目录 | `experiments/ablation_dir3_B1/` |
| 超参 | epoch=100, bs=16, lr=0.002, seed=0 (`--enable_ce True`), CosineDecay, 增强, Objects365 预训练 — **与 A0 全部一致** |
| 训练命令 | `python -u tools/train.py -c configs/ablation_dir3_B1.yml --eval --use_vdl=true --enable_ce True -o save_dir=experiments/ablation_dir3_B1/b1` |

## 5. 预期影响与诚实先验 (诊断结论对 B1 的约束)

- **点名漏检类 (yellow/mold/mosaic) 均被下调权重** (0.270 / 0.790 / 0.908)。诊断显示它们的漏检主因是密集场景置信度压低而非欠训练 → **B1 对这三个类的预期收益低, yellow 甚至可能因 w=0.270 而 AP 回落**。这是设计阶段就要接受的先验, 不是事后找补。
- **B1 的真实作用面**: 上调权重真正惠及 TRAIN 稀少类 (black rot 1.870 / Scab 1.511 / rust 1.267 / lateblight 1.188) —— 这些类 VAL 实例虽少, 但属"相对欠学习"候选。
- **若 B1 达不到 +0.005 门槛, 属预期结果之一** (诊断已证明不平衡非漏检主因), 按 §6 诚实报告, 不调参。

## 6. 监控指标与过拟合 / 强类回退风险设计

### 6.1 强类回退监控 (A0 最强 8 类, 阈值 ΔAP < −0.02 即告警)

| 类别 | A0 AP | w_c | 风险 |
|---|---|---|---|
| Apple leaf | 0.811 | 0.968 | 低 (≈1.0) |
| grape leaf | 0.738 | 1.092 | 低 (上调) |
| rust | 0.650 | 1.267 | 低 (上调) |
| Scab | 0.484 | 1.511 | 中 (上调, N=125 少 → 过拟合风险) |
| **Septoria** | 0.461 | **0.559** | **高 (被下调且是强类 — 最可能回退)** |
| black rot | 0.403 | 1.870 | 中 (大幅上调, N=101 → 过拟合风险最高) |
| late blight | 0.379 | 1.188 | 低 (上调) |
| bacterial spot | 0.284 | 0.832 | 低 |

### 6.2 过采样重复/过拟合风险

- **无过采样**: 本方案不做任何图像重复 → 数据加载器、每 epoch 图像序列、训练步数与 A0 逐位一致, 从机制上排除"重复样本导致过拟合"这一混淆变量。
- **过拟合监控** (针对上调类):
  - 训练期 VAL mAP 随 epoch 曲线 (PP-YOLOE `--eval` 每 epoch 自动评估) — 若上调类 train loss 下降但 VAL AP 停滞/回落 → 过拟合信号
  - 对 N<150 的上调类 (black rot 101, Scab 125, rust 149) 单独看 per-class AP 轨迹
  - best_model 与 model_final 双评估 (best 用于主判据, final 用于稳健性检查)

### 6.3 训练稳定性监控

- 训练日志: loss_cls 曲线应正常下降、无 NaN; 若 loss 量级与 A0 偏离 >±30% 且无收敛 → 停查 (w 均值 1.0 应使量级近似不变)
- seed=0 固定, 排除随机性

## 7. 成功判据 (B1, 同时满足才算有效)

| 判据 | 指标 | 要求 |
|---|---|---|
| 主判据 | VAL mAP@0.5:0.95 (best) | **B1 − A0 ≥ +0.005** |
| 副判据 1 | 弱势类 Recall@conf0.5 | 至少 1 个被点名漏检类 (yellow/mosaic/mold) Recall 不降, 或 TRAIN 稀少类 (blackrot/Scab/rust/lateblight) 任一 Recall 提升 |
| 副判据 2 | 总体 Recall / F1 @conf0.5 | 至少一项优于 A0 |
| 负向红线 | §6.1 强类回退 | 任一 A0 强类 ΔAP < −0.02 即触发告警, 需在报告中评估是否抵消收益 |

**判定逻辑**: 主判据为硬门槛。副判据 2 为硬门槛 (要求总体提升, 不能只看单类)。负向红线用于解释"AP 微升但强类回退"的拆解。全部判据都基于 **best_model** 且用同一评估脚本复算 A0, 避免跨脚本偏差。

## 8. 与方向2 的关系

- B1 与 CGPM (方向2) 正交且互斥: B1 仅含 class-weighted VFL, **不叠加 CGPM**。方向2 已正式关闭, 其失败结论不进入 B1。
- A0 臂复用的 checkpoint 与方向2 A0 完全相同, 三份文档共享同一基准, 便于横向对照。

## 9. B2 (gated, 本期不执行)

- B2 = Density-Aware Inference, 方向与诊断结论直接对应 (降阈值/分场景置信度策略可直接收复 conf 0.3-0.5 的 125 个漏检)。
- **仅当 B1 完成且有效后才设计/执行**, 本期仅记录其存在。
