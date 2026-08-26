# DIRECTION2_DESIGN — 组内判别增强: 3 个候选方案设计

- 日期: 2026-08-15 | 设计阶段 (禁止训练 / 禁止修改正式模型 / TEST 封闭)
- 数据基础: **Arm B (exiffix)** —— 方向1 结论"有效", 作为后续实验候选数据基础
- 基线对照: Arm B `best_model` (VAL mAP@0.5:0.95 = **0.401**)
- 设计约束: 针对**真实错误模式** (见 DIRECTION2_DIAGNOSTIC.md), 不直接套 CBAM/SE 等通用模块; 保持 PP-YOLOE 轻量 + 高 FPS; 每个方案可被严格 ablation 证明。
- 三个方案均只作用于**分类判别**, 不动定位回归, 因此不损害已有 92% IoU≥0.75 的定位质量。

---

## 方案 A（推荐主推）: CGPM — Confusion-Guided Pair-Margin 混淆引导成对边界损失

### 0. 针对的真实错误模式 (数据证据)
- P1 分类混淆: 41 例定位对但类错 (占已定位 24.8%); 混淆对得分分布重叠 0.889 (bact↔Septoria) / 0.869 (mosaic↔yellow) → 特征坍缩。
- 9 例"自信地错" (score≥0.7): mold→lateblight @0.894、bact→Septoria @0.74@IoU0.961。
- VFL 分类损失 (当前 `PPYOLOEHead.use_varifocal_loss=True`) **无类别间判别边界**。

### 1. 具体解决的问题
在不损失定位质量与 FPS 的前提下, 给分类分支引入**类别间显式判别边界**, 专门拉开经验上高混淆的病害类对 (bact↔Septoria、mosaic↔yellow、leaf↔lateblight、Scab↔rust、EB↔Septoria 等), 使"定位对但类错"与"高置信类错"下降。

### 2. 算法机制 (新颖点: 边界来自经验混淆矩阵, 而非固定常数)
在 `PPYOLOEHead` 分类分支的预-sigmoid logits 上叠加一个**余弦成对边界项**, 与现有 VFL 并行:

1. **特征归一化**: 取每个正样本 (TaskAlignedAssigner 已对齐) 的分类特征 `f` 与类原型权重 `W` (13×C), L2 归一化, 得余弦相似度 `cosθ_c = <f̂, ŵ_c>`。
2. **混淆引导的成对 margin**: 对 GT 类 g, 仅对其"经验混淆对手"集合 `R(g)` (从 Arm B 在 **train** 集上的混淆矩阵取 top-K, K≤3, 已确认的类对) 施加**额外边界**: 对该样本, 将 GT 类 logit 推高 `s·cos(θ_g) + m_g`, 同时将 `R(g)` 类 logit 压低 `s·cos(θ_r) − m_r`, 其中 `m_g / m_r` 正比于混淆率 (从 train 混淆矩阵量化, 冻结一次)。
3. **损失**: 在 GT 类 softmax 分母中同时计入 margin 后做 cross-entropy 辅助损失, 以权重 `λ` (默认 0.25) 与 VFL 主损失相加。仅训练时计算 → **推理零开销, FPS 不变**。
4. **尺度 `s`**: 余弦尺度 (默认 16), 与 ArcFace 同思路但 margin 是**数据驱动、按类对加权、且只作用于 top-K 混淆对手** —— 这是与通用 ArcFace/CosFace 的关键区别: 不加固定 margin, 不惩罚所有非 GT 类。

### 3. 需要修改的代码位置 (PaddleDetection)
| 位置 | 改动 |
|---|---|
| `ppdet/modeling/heads/ppyoloe_head.py` `PPYOLOEHead.get_loss` (~L492 `if self.use_varifocal_loss`) | 增加 `loss_cls_cg = ConfusionMarginLoss(...)` 辅助项并加权加入总 loss; `forward_train` 透传分类特征 |
| `ppdet/modeling/losses/confusion_margin_loss.py` **(新增)** | 余弦归一化 + top-K 混淆对手 margin + cross-entropy 实现 |
| `configs/ppyoloe/_base_/ppyoloe_plus_crn.yml` `PPYOLOEHead:` | 新增 `confusion_margin: {scale: 16.0, margin_base: 0.15, lambda: 0.25, top_k_rivals: 2}` (默认关闭) |
| 训练脚本 | 无需改网络结构/assigner/增强 |

### 4. 训练参数是否改变
新增 4 个超参 (scale/margin_base/lambda/top_k_rivals); 其余 (seed=0, bs=16, lr=0.002, epoch=100, 数据=exiffix, 增强, assigner) **逐字段不变** —— 保证 ablation 唯一变量是 CGPM 本身。

### 5. 计算量增加
**训练**: 仅损失层一个 L2 归一化 + 13×C 点积, <1% FLOPs。**推理**: 0 (损失只在训练计算) → FPS 与 baseline 完全相同。

### 6. 预期影响的类别
直接命中 P1: bact (AP 0.284→↑)、Septoria、mosaic、yellow、leaf、lateblight、EB、mold、Apple Scab、Apple rust。预期"定位对但类错"41→下降, 9 例高分类错减少; mAP@0.5:0.95 主要受益。

### 7. 实验对照组 (严格 ablation)
1. **A0 = Arm B (exiffix, 无 CGPM)** — 已有 checkpoint, 直接复用。
2. **A1 = Arm B + CGPM** (混淆引导 margin, λ=0.25)。
3. **A2 = Arm B + 固定 margin** (margin 常数, 不按混淆加权) —— 用于证明"数据驱动混淆引导"相对"固定 margin"的增益 (创新点对照)。
4. **A3 = λ 灵敏度** {0.1, 0.25, 0.5} 中最优 vs A0。
- 成功判据: VAL mAP@0.5:0.95 提升 > 0.005 且 4 个命名混淆对 (bact↔Septoria、mosaic↔yellow、EB↔bact、mold↔EB) 的 per-class AP 多数提升、无显著回退 (>0.02 降幅的类数 ≤2)。

### 8. 可能失败的原因
1. margin 过大 → 精度回退或训练不稳 (用 A3 灵敏度兜底)。
2. VFL 已主导 loss → CGPM 只轻微重排 logits, mAP 增益 < 0.005 → 如实记录"方向2 分类边界非主瓶颈", 转方向3。
3. 混淆对手集合从 train 混淆矩阵导出, 若 train 混淆与 VAL 混淆分布偏差大 → 增益有限 (用 top-K≤3 缓解)。
4. 不解决漏检 (185) 与低置信正确 (38) → 若最终提升主要来自 mAP 而非 recall, 需配合方向3 的召回增强。

---

## 方案 B（推荐备用）: CCDH — Crop-Conditioned Discriminative Head 作物条件化判别头

### 0. 针对的真实错误模式 (数据证据)
- P2 作物内容量错配: 跨作物类错仅 3/41 (7.3%), 组内混淆 38/41 (92.7%); 番茄 8 类占 74% 目标。
- 单一 13 类判别器把容量花在"作物层 (已可分)"上, 番茄 8 类细粒度判别容量被摊薄。

### 1. 具体解决的问题
把 13 类判别**解耦为"作物门控 + 作物内细粒度判别"**两级: 作物层几乎零混淆 (数据证据), 因此分级可把判别容量集中到番茄 8 类的组内区分, 消除跨作物耦合对细粒度判别的干扰。

### 2. 算法机制
单头内实现两级分类 (不改 backbone、不改回归):
- **作物门控**: `pred_crop` 输出 3 维 (Tomato/Apple/Grape) sigmoid; 代价近乎为零。
- **作物内细粒度**: `pred_fine` 按作物分 3 组权重 (8/3/2 维), 共享 stem 特征。
- **最终类 logit**: `logit_total[c] = crop_gate[crop(c)] · fine_logit[c]` (逐类取两者激活度), 推理仍输出 13 类 → NMS/评估流程不变。
- **训练**: 作物 BCE 损失 (权重 0.5) + 作物内 VFL。分配器仍用 TaskAlignedAssigner 对 13 类 GT 对齐 (不变)。
- 参数量: 3 + 8 + 3 + 2 = 16 输出头 vs 13, 增加 <3% (仅最后一层) → 可忽略。

### 3. 需要修改的代码位置
| 位置 | 改动 |
|---|---|
| `ppdet/modeling/heads/ppyoloe_head.py` | `PPYOLOEHead` 的 `stem_cls/pred_cls` 改造为 `pred_crop (3)` + `pred_fine` (按作物分组权重); `forward_train`/`forward_eval` 与 `get_loss` 相应调整 |
| 或新建 `ppdet/modeling/heads/ppyoloe_crop_head.py` (继承 PPYOLOEHead) | 隔离改动, 便于回滚 |
| `configs/ppyoloe/_base_/ppyoloe_plus_crn.yml` | `yolo_head: PPYOLOECropHead`, 新增 `crop_loss_weight: 0.5` |

### 4. 训练参数是否改变
新增 `crop_loss_weight`; 其余不变。网络结构改动**仅在 head 末层**, backbone/neck/assigner/增强不动。

### 5. 计算量增加
<3% 参数 (仅末层); 推理前向多 3 个 sigmoid → **FPS 基本不变**。

### 6. 预期影响的类别
番茄 8 类 (若耦合确实在伤害细粒度判别, 则 Septoria/bact/mosaic/yellow/EB/mold 均受益); Apple Scab↔rust 受益; 预期跨作物耦合相关错误下降。

### 7. 实验对照组
1. B0 = Arm B (无结构改动)。
2. B1 = Arm B + CCDH (两级头)。
3. B2 = B1 去掉 crop_loss (只分级权重, 不加作物损失) —— 分离"结构解耦"与"作物监督"的各自贡献。
- 成功判据: 同方案 A (mAP +0.005 阈值 + 命名对 per-class 无回退)。若 B2≈B1, 说明主要来自结构; 若 B1≈B0, 说明耦合不是主瓶颈, 如实记录。

### 8. 可能失败的原因
1. 番茄 8 类样本在单组头中仍不足 → 细粒度可分性不来自容量错配, 而来自特征本身 (P3 通道不足) → 增益小。
2. 作物门控错误 (现实中 ~0, 但一旦错则灾难) → 需要 crop_gate 高置信才走细粒度。
3. Grape 组仅 2 类/36 GT, 组头训练不足 → 葡萄类可能轻微回退 (用共享 stem + 小分类头缓解)。
4. 结构改动比损失改动风险高, 需更仔细的回归测试。

---

## 方案 C: ATRR — Ambiguity-Triggered Texture Refinement 模糊触发细粒度纹理精化

### 0. 针对的真实错误模式 (数据证据)
- P3 细粒度纹理表达不足: bact↔Septoria 靠局部病斑微观结构区分, 最大混淆边 43.8%; P3 通道 (width 0.5) 摊薄。
- 校准失准: 38 例正确检测 score<0.6, 9 例高分类错 → 单一置信度不可靠。
- 混淆类不共现 → 只能靠局部纹理。

### 1. 具体解决的问题
给分类分支提供一个**更高分辨率、聚焦局部病灶纹理**的精化通路: 当 top-2 类得分在模糊带 (|logit1−logit2|<δ) 内时, 用局部 P3 纹理特征对候选两类重打分, 把"凭整体特征难分、凭局部纹理可分"的对 (bact↔Septoria) 救回来。

### 2. 算法机制
- **训练**: 对每个正样本 (TaskAlignedAssigner 已对齐), 用 RoIAlign 从 P3 (stride 8, 最高分辨率) 提取 7×7 病灶 crop → 微型编码器 (2×Conv, 32 通道) → 与混淆对手集合做**小 margin 余弦对比** (复用方案 A 的 margin 机制), 作为辅助损失。
- **推理 (稀疏激活)**: 仅当 `|logit1−logit2| < δ` 时, 对该检测取 P3 crop 过 2 层卷积 + 小型 MLP, 重加权 top-2 类 logit。模糊带外不触发 → 大多数推理路径零额外开销。
- 与现有 `PPYOLOEContrastHead` (通用 SupContrast, PP-YOLOE-R 变体) 的区别: ① 只对**混淆对手**做判别 (非全部类); ② **推理时按模糊度触发** (非全程计算); ③ 作用于 P3 局部纹理 (非整体池化特征)。

### 3. 需要修改的代码位置
| 位置 | 改动 |
|---|---|
| `ppdet/modeling/heads/atrr_head.py` **(新增)** | RoIAlign (可用 `paddle.vision.ops.roi_align` 或 `ppdet/modeling/ops`) + 微型纹理编码器 + 模糊触发精化 MLP; 继承/组合 PPYOLOEHead |
| `ppdet/modeling/losses/texture_margin_loss.py` **(新增)** | 局部纹理 margin 辅助损失 |
| `configs/ppyoloe/_base_/ppyoloe_plus_crn.yml` | 挂载 ATRR, 新增 `atrr: {roi_size: 7, delta: 0.15, lambda: 0.2}` |

### 4. 训练参数是否改变
新增 RoIAlign 与微型编码器参数 (极小); 新增 `delta/lambda`; 其余不变。

### 5. 计算量增加
训练: +5–10% (RoIAlign + 微编码器仅作用于正样本)。推理: +1–3% (仅模糊带内触发; δ 默认 0.15 时触发比例低)。FPS 影响可测但小。

### 6. 预期影响的类别
bact↔Septoria (最高价值)、mosaic↔yellow、EB↔mold 等**局部纹理可分**的对; 同时通过重打分改善低置信正确检测 (score 抬升) 与高分类错。

### 7. 实验对照组
1. C0 = Arm B。
2. C1 = Arm B + ATRR (训练+推理触发)。
3. C2 = C1 去掉推理触发 (仅训练正则) —— 分离"推理精化"与"训练判别"的贡献。
4. C3 = δ 灵敏度 {0.1, 0.2, 0.3}。
- 成功判据: mAP +0.005 且命名对 per-class 提升, 推理 FPS 下降 ≤5%。

### 8. 可能失败的原因
1. 病灶极小 (bact 点斑) → RoIAlign 在 P3 上的 crop 分辨率仍不足 → 增益小。
2. 局部纹理与整体特征信号弱相关 → 重打分无效甚至干扰 (用 C2 判定)。
3. 推理触发逻辑 (RoIAlign 在图内随机位置取点) 增加工程复杂度与延迟抖动。
4. 改动最大、风险最高; 收益若被方案 A 覆盖, 则优先级后移。

---

## 主/备推荐与依据

### 主方案: 方案 A CGPM (Confusion-Guided Pair-Margin)
- **直接命中测量到的头号错误** (定位对但类错 41 例 / 得分重叠 0.889 / 9 例高分类错)。
- **零推理开销, FPS 完全不变** —— 最符合"轻量化 + 高 FPS"约束。
- **创新点可辩护**: margin 由经验混淆矩阵按类对加权导出、只作用于 top-K 混淆对手, 区别于通用 ArcFace/CosFace 的固定 margin, 可写论文消融 (A2 固定-margin 对照)。
- **Ablation 最干净**: 唯一新增是损失项, 其他逐字段不变; 失败时损失几乎为零。
- 计算量 <1% (仅训练)。

### 备用方案: 方案 B CCDH (Crop-Conditioned Discriminative Head)
- 数据证据同样强 (跨作物类错 3/41 → 作物层几乎可分), 结构改动仅 head 末层, FPS 基本不变。
- 与 CGPM 可叠加 (互不冲突): 若 A 提升有限, B 提供独立的容量重分配路径。

### 不推荐先做: 方案 C ATRR
- 创新度最高但改动最大、FPS 风险与工程风险最高; 收益很可能被 A (margin) 部分覆盖。作为 Phase-2 扩展保留。

---

## 实现顺序建议 (设计通过后执行, 当前不训练)
1. 用 `scripts/direction2_diagnosis.py` 的匹配逻辑, 对 **Arm B 在 train 集**上评估一次, 导出 train 混淆矩阵 → 冻结 CGPM 的 `R(g)/m_g` 查表 (存 JSON)。
2. 实现 CGPM (损失 + 配置), 先跑 A1 vs A0 单轮验证。
3. 依次执行 ABLATION_DIRECTION2_PLAN.md 的对照表; 每轮仅 VAL 评估, TEST 封闭。
4. 若 A 提升 <0.005, 再实现 CCDH (B1/B2)。

> 本阶段仅设计, 未修改任何模型/代码/数据, 未开始训练, TEST 未触碰。
