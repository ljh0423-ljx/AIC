# S1 CG-ACS 设计文档 (CG_ACS_DESIGN)

> **S1 = CG-ACS:Classification-Grounded Auxiliary Classification Supervision(分类锚定的辅助分类监督)**
> 阶段:0(门槛核验 + 最小可行设计 + smoke test)**不进行 100 epoch 正式训练**。
> 约束:不修改历史实验;检测头 PPYOLOEHead 完全不变;分类头仅训练、不参与推理;`lambda_cls=0.1` 为唯一新增变量,不调参。
> 日期:2026-08-16

---

## 1. 动机与创新点

### 1.1 解决 A0 的什么瓶颈
A0(PP-YOLOE+-s,PlantDoc 13 类,VAL mAP@0.5:0.95 = **0.4012**)的主要瓶颈(P1,P2,见 `NOVELTY_DESIGN_REPORT.md`):

- **P1 类间混淆**:番茄病害形态高度相似(Early blight / Septoria / Leaf Mold / Bacterial spot 等),框级 VFL 只惩罚"锚点格上的误分类",缺乏**整图层面的语义判别**约束。
- **P2 数据稀疏**:PlantDoc 检测训练集仅 **915 张**(3084 个框),13 类严重不平衡(Tomato 占 ~80%),代表性不足。
- **P3 分类图未利用**:PV_CORE 提供 **19,233 张**与 13 类 1:1 精确对齐的整图分类样本,当前完全未被使用。

### 1.2 CG-ACS 创新机制
- 在**共享的 backbone+neck 特征**上增加一个**图像级 13 类辅助分类头**(aux head),与检测头**并行**。
- **PlantDoc 样本**:同一张图同时产生 ① 原检测 loss(不变)② 图像级分类 loss(该图实际出现的类别集,multi-label)。
- **PV_CORE 样本**:仅产生图像级分类 loss(one-hot,严格不生成伪 bbox、不当作检测图)。
- **两股监督共用同一份特征**,让共享表示同时满足"定位+分类"与"整图语义判别"两个目标,从数据层面扩大分类监督规模(915 → 915+19233)。
- 这是**多任务图像级辅助分类**(image-level auxiliary classification supervision),不是伪标签、不是自训练、不改变检测头内部任何逻辑。

### 1.3 为什么不用伪框 / 伪标签
PV_CORE 图是**单一/双叶居中、统一灰底、无框标注**。任何伪框方案都要先检测再过滤,噪声大、成本高、且把"分类监督"退化成"带噪检测监督"。而 PV_CORE 的**类别标签精确**,图像级监督是信噪比最高的用法。见 `CG_ACS_MECHANISM_CHECK.md` 门槛 3。

---

## 2. 总体架构

```
                    ┌─────────────────────────────────────────────┐
   image            │           共享 Backbone (CSPResNet)          │
   ────────────────►│            + 共享 Neck (CustomCSPPAN)        │
                    └─────────────────────────────────────────────┘
                                      │  neck_feats
                    ┌─────────────────┼──────────────────────────┐
                    │                 │                          │
                    ▼                 ▼                          ▼
      PPYOLOEHead (检测头)     CGACSAuxClsHead (辅助分类头)      ← 仅在训练
      ★ 完全不变               P3, stride 8, 96ch               │  中激活
      VFL+IoU+DFL              GAP → FC(96→13) → [B,13]          │
                    │                 │                          │
                    ▼                 ▼                          ▼
         det loss (A0 不变)      λ·BCE(图像级)                 推理
                                 λ = lambda_cls = 0.1          旁路
```

- **共享**:backbone `CSPResNet`(layers[3,6,6,3], width 0.5, Objects365 预训练)+ neck `CustomCSPPAN`(P3/P4/P5 = 96/192/384 ch)。
- **检测头 `PPYOLOEHead` 完全不变**:参数、前向、assigner、NMS、loss 计算逐字节与 A0 相同。
- **辅助分类头 `CGACSAuxClsHead`(新增,默认关闭)**:输入 `neck_feats[0]`(P3,96ch,stride 8)。

---

## 3. 特征层选择与辅助头结构

### 3.1 选 P3(stride 8,96ch)的原因
> 运行时实测:PP-YOLOE+ CustomCSPPAN 的返回顺序为 **[P5, P4, P3]**(与检测头 `fpn_strides: [32,16,8]` 对齐),因此 P3 位于 **neck_feats[2]**,通道 96,stride 8。

| 候选 | 返回索引 | 通道 | 分辨率 | 语义粒度 | 结论 |
|---|---|---|---|---|---|
| P5 (stride 32) | 0 | 384 | 20×20@640 | 全局语义强,但病斑细节弱 | 不选 |
| P4 (stride 16) | 1 | 192 | 40×40 | 折中 | 备用 |
| P3 (stride 8) | **2** | **96** | 80×80@640 | 高分辨率,纹理/病斑细节最丰富 | **选用** |

病害判别依赖**局部纹理细节**(病斑形状、颜色、分布),P3 语义信息最充足;同时 P3 是 3 个输出层里通道最少(96)的,辅助头参数增量最小。

### 3.2 辅助头结构
```
neck_feats[2]  (P3, [B,96,H,W])
   → AdaptiveAvgPool2D(1)      # GAP, [B,96,1,1]
   → Flatten                  # [B,96]
   → Linear(96 → 13, bias)     # [B,13]  logits
```
- **参数量**:96×13 + 13 = **1,261** 个参数。
- **无 BN/Dropout**:单层线性,GAP 后已无空间结构,不需要归一化;避免训练/推理模式切换问题。
- **激活**:无(输出 logits,直接进 BCE-with-logits)。

---

## 4. 损失函数

### 4.1 定义
辅助分类 loss = **BCE-with-logits(multi-label)**:

```
L_aux(y_hat, y) = mean_{b,c} BCE(logit_{b,c}, y_{b,c})        # 对 [B,13] 全元素取平均
```

- **PlantDoc 检测流** 目标 `y` = multi-hot:第 c 类在该图中存在任意 bbox → 1,否则 0。
- **PV_CORE 分类流** 目标 `y` = one-hot(该图精确类别)。

### 4.2 总损失
```
det 流: L_det_total = L_det(VFL + 2.5·IoU + 0.5·DFL) + λ · L_aux(det batch)
cls 流: L_cls_total = λ · L_aux(PV_CORE batch)
每优化步: L_total = L_det_total + L_cls_total
其中 λ = lambda_cls = 0.1(唯一新增变量,阶段 0 固定,不调参)
```

### 4.3 为什么 BCE-with-logits + multi-label
- PlantDoc 单图**可能含多个类别**(多框不同类),multi-label 是正确的图像级监督形式。
- PV_CORE 单图 one-hot 是 multi-label 的特例,同一函数无缝覆盖两股流。
- 对 13 类全平均,数值与 batch 无关地稳定在 O(0.1) 量级,不会压过检测 loss。

---

## 5. 训练数据流(双流 batch 设计)

### 5.1 两股数据流
| 流 | 来源 | 每批 | 监督 | 采样 |
|---|---|---|---|---|
| det 流 | PlantDoc train(exiffix, 915 图) | 16 | det loss + λ·L_aux(multi-label) | 原 A0 COCODataSet + A0 增强 |
| cls 流 | PV_CORE(19,233 图) | 16 | λ·L_aux(one-hot) | PVClsDataset,逐 epoch shuffle |

> 修正注记(2026-08-16,正式训练前经用户裁决):S1 det 流 batch size 采用 **16**(与 A0 实际运行一致),cls 流 **16**(保持"每张约见 ~5 次"采样比);`base_lr=0.002`(与 A0 一致)。原稿 "8/8、lr0.001" 系按 A0 配置文件默认值书写,未与 A0 实际运行(bs16/lr0.002)对齐;为满足"禁止修改 batch size/learning rate 等既有训练条件",统一按 A0 实际运行执行。

### 5.2 双流 batch = 梯度累积(1 det + 1 cls 每优化步)
```
for epoch in range(epochs):
    for det_batch in det_loader:              # 16 张 PlantDoc, 640 多尺度
        outputs_det = model(det_batch)        # aux_mode='det'
        outputs_det['loss'].backward()        # 累积 det 梯度
        cls_batch = next(cls_loader)          # 16 张 PV_CORE, 320
        outputs_cls = model(cls_batch)        # aux_mode='cls' → 仅分类头
        outputs_cls['loss'].backward()        # 累积 cls 梯度
        optimizer.step(); lr.step(); optimizer.clear_grad(); ema.update()
```

**关键性质(与 A0 完全一致)**:
- det 流迭代次数 = A0(bs16 下每个 epoch 57 批),**优化器步数、LR 计划(CosineDecay+Warmup)逐步与 A0 相同**。
- 梯度在 step 前同时累积(det 与 cls 两股 loss 合并更新),等价于 `L_total` 的单次 backward。
- det 流每个样本的训练强度与 A0 完全一致,额外获得的是 PV_CORE 的图像级监督。

### 5.3 两股流在模型内如何区分
`inputs['aux_mode']`:
- `'det'`:完整 A0 前向(det head)+ 辅助分类头(multi-hot 目标)。
- `'cls'`:**跳过 det head**(PV 无 bbox),只算 `backbone→neck→aux head` 的 λ·L_aux。

### 5.4 PV_CORE 采样器
- 索引:由 `file_manifest.csv` 生成 `experiments/direction5/pv_cls_index.csv`(绝对路径 + aic_class_id),**只读引用 PV_CORE,不复制、不修改原图**。
- 每优化步消耗 16 张;100 epoch 下 PV_CORE 每张约见 ~5 次(16×57×100/19233≈4.7)。类别不平衡保留(与检测流共享同一套类别先验,不做重采样,阶段 0 最小实现)。

---

## 6. 参数量 / 显存 / 计算开销估算

| 项目 | A0 | A0 + S1 (aux head) | 增量 |
|---|---|---|---|
| 模型参数量 | PP-YOLOE+-s **7,701,367**(实测) | +**1,261** | **+0.016%**(可忽略) |
| 辅助头 FLOPs | — | GAP+Linear(96→13)≈0.4 MFLOP/batch | **≈0%** |
| 检测流前向(640) | **16.30 G-FLOP/img**(实测,smoke) | 同 A0 | 0 |
| cls 流前向(320×320) | — | **3.50 G-FLOP/img**(实测,仅 backbone+neck+aux) | 新增 |
| **每优化步总计算** | 1× det | 1× det + 1× cls(320) | **+22%(相对 640)~ +30%(相对平均缩放)** |
| 激活显存峰值 | bs16@640(占满) | bs8@avg~512~544 + bs8@320 | 约为 A0 的一半 |
| 优化器状态 | — | aux 头 1,261×3 字节 | 可忽略 |

> 注(2026-08-16 smoke 实测,训练前向 bs8):det 流 640/544/320 分别为 **16.30 / 11.77 / 4.07 G-FLOP/img**(BatchRandomResize 320~768 随机,首批实测 512);cls 流固定 320 为 **3.50 G-FLOP/img**,仅 backbone+neck+aux(无检测头)。因此每步计算增量 = 3.50/det@avg ≈ **+22%~+30%**。
> > 修正注记(正式训练,bs16/16):A0(bs16@640)显存实测 max_mem_reserved ≈ 10.9GB;S1(bs16 det 顺序 backward + bs16@320 cls)显存 ≈ A0 + cls 流激活(~2GB),峰值约 13GB,24GB 无 OOM(正式训练实测确认)。
>
> > 修正说明:设计初稿曾引用 "43.9 GFLOPs" 为 det 流 640 前向,经 smoke 重新实测应为 **16.30 G-FLOP/img**(旧值疑似将 batch 内多次前向或某次多尺度前向混入统计)。

---

## 7. 推理旁路设计(分类头不参与推理)

- `yolo.py` `_forward()` 中,`self.aux_cls_head` **仅在 `self.training` 分支被访问**。
- 推理分支(`else`)完全不引用 aux head → 前向结构、后处理、NMS 与 A0 逐字节一致。
- 导出/部署:aux head 参数存在于 state_dict(用于训练续训),但不进入任何推理路径,推理 FPS 与 A0 相同,零开销。
- smoke test 验证(f):用 S1 训练权重跑 `tools/eval.py`,输出结构与 A0 评估完全一致。

---

## 8. 与 A0 的一致性保障

| # | 保障措施 |
|---|---|
| 1 | `aux_cls_head` 默认 `None`;未配置时 `_forward` 走原逻辑,输出与 A0 逐键一致 |
| 2 | 检测头 PPYOLOEHead 源码零改动(不触碰 ppyoloe_head.py) |
| 3 | det 流数据读取、增强、采样完全复用 A0 配置,seed=0 |
| 4 | 优化器/EMA/LR 计划与 A0 相同;每优化步恰好 1 个 det 批 |
| 5 | smoke test (a):aux 关闭时逐项 loss 与 tools/train.py 一致到 A0 自身可复现精度(step0 逐位一致;全程统计一致,因 GPU FP 非确定性 A0-vs-A0 亦发散,见阶段0报告 §6a) |

---

## 9. 超参数清单

| 参数 | 值 | 说明 |
|---|---|---|
| `lambda_cls` | **0.1** | 唯一新增变量,阶段 0 固定不调参 |
| `feat_idx` | 2 (P3) | 辅助头输入特征层(neck 返回序 [P5,P4,P3] 中 P3) |
| `aux_resize` | 320 | cls 流输入分辨率(640 的 1/4 FLOPs) |
| det bs / cls bs | **16 / 16**(正式) | 每优化步 1 det + 1 cls;**与 A0 实际运行一致**(修正注记见 §5.1) |
| base_lr | **0.002**(正式) | 与 A0 实际运行一致(线性缩放 0.001×16/8);smoke 用配置默认 |
| seed | 0 | 与 A0 一致(`set_random_seed(0)`) |
| epoch | 100(正式)/ 2(smoke) | smoke 用 2e 配置 |
| 其余 | 全部继承 A0 | backbone/neck/head/loss_weight/EMA/增强/LR |

---

## 10. 实现文件清单(阶段 0)

| 文件 | 类型 | 说明 |
|---|---|---|
| `PaddleDetection/ppdet/modeling/heads/cgacs_aux_cls_head.py` | 新增 | CGACSAuxClsHead(workspace 注册,默认关闭) |
| `PaddleDetection/ppdet/modeling/architectures/yolo.py` | 修改(加性) | 可选 aux_cls_head 钩子 + aux_mode 分流,默认 None |
| `scripts/cgacs/pv_cls_dataset.py` | 新增 | PVClsDataset(paddle 原生 Dataset,只读 PV_CORE) |
| `scripts/train_cgacs.py` | 新增 | 双流梯度累积训练脚本(可 `--aux-off` 复现 A0) |
| `configs/ppyoloe_plus_crn_s_2e_cgacs_smoke.yml` | 新增 | S1 smoke 配置(aux head on) |
| `experiments/direction5/pv_cls_index.csv` | 生成 | PV_CORE 路径+class_id 索引 |

> 所有修改均为**加性/默认关闭**,`git diff` 可追溯;不改动任何历史实验与数据集。

---

## 11. 预期收益与成功判据(正式阶段)

- 分类监督规模扩大 **20 倍**(915 → 915+19233),缓解 P2;
- 图像级语义判别缓解 P1 类间混淆;
- 成功判据(正式训练后):VAL mAP@0.5:0.95 **≥ 0.4062** 且高于 A0;并评估每类 AP 的混淆改善。
- 阶段 0 成功判据 = 门槛 1~4 全 PASS + smoke test (a)~(g) 全 PASS(见 `CG_ACS_STAGE0_REPORT.md`)。
