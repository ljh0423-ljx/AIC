# S1 CG-ACS 阶段 0 门槛核验报告 (CG_ACS_STAGE0_REPORT)

> **阶段**:0(门槛核验 + 最小可行实现 + 1~2 epoch smoke test)
> **日期**:2026-08-16
> **结论**:门槛 1~4 全 PASS,smoke 验证 (a)~(g) 全 PASS。**不执行 100 epoch 正式训练,停在此处等待批准。**
> 配套: `CG_ACS_DESIGN.md`(设计)、`CG_ACS_MECHANISM_CHECK.md`(机制核验)、`smoke_logs/`(原始日志与权重)

---

## 0. 阶段 0 范围

- 只读核验 + 最小实现 + smoke test,**无正式训练**。
- 未触碰:任何历史实验目录、`processed_detection*` 数据、TEST 相关配置。

---

## 1. 门槛 1:P/D 双数据集 13 类映射与 class_id 一致性 ✅

**复验方式(2026-08-16 重新核验)**:PlantDoc COCO 标注 `categories` vs PV_CORE `file_manifest.csv` 的 `aic_category_name`。

| PlantDoc category_id (1-based) | PV_CORE aic_class_id (0-based) | 类别名 |
|---|---|---|
| 1 | 0 | Tomato Early blight leaf |
| 2 | 1 | Tomato Septoria leaf spot |
| 3 | 2 | Tomato leaf |
| 4 | 3 | Tomato leaf bacterial spot |
| 5 | 4 | Tomato leaf late blight |
| 6 | 5 | Tomato leaf mosaic virus |
| 7 | 6 | Tomato leaf yellow virus |
| 8 | 7 | Tomato mold leaf |
| 9 | 8 | Apple Scab Leaf |
| 10 | 9 | Apple leaf |
| 11 | 10 | Apple rust leaf |
| 12 | 11 | grape leaf |
| 13 | 12 | grape leaf black rot |

- **13/13 类别名字完全一一对应**(`category_id = aic_class_id + 1`);COCO 数据集中 clsid 为 0-based,与 PV_CORE `aic_class_id` **直接一致**。
- PV_CORE 19233 张:class 计数 {0:1000, 1:1771, 2:1591, 3:2127, 4:1909, 5:373, 6:5357, 7:952, 8:630, 9:1645, 10:275, 11:423, 12:1180}(与 `file_manifest.csv` 一致)。

**✅ class_id 完全一致。**

---

## 2. 门槛 2:PlantDoc ↔ PV_CORE 交叉去重(含 TEST 泄露检查)✅

**复验方式**:对 PlantDoc 全部 1142 张图(train 915 + val 113 + test 114)与 PV_CORE manifest 19233 行做全量比对。

| 检查项 | 结果 |
|---|---|
| 文件名(basename)交集 | **0** |
| MD5 交集(PlantDoc 图逐张 MD5 比对 PV_CORE md5 列) | **0** |
| 早前 dHash/pHash 近重复检查(本会话早前完成) | **0 命中**(PASS) |
| **PV_CORE ↔ TEST** 专项检查 | 0 命中(无任何 TEST 图与 PV_CORE 相同或近相同) |

**✅ PlantDoc train/val/test 与 PV_CORE 无任何内容重复/近重复,PV_CORE 与 TEST 零泄露。**

---

## 3. 门槛 3:PV_CORE 仅作"图像级分类监督"✅

- S1 代码路径中,PV_CORE 样本 `aux_mode='cls'` → `yolo.py` 训练分支**完全跳过检测头**,仅 `backbone→neck→aux_cls_head`。
- PV_CORE 样本监督字段只有 `image + aux_cls_target(one-hot) + aux_mode`;**无 gt_bbox / gt_class / is_crowd**。
- 不生成、不回传任何伪 bbox(设计 §1.3 明确放弃伪框方案;代码无 pseudo-bbox 逻辑)。
- 动态证据:`s1_on.log` cls 流每步只有 `cls:loss_aux_cls=0.07xx`,**无任何 det 键**。

**✅ PV_CORE 仅图像级分类监督;不生成伪 bbox;不把分类图当检测图。**

---

## 4. 门槛 4:A0 基线冻结 ✅

| 检查项 | 结果 |
|---|---|
| A0 配置 `ppyoloe_plus_crn_s_100e_agrivision.yml` | 未改(`git diff` 无命中 cgpm,rc=1) |
| S1 代码改动范围 | 仅 3 处,均**加性/默认关闭**:`yolo.py`(+32/−1,`aux_cls_head=None` 默认)、`heads/__init__.py`(+1)、`cgacs_aux_cls_head.py`(新增) |
| 检测头 `ppyoloe_head.py` | **S1 未触碰**(mtime 00:12 早于 S1 18:23);其内含 direction2(CGPM)遗留的默认关闭参数(`use_cgpm=False` 等),A0 配置不引用,行为不受影响 |
| 历史实验目录 | `baseline_v1`、`direction2`、`direction3`、`direction4`、`ablation_dir2_*` 均未改 |
| 数据 | `processed_detection*`、PV_CORE 原始目录零写入 |

**✅ A0 冻结;S1 改动全部加性可追溯(`git diff HEAD -w`)。**

---

## 5. S1 最小可行实现(阶段 0)

| 文件 | 说明 |
|---|---|
| `PaddleDetection/ppdet/modeling/heads/cgacs_aux_cls_head.py` | CGACSAuxClsHead:GAP→Flatten→Linear(96→13),λ=0.1,`feat_idx=2`(P3) |
| `PaddleDetection/ppdet/modeling/architectures/yolo.py` | 加性钩子:训练分支两处(aux_mode='cls' 纯分类 / det 流附加 λ·L_aux);推理分支完全不引用 aux head |
| `PaddleDetection/ppdet/modeling/heads/__init__.py` | +1 行 import |
| `scripts/cgacs/pv_cls_dataset.py` | PVClsDataset(paddle 原生,只读 PV_CORE)+ collate + 数值参考 BCE |
| `scripts/train_cgacs.py` | 双流梯度累积训练脚本(1 det + 1 cls / 优化步;可关闭 cls 流复现 A0) |
| `configs/ppyoloe_plus_crn_s_2e_cgacs_smoke.yml` | S1 smoke 配置(2 epoch) |

**参数量实测**:总 7,701,367,aux 头 **1,261**(+0.016%,可忽略)。检测头 PPYOLOEHead 结构零改动。

---

## 6. Smoke test 七项验证 (a)~(g)

### 运行清单(smoke_logs/)

| 运行 | 命令要点 | 产物 |
|---|---|---|
| Run 1 A0 参考 | `tools/train.py -c ppyoloe_plus_crn_s_2e_smoke.yml --enable_ce True -o log_iter=1 ...` | `a0_ref.log`(228 步) |
| Run 2 S1 关闭 | `scripts/train_cgacs.py -c ...smoke.yml --cls_stream 0 ...` | `s1_off.log`(228 步) |
| Run 3 S1 开启 | `scripts/train_cgacs.py -c ...cgacs_smoke.yml --cls_stream 1 --grad_check 1 ...` | `s1_on.log`(228 步) |
| A0 复现对照 | `tools/train.py` 同参数再跑两次 | `a0_t2.log`、`a0_t3.log` |
| 推理旁路评估 | `tools/eval.py -o weights=cgacs_final_model.pdparams`(配置仍含 aux_cls_head) | `s1_eval_val.log` |

### (a) S1 关闭时 loss 逐项与 A0 一致 ✅(统计一致性)

**step 0 逐位一致**:A0 vs S1-off 在 step 0 的 5 个 loss 分量均一致到 5 位小数(diff ≤ 4e-6,FP 噪声级);而 A0 自身两次运行(a0_ref vs a0_t2)**step 0 逐位一致(diff=0.00)**。

**逐位逐项一致不可能实现——A0 与自己都无法逐位复现**。三对 `tools/train.py` 同种子复跑均发散到 O(1):

| 比较对 | loss mean | med | p90 | max |
|---|---|---|---|---|
| a0_ref vs a0_t2(tools×tools) | 0.137 | 0.093 | 0.323 | 0.762 |
| a0_ref vs a0_t3(tools×tools) | 0.119 | 0.092 | 0.280 | 0.843 |
| a0_t2 vs a0_t3(tools×tools) | 0.131 | 0.085 | 0.339 | 0.636 |
| **a0_ref vs s1_off(tools×mine)** | **0.131** | **0.090** | **0.318** | 1.067 |

- 我的脚本(S1-off)与 A0 的发散分布(mean/med/p90)**完全落在 A0 自身 run-to-run 包络内**。
- 根源:GPU cuDNN/cuBLAS FP 非确定性,经优化轨迹混沌放大到 O(1)(step 0 逐位一致证明数据/模型/初始化对齐)。
- 逐 epoch 均值:A0 三次 {0:3.82~3.84, 1:3.49~3.51},S1-off {0:3.89, 1:3.47},同分布。

**✅ S1 关闭时与 A0 统计一致(可达性与 A0 自身一致;比 A0 自可比更为严格的"逐位"要求在本 GPU 上不成立,已用 A0-vs-A0 证明)。**

### (b) PV_CORE batch 进入分类 loss ✅

`s1_on.log` 每步 cls 流:`cls:loss=0.0746 | loss_aux_cls=0.0746`(λ·BCE,one-hot;≈0.1×0.75,符合随机初始化 logits 的 ln2≈0.693 量级)。**只出现 loss_aux_cls,无任何 det 键** → PV 图未进检测头。

### (c) PlantDoc batch 同时计算 det loss + cls loss ✅

`s1_on.log` step 0 det 流:
```
det:loss=4.77833 | loss_cls=2.98629 | loss_iou=0.36219 | loss_dfl=1.63431 | loss_l1=1.27569 | loss_aux_cls=0.06940 | det_total=4.77833
```
- 4 个检测分量与 A0 step 0 **逐位一致**(2.986294/0.362193/1.634314/1.275686);
- `loss_aux_cls=0.06940` = 0.1 × 0.694(multi-hot 目标 BCE),`det_total = A0_loss + λ·L_aux`(4.708932+0.06940=4.77833 ✓)。

### (d) 梯度回传 ✅

`[grad_check] step0 | aux_fc.grad L1=4.5524 | proj_conv.grad L1=None | backbone.grad L1=182.51`
- aux 头收到梯度(4.55)、backbone 收到梯度(182.51)→ 双流 backward 均正常回传。
- `proj_conv.grad=None` 是**设计使然**:PPYOLOEHead 中 `proj_conv.weight.stop_gradient=True`(预计算 DFL 固定投影,不学习),非失败。

### (e) 无 NaN/Inf/OOM ✅

`=== done: total_steps=228 | finite steps=228 | nan_occurred=False ===`;全部 228 步 `finite=True`;`.out` 无 Traceback/CUDA OOM(24GB)。

### (f) 分类头不进入推理(零 FPS 开销)✅

- 静态:`yolo.py` `else`(推理)分支完全不引用 `aux_cls_head` → 推理结构与 A0 逐字节一致。
- 动态:用 S1 训练权重(配置仍含 `aux_cls_head`)跑 `tools/eval.py` 在 **VAL(113 张)** 成功:
  - 输出标准 COCO bbox 指标表(AP@0.5:0.95=0.035,2-epoch smoke 权重数值低属预期);
  - 结构、键、后处理与 A0 评估一致;aux 头未参与计算。

### (g) TEST 完全隔离 ✅

| 检查 | 结果 |
|---|---|
| 全部 smoke 日志/输出 grep `test.json` | **0 命中** |
| `pv_cls_index.csv` 19233 行路径 | 全部在 `PV_CORE_19233_audit/raw/color/` 下(0 异常) |
| `train_cgacs.py`/`pv_cls_dataset.py`/smoke 配置 grep `test.json` | 0 命中 |
| 评估实际加载 | `annotations/val.json`(113 samples) |

---

## 7. 计算/显存开销实测(修正设计文档 §6)

训练前向实测(bs8,MAC 计数器):
| 项 | 值 |
|---|---|
| det 流 640 / 544 / 320 | 16.30 / 11.77 / 4.07 G-FLOP/img |
| cls 流 320(仅 backbone+neck+aux) | 3.50 G-FLOP/img |
| 每优化步增量 | +22%(相对 640)~ +30%(相对平均缩放) |
| 参数量 | +1,261(+0.016%) |

> 设计初稿的 "43.9 GFLOPs" 经实测修正为 16.30 G-FLOP/img@640(详见 `CG_ACS_DESIGN.md` §6 修正说明)。

---

## 8. 结论与下一步

- **门槛 1~4:全部 PASS**;smoke 验证 (a)~(g):全部 PASS。
- S1 CG-ACS 最小可行实现验证通过:PV_CORE 作为图像级分类监督源、检测头零改动、aux 头仅训练、推理旁路、TEST 零访问、无 NaN/OOM、开销可控。
- **按阶段 0 要求:此处停止,不执行 100 epoch 正式训练。**
- 下一步(已批准):以 `lambda_cls=0.1` 为唯一新增变量,运行 100 epoch 正式训练。训练条件与 A0 实际运行完全一致:det bs=**16**、base_lr=**0.002**、seed0、EMA、Objects365 预训练、epoch100、snapshot5、每 snapshot VAL 评估(2026-08-16 用户裁决,修正注记见 `CG_ACS_DESIGN.md` §5.1)。成功判据 VAL mAP@0.5:0.95 ≥ 0.4062 且高于 A0(以统一 VAL 评估复测 A0 best_model/model_final 为准)。
