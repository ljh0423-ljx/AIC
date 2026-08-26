# S1 CG-ACS 机制核验 (CG_ACS_MECHANISM_CHECK)

> 对 S1 CG-ACS 的三个关键机制承诺做静态+动态双重核验。
> 核验日期:2026-08-16 | 只读核验 + smoke test 证据
> 配套: `CG_ACS_DESIGN.md`(设计)、`CG_ACS_STAGE0_REPORT.md`(门槛与 smoke 汇总)、`smoke_logs/`(原始日志)

---

## 核验点 1:PV_CORE 仅作为"图像级分类监督"使用(门槛 3)

### 承诺
PV_CORE 的 19,233 张图**只提供图像级类别监督**;绝不生成伪 bbox、绝不把分类图当作检测图。

### 静态核验
| 检查项 | 结果 |
|---|---|
| S1 代码路径中 PV_CORE 样本的 `aux_mode='cls'` | `yolo.py` 训练分支:遇到 `aux_mode=='cls'` 时**完全跳过检测头**(不调用 `yolo_head`),仅执行 `backbone→neck→aux_cls_head` |
| PV_CORE 样本携带的监督字段 | 只有 `image` + `aux_cls_target`(one-hot 13 类)+ `aux_mode='cls'`;**无任何 gt_bbox / gt_class / is_crowd** |
| 是否生成/回传伪框 | `CG_ACS_DESIGN.md` §1.3 明确放弃伪框方案;代码中无任何 pseudo-bbox 生成逻辑 |
| 检测数据流与分类数据流的隔离 | det 流=PlantDoc COCO(exiffix),cls 流=PV_CORE 索引(`experiments/direction5/pv_cls_index.csv`),两流样本无交叉 |
| PV_CORE 索引只读性 | 索引只含**绝对路径**,训练仅 `cv2.imread` 读取;PV_CORE 原始目录零写入(见 Gate2/审计结论) |

### 动态核验(smoke test)
- `smoke_logs/s1_on.log`:cls 流日志 `cls:loss_aux_cls=...` 表明 PV_CORE batch 进入分类 loss 且**无任何 det 键**(loss_cls/loss_iou/loss_dfl 均不出现),证明分类图没有走检测头。
- `smoke_logs/a0_ref.log` 与 `smoke_logs/s1_on.log` 中 det 流键完整(5 键)。

**结论:✅ PV_CORE 仅作图像级分类监督;无伪 bbox;不把分类图当检测图。**

---

## 核验点 2:检测头 PPYOLOEHead 完全不变

### 承诺
S1 共享 backbone/neck,新增独立分类头;检测头参数、前向、assigner、loss 计算与 A0 逐字节一致。

### 静态核验
| 检查项 | 结果 |
|---|---|
| **S1 本次**是否修改 `ppyoloe_head.py` | **否**。mtime 00:12(早于 S1 开始 18:23),S1 的三个改动文件为 `yolo.py`(+32/-1,加性)、`heads/__init__.py`(+1)、`cgacs_aux_cls_head.py`(新增),见附录与 `git diff` |
| `ppyoloe_head.py` 的既有改动来源 | 该文件含 **direction2(CGPM)** 遗留的默认关闭参数(`use_cgpm=False` 等,2026-08-16 00:12 由 direction2 实验写入);**A0 基线配置 `ppyoloe_plus_crn_s_100e_agrivision.yml` 不引用 cgpm(rc=1,无命中)**,`use_cgpm` 默认 False,故 A0 检测头行为与此改动前完全一致 |
| 新增代码位置 | 仅 3 处:`cgacs_aux_cls_head.py`(新文件)、`yolo.py`(加性钩子,默认 `aux_cls_head=None`)、`heads/__init__.py`(一行 import) |
| 检测头训练路径 | 当 `aux_cls_head is None` 时 `_forward` 走 A0 原逻辑,输出逐键一致 |
| aux 头是否影响检测头内部 | aux loss 只作为额外 loss 项加入 `yolo_losses['loss']`;**不改检测头任何内部计算**(S1 未触碰 ppyoloe_head.py) |

### 动态核验(smoke test)
- **A0-off ≡ A0**:`s1_off.log`(S1 关闭,仅 det 流)与 `a0_ref.log`(tools/train.py 原版)每个日志点逐项对比:
  `loss / loss_cls / loss_iou / loss_dfl / loss_l1` 全部一致(见 `CG_ACS_STAGE0_REPORT.md` 表)。
  证明:未配置 aux 头时,训练前向/损失与 A0 完全相同。

**结论:✅ 检测头未改动;S1 关闭时与 A0 逐项一致。**

---

## 核验点 3:分类头仅训练、不参与推理(零推理开销)

### 承诺
aux 分类头只存在于训练路径;推理时被完全旁路,推理结构与 A0 一致,不增加 FPS 开销。

### 静态核验
`yolo.py _forward()` 中 `self.aux_cls_head` 仅在 `if self.training:` 分支内被访问(两处:cls 流分支、det 流 aux loss 分支);`else`(eval)分支完全不引用 aux head → 推理前向 = A0。

### 动态核验(smoke test)
- 用 S1 训练的 smoke 权重运行 `tools/eval.py`(配置仍含 `aux_cls_head`,但 eval 模式走推理分支),成功产出 val COCO 指标,输出结构与 A0 评估一致(见 `CG_ACS_STAGE0_REPORT.md` 验证 f)。
- 该运行中 `model.training=False`,aux 头未参与计算 → 推理 FPS 与 A0 相同。

**结论:✅ 分类头 train-only,推理旁路,零 FPS 开销。**

---

## 核验点 4:TEST 完全隔离(不得访问)

### 承诺
阶段 0 全程绝不读取 `annotations/test.json`(TEST 仅留给未来最终评估)。

### 静态核验
| 检查项 | 结果 |
|---|---|
| det 流配置 | `TrainDataset → annotations/train.json`、`EvalDataset → annotations/val.json`(A0 原配置,未改) |
| PV_CORE 索引 | 仅含 `PV_CORE_19233_audit/raw/color/...` 路径,与 PlantDoc/TEST 零交集 |
| 无 TestReader 引用 | 阶段 0 运行命令均不带 eval/test 数据集覆盖 |
| 评估脚本 | 本阶段仅对 **val** 评估;`eval_baseline.sh`(TEST 专用)未在本阶段调用 |

### 动态核验
- 对全部 smoke 日志与运行命令 `grep test.json` → 零命中(见 `CG_ACS_STAGE0_REPORT.md` 验证 g)。

**结论:✅ TEST 在阶段 0 完全未被访问。**

---

## 附录:git diff 清单(S1 阶段 0 全部改动)

| 文件 | 状态 | 内容 |
|---|---|---|
| `PaddleDetection/ppdet/modeling/heads/cgacs_aux_cls_head.py` | 新增 | CGACSAuxClsHead |
| `PaddleDetection/ppdet/modeling/architectures/yolo.py` | 修改(加性) | `aux_cls_head` 可选参数 + `_forward` 两处钩子 |
| `PaddleDetection/ppdet/modeling/heads/__init__.py` | 修改(加性) | 1 行 import |
| `scripts/cgacs/pv_cls_dataset.py` | 新增 | PVClsDataset / 索引生成 / 数值参考 BCE |
| `scripts/train_cgacs.py` | 新增 | 双流梯度累积训练脚本 |
| `configs/ppyoloe_plus_crn_s_2e_cgacs_smoke.yml` | 新增 | S1 smoke 配置 |
| `experiments/direction5/pv_cls_index.csv` | 生成 | PV_CORE 分类索引(19233 行) |

**git diff 佐证**(`git diff HEAD -w --ignore-blank-lines`,排除行尾/许可证噪声):
- `yolo.py`: +32/−1(仅 aux 钩子,默认 `aux_cls_head=None`);`heads/__init__.py`: +1 行 import;`cgacs_aux_cls_head.py`: 新增。
- `ppyoloe_head.py`: 存在 **direction2(CGPM)** 遗留改动(+143/−10,默认关闭参数),S1 未触碰;A0 配置不启用,行为不受影响。

> **S1 未改动**:任何历史实验目录、`processed_detection*` 数据、TEST 相关配置、`ppyoloe_head.py`(后者仅含 direction2 遗留的默认关闭 cgpm 参数,非 S1 所改,详见核验点 2)。
