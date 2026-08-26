# B1 PV_CORE 纯预训练两阶段实验最终报告 (B1_PVPRETRAIN_FINAL_REPORT)

> **方向**:B1 — PV_CORE 分类预训练 backbone → PlantDoc 检测微调 (路线 A, PV 方向最后一次验证)
> **阶段**:两阶段全部完成 (PV 分类预训练 200 epoch → PP-YOLOE+-s 检测微调 100 epoch)
> **日期**:2026-08-17 完成
> **结论**:❌ **未达标 (FAIL)**。B1 best VAL mAP@0.5:0.95 = **0.211**,远低于成功判据 **0.4100**,且远低于 A0 对照(同数据 exiffix VAL 上 A0 best = 0.410)。**PV_CORE 分类预训练作为 backbone 初始化来源不仅未带来增益,反而造成检测性能大幅退化。按既定判据,正式判定 PV_CORE 预训练方向失败,立即停止 PV 方向。**

---

## 0. 一句话结论

将 PV_CORE 19233 张图像级分类预训练的 backbone 作为 PP-YOLOE+-s 检测微调初始化后,VAL mAP@0.5:0.95 从 A0 的 **0.410** 降至 **0.211**(Δ −0.199);**13 类 AP 与 13 类 Recall 全部下降**,Recall 从 0.371→0.151,F1 从 0.444→0.232,密集场景 Recall 从 0.129→0.000。图像级分类任务学到的特征与实例级检测任务所需特征存在系统性不匹配,预训练形成负迁移。

---

## 1. 实验设计 (两阶段方案)

| 项 | 内容 |
|---|---|
| 阶段 1 | 仅用 PV_CORE 19233 张分类图片做 13 类图像分类预训练 (90/10 分层 → train 17310 / val 1923),backbone 从零随机初始化 |
| 阶段 1 产物 | 仅提取与 PP-YOLOE+-s 兼容的 backbone 预训练权重 (best_backbone.pdparams, 199 键裸键) |
| 阶段 2 | 与 A0 完全一致的 PlantDoc EXIF-fix 检测训练配置,100 epoch,不使用任何 PV 分类 loss、不冻结 backbone、不加辅助头、不改网络结构 |
| 唯一变量 | backbone 初始化来源 (A0=obj365 backbone, B1=PV_CORE 预训练 backbone);neck/head 均保持 obj365 原值 |
| 预训练权重 | 合并文件 = obj365 neck+head + PV backbone (详见 §3) |

**严格对照**:A0 = `experiments/baseline_v1` (PP-YOLOE+-s, obj365 预训练,同数据集同命令同超参)。除 pretrain_weights 路径外,阶段 2 全部训练条件与 A0 逐字一致(§2.2)。

---

## 2. 训练记录

### 2.1 阶段 1 — PV_CORE 分类预训练

```bash
cd /root/autodl-tmp/AIC2026_AgriVision
python -u scripts/b1/pv_cls_pretrain.py \
  --index_csv experiments/direction5/pv_cls_index.csv \
  --out_dir experiments/direction5/B1_PVPRETRAIN_100e/pretrain \
  --epochs 200 --batch 256 --lr 0.1 --warmup 5 \
  --resize 320 --workers 8 --seed 0 --label_smooth 0.1 --val_every 1
```

- 模型:`CSPResNet(**BB_KWARGS)`(与 PP-YOLOE+-s backbone 完全同构:layers=[3,6,6,3], channels=[64,128,256,512,1024], use_large_stem, use_alpha, width_mult=0.50, depth_mult=0.33)+ GAP + Linear(512,13),参数量 3,007,603。
- 数据:PV_CORE 19233 → 分层划分 train 17310 / val 1923;输入 320×320 /255 (BGR),与检测管线 NormalizeImage 一致。
- 优化:SGD momentum 0.9, wd 1e-4, LinearWarmup(5 epoch)+ Cosine, base_lr 0.1, CrossEntropy(label_smoothing=0.1)。
- 结果:**200 epoch 完成,总耗时 3618s;best val_acc = 0.9984 @ epoch 158**,final val_acc 0.9969。train_loss 2.08→0.538,收敛充分。全程 loss 有限,无 NaN。
- 关键工程修复:`cv2.setNumThreads(1)`(默认 128 线程导致每张 resize ~3ms,设 1 后 ~0.3ms),单 epoch 构建从 86s → 3.9s,整 epoch ~18s。

### 2.2 阶段 2 — 检测微调 (与 A0 命令逐字对齐)

```bash
cd /root/autodl-tmp/AIC2026_AgriVision/PaddleDetection
python -u tools/train.py \
  -c ../configs/ppyoloe_plus_crn_s_100e_b1.yml \
  --eval --use_vdl=true \
  --vdl_log_dir=../experiments/direction5/B1_PVPRETRAIN_100e/logs \
  -o save_dir=../experiments/direction5/B1_PVPRETRAIN_100e/checkpoints \
     TrainReader.batch_size=16 \
     LearningRate.base_lr=0.002
```

| 条件 | A0 | B1 | 一致性 |
|---|---|---|---|
| 配置文件 | `ppyoloe_plus_crn_s_100e_agrivision.yml` | `ppyoloe_plus_crn_s_100e_b1.yml` | 仅 pretrain_weights / weights 路径不同 (md5 记录于 DRIFT_GUARD) |
| pretrain_weights | obj365 URL | 本地合并文件 (obj365 neck/head + PV backbone) | **唯一变量** |
| 模型 | PP-YOLOE+-s | 同 | ✅ |
| epoch / snapshot / eval | 100 / 5 / 每5epoch | 同 | ✅ |
| batch_size / base_lr | 16 / 0.002 | 同 | ✅ |
| 数据集 | processed_detection (→ exiffix) | 同 (当前软链指向 exiffix) | ✅ |
| EMA | 0.9998 (默认) | 同 | ✅ |
| 数据增强 | 默认 | 同 | ✅ |

- 2-epoch smoke test 先行通过:合并权重完整加载 (仅 yolo_head.pred_cls 80→13 shape 不匹配自动 reinit,与 A0 相同),loss 下降无 NaN,显存峰值 10.6GB/24GB 无 OOM,VAL 评估管线正常 (113 样本),TEST 零访问。
- 正式 100 epoch:全程 loss 有限无 NaN 无 OOM (显存峰值 ~10.7GB);总耗时约 2.5h。B1 训练期 val mAP 序列见 §3.2。
- **阶段 2 全程未做任何调参** (需求合规)。

---

## 3. 兼容性检查 / 泄漏检查 / 漂移防护

### 3.1 backbone 兼容性 (compat_check.py → PASS)

`compat_report.json`:`CSPResNet(**BB_KWARGS)` 直接构造的分类 backbone 199 键与检测模型 backbone 199 键 **逐键完全一致 (shape 不匹配 0)**。obj365 backbone 缺 6 个 use_alpha 参数 (默认初始化为 1.0,A0 与 B1 相同)。分类 head 输入通道 = P5 = 512。

### 3.2 数据泄漏 (leakage_check.json → PASS)

PV_CORE 19233 vs PlantDoc 1142:**0 个 md5 命中,0 个 basename 命中**,无内容重叠。

### 3.3 防漂移复核 (DRIFT_GUARD.json → 21/21 PASS)

训练前记录 21 个关键文件 md5 (代码 5 + 配置 3 + 数据 4 + 预训练产物 7 + 合并权重 1 + A0 参照 2);训练后复核 **21 项全部一致,0 漂移**。A0 best/final 检查点 md5 与 S1 阶段钉死值完全一致 (best `1b4f669b…`, final `d76b6070…`),A0 冻结未变。

---

## 4. 统一 VAL 评估结果 (B1 vs A0, 同一 exiffix VAL, 113 张)

评估口径与 S1 报告完全一致:`tools/eval.py --classwise`(标准 COCO 指标)+ `analyze_val.py`(IoU=0.5 贪心匹配, conf=0.5 实际工作点)。

> 注:A0 在**当前 exiffix VAL** 上统一重评 = **0.410**(baseline_v1 训练日志记录 clean-VAL 0.408,差异来自 EXIF 修复的 1 张 val 图);B1 与 A0 在完全相同的数据上对比。

### 4.1 标准 COCO 指标 (tools/eval.py)

| 模型 | mAP@0.5:0.95 | mAP@0.5 | mAP@0.75 | AR@100 | 推理 FPS |
|---|---|---|---|---|---|
| **B1 best** (@epoch~99) | **0.211** | 0.326 | 0.227 | 0.612 | 19.34 |
| B1 final (@epoch100) | 0.211 | 0.326 | 0.227 | 0.612 | 18.81 |
| **A0 best** (@epoch~49) | **0.410** | 0.576 | 0.470 | 0.718 | 19.21 |
| A0 final (@epoch100) | 0.389 | 0.551 | 0.443 | 0.704 | 18.69 |

**B1 全程 val mAP 序列** (每5 epoch):`[0.048,0.068,0.079,0.118,0.136,0.143,0.153,0.162,0.177,0.180,0.182,0.185,0.194,0.198,0.198,0.202,0.209,0.210,0.211,0.211]`
**A0 val mAP 序列** (同口径):`[0.158,0.243,0.269,0.301,0.322,0.343,0.366,0.368,0.372,0.391,0.399,0.399,0.408,0.399,0.398,0.400,0.399,0.397,0.389,0.388]`

**B1 从第 1 次评估到第 100 epoch 全程落后 A0 0.11–0.24,差距从未收窄,收敛极慢且峰值极低。**

### 4.2 13 类 AP (tools/eval.py 标准口径)

| 类别 | A0 best | B1 best | Δ |
|---|---|---|---|
| Tomato Early blight leaf | 0.236 | 0.125 | −0.111 |
| Tomato Septoria leaf spot | 0.448 | 0.182 | −0.266 |
| Tomato leaf | 0.224 | 0.189 | −0.035 |
| Tomato leaf bacterial spot | 0.269 | 0.064 | −0.205 |
| Tomato leaf late blight | 0.438 | 0.255 | −0.183 |
| Tomato leaf mosaic virus | 0.247 | 0.074 | −0.173 |
| Tomato leaf yellow virus | 0.192 | 0.058 | −0.134 |
| Tomato mold leaf | 0.160 | 0.098 | −0.062 |
| Apple Scab Leaf | 0.540 | 0.091 | **−0.449** |
| Apple leaf | 0.787 | 0.673 | −0.114 |
| Apple rust leaf | 0.658 | 0.479 | −0.179 |
| grape leaf | 0.712 | 0.343 | **−0.369** |
| grape leaf black rot | 0.415 | 0.108 | **−0.307** |

**13/13 类 AP 全部下降,无一类提升。** 最强类 (Apple Scab −0.449, grape leaf −0.369, grape rot −0.307) 塌陷最严重。

### 4.3 P / R / F1 (analyze_val.py, IoU=0.5, conf=0.5)

| 模型 | Precision | Recall | F1 | tp/fp/fn |
|---|---|---|---|---|
| B1 best | 0.495 | **0.151** | **0.232** | 53/54/297 |
| B1 final | 0.495 | 0.151 | 0.232 | 53/54/297 |
| A0 best | 0.551 | 0.371 | 0.444 | 130/106/220 |
| A0 final | 0.480 | 0.411 | 0.443 | 144/156/206 |

**B1 仅召回 53/350 GT (A0 为 130/350),Recall 下降 0.22,F1 下降 0.21。** Precision 相对跌幅较小 (0.495 vs 0.551),说明 B1 的问题是"漏检"而非"误检"——对实例的定位/检出能力系统性不足。

### 4.4 每类 Recall @conf0.5 (B1 best vs A0 best)

| 类别 | B1 | A0 | Δ | 类别 | B1 | A0 | Δ |
|---|---|---|---|---|---|---|---|
| Early blight | 0.048 | 0.190 | −0.143 | Mosaic | 0.000 | 0.182 | −0.182 |
| Septoria | 0.216 | 0.568 | **−0.351** | Yellow virus | 0.000 | 0.088 | −0.088 |
| Tomato leaf | 0.194 | 0.333 | −0.139 | Mold | 0.062 | 0.156 | −0.094 |
| Bact spot | 0.000 | 0.188 | −0.188 | Apple Scab | 0.033 | 0.467 | **−0.433** |
| Late blight | 0.308 | 0.577 | −0.269 | Apple leaf | 0.714 | 0.929 | −0.214 |
| — | — | — | — | Apple rust | 0.583 | 0.667 | −0.083 |
| grape leaf | 0.429 | 0.857 | **−0.429** | grape rot | 0.136 | 0.545 | **−0.409** |

**B1 有 4 类 recall 为 0 (Bact spot, Mosaic, Yellow virus + 密集场景),所有 13 类 recall 均下降。** 与 A0 的主要混淆集中在类内相近对不同:Apple Scab→Septoria (12)、Early blight→Septoria (8)、Bact spot→Septoria (8)、Mosaic→Yellow virus (8)。

### 4.5 密集场景 Recall (VAL 5 张 ≥10 GT 图, 62 GT)

| 模型 | 命中 GT | 密集 Recall |
|---|---|---|
| **B1 best** | **0** | **0.000** |
| B1 final | 0 | 0.000 |
| A0 best | 8 | 0.129 |
| A0 final | 9 | 0.145 |

**B1 在全部 5 张密集场景图上 0 命中** (A0 命中 8/62)。密集场景是检测能力最直接的体现,B1 的实例级定位能力在密集场景完全失效。

### 4.6 参数量 / 推理开销

- B1 与 A0 架构完全相同:**7,700,106 参数** (429 键);S1 报告的 7,701,367 = 本架构 + aux 头 1,261。
- 推理 FPS 与 A0 基本持平 (B1 19.34 vs A0 19.21),预训练不影响推理开销。

---

## 5. 成功判据核验 (需求 10)

| 判据 | 目标 | 实测 | 判定 |
|---|---|---|---|
| **B1 best VAL mAP@0.5:0.95** | **≥ 0.4100** | **0.211** | ❌ **远未达成** (Δ−0.199 vs 判据; −0.199 vs A0) |
| 13 类 AP | 应不低于 A0 | **13/13 全部下降** | ❌ |
| Recall / F1 | 不明显退化 | Recall 0.371→0.151, F1 0.444→0.232 | ❌ |
| 密集场景 Recall | 不明显退化 | 0.129→0.000 | ❌ |

**主判据与全部辅助判据均未通过 → PV_CORE 预训练方向正式判定失败。**

按既定决策规则:成功判据固定为 B1 ≥ 0.4100;B1 = 0.211 < 0.4100 → **PV_CORE 作为当前 backbone 预训练源未带来有效增益,正式判定失败,立即停止 PV 方向,不再继续训练**。严格禁止继续发展 S1 或设计新的 PV 损失/结构。

---

## 6. 诊断分析 (基于 VAL, 未访问 TEST)

1. **负迁移 (Negative Transfer)**:PV_CORE 分类预训练让 backbone 学会了"整图判类"所需的全局统计特征,而 PP-YOLOE+-s 检测需要的是实例级、空间精确的特征。分类任务不提供定位/边界信息,导致 backbone 特征对检测的适用性显著下降。这与 S1 的结论一致——PV_CORE 图像级监督与 PlantDoc 检测任务存在系统性不匹配。
2. **收敛极慢且峰值极低**:B1 在 epoch 100 时 mAP 仅 0.211 且仍在缓升 (序列末 3 值 0.209/0.210/0.211),而 A0 在 epoch 49 已达 0.408 峰值。B1 的 100 epoch 甚至可能未达到其自身的收敛上限,但即便延长训练,其爬升速度也表明难以接近 A0。
3. **Recall 全面塌陷 = 定位能力缺失**:B1 的 Precision (0.495) 接近 A0 (0.551),但 Recall 仅为 A0 的 41% (0.151 vs 0.371),且 4 类 recall=0、密集场景 0 命中。分类预训练产生的特征对"哪里有目标"的响应严重不足。
4. **强类塌陷最严重**:Apple Scab / grape leaf / grape rot 这三类 A0 中 AP 最高、纹理边界最典型的类,B1 下降最大 (Δ0.31–0.45),进一步说明分类特征丢失了实例级纹理/边界信息。
5. **排除的干扰因素**:泄漏检查 PASS (数据无重叠);兼容性检查 PASS (层名/shape 100% 匹配);漂移防护 21/21 PASS;阶段 2 未调参、未冻结 backbone、未加辅助头。唯一变量确为 backbone 初始化来源,差异全部归因于 PV_CORE 预训练。

---

## 7. 归档清单

`experiments/direction5/B1_PVPRETRAIN_100e/`:

| 内容 | 位置 |
|---|---|
| 最终报告 | `B1_PVPRETRAIN_FINAL_REPORT.md`(本文件) |
| 防漂移哈希 | `DRIFT_GUARD.json`(21 文件,训练后复核 0 漂移) |
| 兼容性报告 | `compat_report.json`(backbone 199/199 键 shape 全匹配) |
| 泄漏检查 | `leakage_check.json`(0 命中) |
| 分类预训练产物 | `pretrain/`(cfg.json、train/val.csv、train.log、val_history.json、best/final_full.pdparams、best/final_backbone.pdparams、run_pretrain.out) |
| 合并权重 | `merge/B1_pretrain_combined.pdparams`(+ `.json` 元数据) |
| 检测微调日志 | `train_finetune.out`(100 epoch 全部步 + 20 次 VAL) |
| 检测检查点 | `checkpoints/`(best_model + model_final + 每5epoch,共 21 组) |
| Smoke 记录 | `smoke/run_smoke.out`、`smoke/checkpoints/` |
| 统一评估 | `eval/`(a0/b1 best/final 的 `.log`、`*_metrics` 相关、`*_analyze.json`、`*_c50.json`、`per_class_ap.json`) |
| 代码快照 | `scripts/b1/`(pv_cls_pretrain.py、merge_pv_backbone.py、compat_check.py、snapshot_guard.py、analyze_val.py) |
| 配置快照 | `configs/ppyoloe_plus_crn_s_100e_b1.yml`、`ppyoloe_plus_crn_s_100e_b1_smoke.yml` |
| 环境 / git | PaddleDetection HEAD `b25522a0`,paddle 2.6.2,RTX 4090 24GB |

---

## 8. 状态

- ✅ 两阶段实验全部完成,负结果如实报告,归档完整,漂移复核 21/21 PASS。
- ✅ **按需求:未访问 TEST,TEST 全程零访问。**
- ⏸ 等待用户确认后再决定是否对 TEST 做最终评估。
