# S1 CG-ACS 正式训练最终报告 (S1_CG_ACS_FINAL_REPORT)

> **方向**:S1 — PP-YOLOE+-s + PV_CORE 图像级分类辅助监督 (CG-ACS)
> **阶段**:100 epoch 正式训练(唯一新增变量 `lambda_cls=0.1`)
> **日期**:2026-08-16 训练完成
> **结论**:❌ **未达标 (FAIL)**。S1 best VAL mAP@0.5:0.95 = **0.3668**,低于成功判据 **0.4062**,且低于 A0 对照(best 0.408)。诚实报告,不做任何调参、不选择性汇报。

---

## 0. 一句话结论

加入 PV_CORE 图像级分类辅助监督(λ=0.1)后,VAL mAP@0.5:0.95 较 A0 **下降约 0.041**(0.3668 vs 0.408);辅助监督未带来正收益,反而引起强类(grape leaf、Apple rust)AP 回归和 Apple Scab→Apple rust 混淆显著恶化。**方向 S1 判定失败。**

---

## 1. 训练记录

### 1.1 训练命令(与 A0 一致,唯一新增 λ)

```bash
cd /root/autodl-tmp/AIC2026_AgriVision
python -u scripts/train_cgacs.py -c configs/ppyoloe_plus_crn_s_100e_cgacs.yml \
  --epoch 100 --det_batch 16 --cls_batch 16 --snapshot_epoch 5 --eval \
  --cls_stream 1 --cls_workers 4 --seed 0 \
  --save_dir experiments/direction5/S1_CG_ACS_100e/checkpoints \
  --cls_index experiments/direction5/pv_cls_index.csv \
  --log_file experiments/direction5/S1_CG_ACS_100e/train.log \
  --vdl_log_dir experiments/direction5/S1_CG_ACS_100e/logs \
  -o LearningRate.base_lr=0.002
```

### 1.2 训练条件对照(需求 2:除 λ 外零改动)

| 条件 | A0 | S1 | 一致性 |
|---|---|---|---|
| 模型 | PP-YOLOE+-s (Objects365 预训练) | 同 | ✅ |
| 检测 batch | 16 | 16 | ✅ |
| cls 流 batch | — | 16 (仅 PV_CORE 分类) | 新增(需求 3) |
| base_lr | 0.002 | 0.002 | ✅ |
| epoch / snapshot | 100 / 5 | 100 / 5 | ✅ |
| EMA | 0.9998 | 0.9998 | ✅ |
| seed | 未固定(同 A0 实际运行) | 0(仅双流随机性,非新增精度来源) | ✅ |
| 数据增强 / 检测数据 | 同 A0 配置 | 同 | ✅ |
| 辅助监督 | — | `lambda_cls=0.1` 固定(需求 2,中途零改动) | **唯一新增变量** |
| PV_CORE 用途 | — | 仅图像级分类 loss,无伪 bbox、无 det loss(需求 3) | 需求 3 |

### 1.3 训练运行状态

- 总步数 5700(det 57 steps/epoch × 100),det loader 实测 batch_size=16、steps/epoch=57(防漂移断言通过)。
- 全部步 `finite=True`,**无 NaN / 无 OOM / 无中断**(GPU 峰值 ~11.5GB / 24GB)。
- 每 5 epoch 用 EMA 权重在 VAL(113 张)评估,best_model 按 mAP@0.5:0.95 选择。

### 1.4 训练后防漂移复核(需求 7,全 PASS)

| 项 | 训练前记录 (DRIFT_GUARD.json) | 训练后复核 | 结果 |
|---|---|---|---|
| S1 配置 md5 | `0f9d3b2a…` | `0f9d3b2a…` | ✅ 一致 |
| A0 配置 md5 | `2bae51ad…` | `2bae51ad…` | ✅ 一致 |
| PV_CORE index md5 | `b2188b09…` | `b2188b09…` | ✅ 一致 |
| train_cgacs.py | `7a1e8b99…` | `7a1e8b99…` | ✅ 一致 |
| pv_cls_dataset.py | `b87c818b…` | `b87c818b…` | ✅ 一致 |
| cgacs_aux_cls_head.py | `6be02d35…` | `6be02d35…` | ✅ 一致 |
| yolo.py | `92944db4…` | `92944db4…` | ✅ 一致 |
| A0 best_model / model_final | `1b4f669b…` / `d76b6070…` | 同 | ✅ A0 冻结未变 |

训练条件中途未做任何修改(需求 2 合规)。

---

## 2. 统一评估结果(需求 9)

评估口径:统一 `tools/eval.py --classwise` 于 VAL(113 张),A0 与 S1 同一脚本同一配置;另用 `analyze_val.py`(IoU=0.5 贪心匹配,conf=0.5 实际工作点)算 P/R/F1、混淆、密集场景。

### 2.1 标准 COCO 指标 (tools/eval.py)

| 模型 | mAP@0.5:0.95 | mAP@0.5 | mAP@0.75 | AR@100 | 推理 FPS |
|---|---|---|---|---|---|
| **S1 best** (@epoch60) | **0.367** | 0.502 | 0.416 | 0.687 | 18.52 |
| S1 final (@epoch100) | 0.354 | 0.498 | 0.403 | 0.660 | 18.42 |
| **A0 best** (@epoch65) | **0.410** | 0.576 | 0.470 | 0.718 | 19.14 |
| A0 final (@epoch100) | 0.389 | 0.551 | 0.443 | 0.690 | 18.83 |

训练期 VAL 序列(S1 vs A0,每 5 epoch):S1 `[0.143,0.205,0.269,0.302,0.322,0.338,0.337,0.341,0.346,0.352,0.354,0.367,0.365,0.365,0.364,0.357,0.359,0.361,0.357,0.354]`;A0 `[0.158,0.243,0.269,0.301,0.322,0.343,0.366,0.368,0.372,0.391,0.399,0.399,0.408,0.399,0.398,0.400,0.399,0.397,0.389,0.388]`。**S1 全程落后 A0 ≈0.03–0.04,差距从未收窄;S1 峰值在 epoch60 后持续退化,与 A0 相同的过拟合趋势但更深。**

### 2.2 13 类 AP (tools/eval.py,标准口径)

| 类别 | A0 best | S1 best | Δ |
|---|---|---|---|
| Tomato Early blight leaf | 0.236 | 0.220 | −0.016 |
| Tomato Septoria leaf spot | 0.448 | 0.398 | −0.050 |
| Tomato leaf | 0.224 | 0.175 | −0.049 |
| Tomato leaf bacterial spot | 0.269 | 0.194 | −0.075 |
| Tomato leaf late blight | 0.438 | 0.348 | −0.090 |
| Tomato leaf mosaic virus | 0.247 | 0.198 | −0.049 |
| Tomato leaf yellow virus | 0.192 | 0.193 | +0.001 |
| Tomato mold leaf | 0.160 | 0.141 | −0.019 |
| Apple Scab Leaf | 0.540 | 0.421 | −0.119 |
| **Apple leaf** | 0.787 | **0.833** | **+0.046** ✅ |
| **Apple rust leaf** | 0.658 | **0.594** | **−0.064** ❌ |
| **grape leaf** | 0.712 | **0.658** | **−0.054** ❌ |
| grape leaf black rot | 0.415 | 0.395 | −0.020 |

**11/13 类 AP 下降;唯一明显提升是 Apple leaf(+0.046);两类强类回归超预算。**

### 2.3 P / R / F1 (analyze_val.py,IoU=0.5,conf=0.5)

| 模型 | Precision | Recall | F1 |
|---|---|---|---|
| S1 best | 0.476 | 0.340 | **0.397** |
| S1 final | 0.421 | 0.383 | 0.401 |
| A0 best | 0.551 | 0.371 | **0.444** |
| A0 final | 0.480 | 0.411 | 0.443 |

S1 best 在**更少预测框**(250 vs 236 接近,但 tp 119 vs 130)下 Precision、Recall、F1 全面低于 A0 best。

### 2.4 每类 Recall @conf0.5 (S1 best vs A0 best)

| 类别 | S1 | A0 | Δ | 类别 | S1 | A0 | Δ |
|---|---|---|---|---|---|---|---|
| Early blight | 0.333 | 0.190 | +0.143 | Mosaic | 0.061 | 0.182 | −0.121 |
| Septoria | 0.514 | 0.568 | −0.054 | Yellow virus | 0.140 | 0.088 | +0.053 |
| Tomato leaf | 0.278 | 0.333 | −0.056 | Mold | 0.094 | 0.156 | −0.062 |
| Bact spot | 0.125 | 0.188 | −0.062 | Apple Scab | 0.267 | 0.467 | **−0.200** |
| Late blight | 0.462 | 0.577 | −0.115 | Apple leaf | 0.929 | 0.929 | 0.000 |
| — | — | — | — | Apple rust | 0.750 | 0.667 | +0.083 |
| grape leaf | 0.714 | 0.857 | −0.143 | grape rot | 0.727 | 0.545 | +0.182 |

**最严重退化:Apple Scab recall 从 0.467 → 0.267(−0.200),Mosaic −0.121,grape leaf −0.143。** S1 提升集中在少样本/难类(Early blight、grape rot、Apple rust、Yellow virus),但以强类 recall 为代价。

### 2.5 主要混淆对 (analyze_val.py,conf=0.5,IoU 贪心跨类)

| S1 best | 数 | A0 best | 数 |
|---|---|---|---|
| **Apple Scab → Apple rust** | **15** | Apple Scab → Apple rust | 3 |
| Early blight → Septoria | 4 | Bact spot → Septoria | 6 |
| Tomato leaf → Late blight | 4 | Tomato leaf → Late blight | 3 |
| Bact spot → Septoria | 4 | Mold → Late blight | 3 |

**关键:Apple Scab→Apple rust 混淆由 3 → 15,激增 5 倍,是 Apple Scab AP/recall 崩塌的直接来源。**

### 2.6 密集场景 Recall (VAL 5 张 ≥10 GT 图,62 GT)

| 模型 | 命中 GT | 密集 Recall |
|---|---|---|
| S1 best | 13 | 0.210 |
| S1 final | 16 | 0.258 |
| A0 best | 8 | 0.129 |
| A0 final | 9 | 0.145 |

**S1 在密集场景 recall 优于 A0(+0.081),是本实验唯一一致的正收益**(与每类 recall 中难类提升吻合)。

### 2.7 参数量 / 推理开销

- S1 总参数量 **7,702,628** = 检测主干 7,701,367 + aux 头 1,261(+0.016%);A0 = 7,701,367。
- 推理 FPS 与 A0 基本持平(aux 头推理时完全旁路,S1 best 18.52 vs A0 best 19.14,差异 <3%,属运行波动)。

---

## 3. 成功判据核验(需求 10)

| 判据 | 目标 | 实测 | 判定 |
|---|---|---|---|
| VAL mAP@0.5:0.95 | ≥ 0.4062 | **0.3668** | ❌ 未达成(且 < A0 0.408) |
| 主要混淆对改进 | 应改善 | Apple Scab→Apple rust **3→15 恶化** | ❌ |
| 强类 AP 回归 ≤ 0.03 | ≤ 0.03 | grape leaf −0.054、Apple rust −0.064、Apple Scab −0.119 | ❌ |
| Recall / F1 不明显退化 | 不明显 | F1 0.397 vs 0.444(降 0.047)、Recall 0.340 vs 0.371 | ❌ |

**主判据未达成,其余 3 项辅助判据亦未通过 → 方向 S1 失败。** 以下按诚实原则报告全部数字,不做任何调整。

---

## 4. 诊断分析(基于 VAL,未经任何 TEST)

1. **辅助监督的"平均化"副作用**:图像级分类监督将 logits 向类别的全局统计平均方向挤压,损害了对边界/纹理细节敏感的高 AP 类(Apple Scab、grape leaf、Apple rust)。这与 Apple Scab→Apple rust 混淆 3→15 一致——aux 头学到的类间相似度放大了这两个类在检测头中的可分性压力。
2. **监督尺度不匹配**:PV_CORE 是独立真实采集的图像级标注(19233 张),与 PlantDoc 拍照条件/分布存在差异(Phase-1 审计仅保证"无内容重叠",不保证"分布一致")。将整图类别当作监督信号,弱化了检测所需的实例级区分。
3. **λ=0.1 扰动过大**:det 流附加 0.1×BCE ≈ 6.9% 训练损失权重,却未获得定位信息;增益(难类 recall、密集场景)无法抵消强类 AP 与精确率损失。
4. **后期退化更深**:epoch60 后 S1 mAP 从 0.3668 → 0.3538(−0.013),A0 从 0.408 → 0.388(−0.020);两者均过拟合,但 S1 峰值更低、波动更大,aux 头未提供正则化保护。

---

## 5. 归档清单(需求 6)

`experiments/direction5/S1_CG_ACS_100e/`:

| 内容 | 位置 |
|---|---|
| 最终报告 | `S1_CG_ACS_FINAL_REPORT.md`(本文件) |
| 防漂移哈希记录 | `DRIFT_GUARD.json` |
| 完整归档(代码快照/配置/日志/指标) | `archive/`(2.9MB:train_cgacs.py、pv_cls_dataset.py、analyze_val.py、yolo.py、cgacs_aux_cls_head.py、A0/S1 配置、pv_cls_index.csv、train.log、run.out、eval 全部 .log/.json、ENV_GIT.txt、TRAIN_COMMAND.txt) |
| 训练日志 | `train.log`(含全部 5700 步 + 20 次 VAL + best 选择) |
| 运行输出 | `run.out` |
| 标准评估结果 | `eval/s1_best.log / s1_final.log / a0_best.log / a0_final.log` + `*_metrics.json`、`*_metrics_c50.json` |
| Checkpoints | `checkpoints/`(epoch5–95 每 5 张 + best_model + model_final,共 21 组,1.9GB;best_model 含 EMA 权重) |
| VisualDL 日志 | `logs/vdlrecords.1786881815.log` |
| 环境 / git | `archive/ENV_GIT.txt`(PaddleDetection HEAD `b25522a0`,paddle 2.6.2,RTX 4090) |

> git 说明:PaddleDetection 工作树含全库行尾归一化引入的 2011 文件改动(与本次工作无关,非本项目产生);本项目全部改动文件已按 md5 快照与副本入档,归档不依赖工作树 git diff。

---

## 6. 下一步建议(仅供决策,未执行)

- **建议采用 A0(0.408)作为最终检测模型**;S1 未达标的负结果已完整归档。
- 若后续仍想探索辅助监督(需另行批准,本次不执行):λ 显著调小(如 0.01)、aux loss 改用 focal 降低易分类样本主导、在不同 feat 层(`feat_idx` 0/1)试验、或加 det 流硬负采样缓解 Apple Scab/Apple rust 混淆。以上均为新实验,不在本次"唯一新增变量 λ"约束内。

---

## 7. 状态

- ✅ 100 epoch 正式训练完成,负结果如实报告,归档完整,防漂移复核全 PASS。
- ✅ **按需求 12:未运行 TEST 最终评估,TEST 全程零访问。**
- ⏸ 等待用户确认后再决定是否对 TEST 做最终评估。
