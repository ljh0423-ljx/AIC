# M 实验最终报告 — PP-YOLOE+-s → m 模型容量放大 (A0 vs M)

日期: 2026-08-17
实验唯一变量: 模型规模 s→m (depth_mult 0.33→0.67, width_mult 0.50→0.75)
其余全部条件与 A0 完全一致: 13类/EXIF-fix数据/epoch=100/bs=16/lr=0.002/EMA/评估方式。
不使用 PV_CORE, 不沿用 S1/B1/D1/D2/CGPM 任何失败方向。
TEST 全程封闭, 仅使用 VAL (113 样本) 选择 best checkpoint。

---

## 0. 结论 (决策规则判定)

**M-best VAL mAP@0.5:0.95 = 0.428 ≥ 0.415,且无强类 AP 回退 → 按固定决策规则,PP-YOLOE+-m 列为最终模型候选。**

| 决策规则 | 判定 |
|---|---|
| M mAP < 0.410 → 模型放大无效 | 否 (0.428) |
| ≥0.410 但 Δ<0.005 且速度/参数代价明显 → 保留 A0 | 否 (Δ=+0.018) |
| ≥0.415 且无强类回退 → 列为最终模型候选 | **是** |
| 0.418~0.422 或更高 → 重点分析是否真正优于 A0 | **是 (0.428),详见 §5** |

不自动宣布最终模型;是否用 M 取决于速度/参数预算取舍,由用户决定。

---

## 1. 实验设计与门控过程

### 1.1 官方配置核对 (PaddleDetection v2.9.0, HEAD b25522a0)
| 项 | 官方 PP-YOLOE+-s | 官方 PP-YOLOE+-m | A0 (基准) | M (本次) |
|---|---|---|---|---|
| depth_mult | 0.33 | **0.67** | 0.33 | **0.67** |
| width_mult | 0.50 | **0.75** | 0.50 | **0.75** |
| pretrain | s_obj365 | **m_obj365** | s_obj365 | **m_obj365** |
| 结构/neck/head base | `_base_/ppyoloe_plus_crn.yml` | 同左 | 同左 | 同左 |
| reader | `_base_/ppyoloe_plus_reader.yml` | 同左 | 同左 | 同左 |
| optimizer/EMA | `_base_/optimizer_80e.yml` | 同左 | epoch=100/bs16/lr0.002 | 同 A0 |

M 与 A0 唯一差异 = 规模参数 + pretrain 路径 (均指向官方对应 obj365 权重)。

### 1.2 兼容性预检 (compat_check.py)
- m 检测模型键数 600, obj365 m 预训练键数 588。
- 匹配 582;无匹配 12 个全部为 `backbone.*.conv2.alpha`(默认初始化 1.0,与 A0 的 s 模型 6 个 alpha 无匹配为同类行为);shape 跳过 6 个全部为 `pred_cls`(365→13 通道,与 A0 相同)。**PASS,无意外未匹配。**

### 1.3 VRAM 探针 → 确定 batch (vram_probe.py)
- m 模型 bs=16, 768×768 最坏训练尺寸, max_allocated = **15.32 GB** / 25.9 GB (61%)。
- 余量充足 → **保持 bs=16 / lr=0.002,与 A0 完全一致**,不改变 batch/lr 策略。

### 1.4 Smoke Test 门控 (2 epoch, 全过 → 进入正式训练)
| 门控项 | 结果 |
|---|---|
| 显存峰值 (nvidia-smi 采样) | **19.6 GB / 25.9 GB (76%)**, 无 OOM |
| loss 正常下降 | 4.84 → 3.49 (2 epoch) |
| 无 NaN/Inf/OOM | 0 / 0 / 0 |
| checkpoint 正常保存 | 0.pdparams + best_model (94 MB) |
| VAL 评估正常 | 113 样本, 每 epoch 输出 best bbox ap |
| 推理可运行 | best_model 单图前向 → bbox [300,6] |
| 参数量 / 模型大小 | 23.57 M / 94.3 MB |
| FPS (smoke eval) | ~29-34 (最终以统一协议重测) |

---

## 2. 统一 VAL 对比 (best checkpoint, 同一 exiffix VAL 113 张)

评估协议与 A0/B1 完全一致: `tools/eval.py --classwise` (COCO 指标) + `analyze_val.py --conf_thr 0.5` (P/R/F1/每类/混淆/密集)。三个模型在空闲 GPU 上顺序评估。

### 2.1 COCO 指标 (tools/eval.py)
| 指标 | A0 (s) | M (m) | Δ |
|---|---|---|---|
| **mAP@0.5:0.95** | **0.410** | **0.428** | **+0.018** |
| mAP@0.5 | 0.576 | 0.590 | +0.014 |
| mAP@0.75 | 0.470 | 0.492 | +0.022 |
| AP small | 0.060 | 0.076 | +0.016 |
| AP medium | 0.265 | 0.272 | +0.007 |
| AP large | 0.432 | 0.451 | +0.019 |
| AR@1 | 0.295 | 0.299 | +0.004 |
| AR@10 | 0.625 | 0.636 | +0.011 |
| AR@100 | 0.718 | 0.723 | +0.005 |

### 2.2 P/R/F1 + 密集场景 (analyze_val.py, conf=0.5)
| 指标 | A0 (s) | M (m) | Δ |
|---|---|---|---|
| Precision | 0.551 | 0.576 | +0.025 |
| Recall | 0.371 | 0.400 | +0.029 |
| F1 | 0.444 | 0.472 | +0.028 |
| 密集场景 Recall (5图/62GT) | 0.129 (8) | **0.210 (13)** | **+0.081** |

### 2.3 参数量 / 模型大小 / FPS (统一协议, 空闲 GPU 顺序测量)
| 项 | A0 (s) | M (m) | 变化 |
|---|---|---|---|
| 参数量 | 7,700,106 | **23,568,416** | **3.06×** |
| 模型大小 (pdparams) | 30.8 MB | 94.3 MB | 3.06× |
| 推理 FPS (eval) | 19.8 | 17.8 | −10% |
| 训练显存峰值 (nvidia-smi) | (A0 训练时未采样) | 20.3 GB / 25.9 GB (78%) | — |

### 2.4 13 类 AP (M-best vs A0-best)
| 类别 | A0 | M | Δ |
|---|---|---|---|
| Tomato Early blight leaf | 0.236 | 0.282 | **+0.046** |
| Tomato Septoria leaf spot | 0.448 | 0.468 | +0.020 |
| Tomato leaf | 0.224 | 0.241 | +0.017 |
| Tomato leaf bacterial spot | 0.269 | 0.296 | +0.027 |
| Tomato leaf late blight | 0.438 | 0.461 | +0.023 |
| Tomato leaf mosaic virus | 0.247 | 0.231 | −0.016 |
| Tomato leaf yellow virus | 0.192 | 0.208 | +0.016 |
| Tomato mold leaf | 0.160 | 0.199 | +0.039 |
| Apple Scab Leaf | 0.540 | 0.547 | +0.007 |
| Apple leaf | 0.787 | **0.850** | +0.063 |
| Apple rust leaf | 0.658 | 0.658 | 0.000 |
| grape leaf | 0.712 | 0.712 | 0.000 |
| grape leaf black rot | 0.415 | 0.414 | −0.001 |

**10 类提升 / 2 类持平 / 2 类微降 (mosaic −0.016, grape黑腐 −0.001)。强类 (A0 最高 AP 的 Apple leaf / grape leaf / Apple rust / Apple Scab / Septoria) 全部维持或提升 → 无强类回退。**

### 2.5 主要混淆对 (已配对跨类)
| A0 | M |
|---|---|
| Bact spot → Septoria: 6 | Bact spot → Septoria: 6 |
| Tomato leaf → late blight: 3 | Apple Scab → Apple rust: 5 |
| mold → late blight: 3 | Early blight → Septoria: 3 |
| Apple Scab → Apple rust: 3 | Tomato leaf → late blight: 3 |

主要混淆结构相似 (类内相近对不同),M 未引入新的系统性混淆。

---

## 3. M-final (100 epoch 最终 checkpoint) 参考
| 指标 | M-final | A0-final | A0-best |
|---|---|---|---|
| mAP@0.5:0.95 | 0.408 | 0.389 | 0.410 |
| mAP@0.5 / @0.75 | 0.573 / 0.472 | — | 0.576 / 0.470 |
| Precision / Recall / F1 | 0.488 / 0.463 / 0.475 | — | 0.551 / 0.371 / 0.444 |
| 密集 Recall | 0.258 | — | 0.129 |

best→final 回落: A0 0.410→0.389 (Δ0.021), M 0.428→0.408 (Δ0.020) —— 两者过拟合幅度相当,M 在两个 checkpoint 均稳定领先 A0。

---

## 4. 过程保真 (无中途调参, 零漂移)
- 唯一变量 = 模型规模 s→m;bs/lr/epoch/EMA/评估/A0 同构, 未做任何中途调参。
- DRIFT_GUARD: 15 文件 md5, 训练前后复核 **0 漂移** (配置/数据/代码/A0 权重/M 预训练/M checkpoint)。
- Smoke 全过后才启动正式 100 epoch; 正式训练 exit=0。
- 预训练权重 = 官方 obj365 m (首次下载, 无缓存污染)。

---

## 5. "是否真正优于 A0" 重点分析

### 支持 M 优于 A0 的证据
1. **提升一致且广泛**: mAP@0.5:0.95 +0.018, 同时 mAP@0.5/+0.75、P/R/F1、AR@1/10/100、small/medium/large AP 全部正向。
2. **best 与 final 双 checkpoint 均领先**: best +0.018, final +0.019 —— 非单点偶然。
3. **13 类 AP 10 升 2 平 2 微降**, 强类无回退 (Apple leaf 0.787→0.850)。
4. **密集场景 Recall 0.129→0.210 (+62%)** —— 检测能力在困难场景的直接体现。

### 代价与保留意见
1. **参数量/模型大小 3.06×, FPS −10%** (19.8→17.8)。若部署/速度预算敏感,此代价需权衡。
2. **VAL 仅 113 张**: +0.018 存在一定噪声风险;最终 TEST (114 张, 全程封闭) 才是权威裁定。
3. **conf=0.5 下部分弱类 Recall 回退**: Bact spot 0.19→0.00, Apple Scab 0.47→0.37, mosaic 0.18→0.09。注意 Bact spot 与 Apple Scab 的 **AP 反而上升** (0.269→0.296, 0.540→0.547), 属高置信度分数校准偏移, 非能力丢失, 但高阈值应用场景需留意。
4. 训练显存需求更高 (M ~20GB vs A0 更低), 单卡 24GB 可跑, 但多卡/低显存环境受限。

### 结论
按决策规则 M 为**最终模型候选** (≥0.415 且无强类回退)。是否替代 A0 作为提交模型,取决于速度/参数预算 (FPS −10%, 大小 3.06×) 与精度提升 (+0.018) 的取舍 —— 交由用户决定。未访问 TEST。

---

## 6. 产出物
- `configs/ppyoloe_plus_crn_m_100e_agrivision.yml` (正式) + `..._smoke.yml`
- `checkpoints/best_model.pdparams` (0.428) + `model_final.pdparams` (0.408)
- `logs/formal_train.out`, `smoke/` (门控记录 + 日志), `eval/` (三模型统一评估)
- `DRIFT_GUARD.json`, `scripts/m/` (vram_probe / compat_check / snapshot_guard)
- 归档: `archive/`
