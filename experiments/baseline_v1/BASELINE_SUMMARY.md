# PP-YOLOE+-s Baseline v1 — 最终汇总报告

- 数据集: AIC2026 农业视觉 (PlantDoc 13 类目标检测)
- 模型: PP-YOLOE+-s (depth_mult=0.33, width_mult=0.50), 严格 Baseline, 无任何结构/损失/增强创新
- 训练完成日期: 2026-08-14
- 训练时长: ≈ 2.9 小时 (100 epoch, RTX 4090)

---

## 1. 环境

| 项目 | 值 |
|---|---|
| GPU | NVIDIA RTX 4090, 24 GB |
| Driver / CUDA Runtime | 580.76.05 / 11.8 |
| PaddlePaddle | 2.6.2 (cu118, compiled_with_cuda) |
| PaddleDetection | v2.9.0 `b25522a0` |
| Python | 3.12.3 |
| 训练显存峰值 (bs=16) | ≈ 10.9 GB / 24 GB |

详见 `environment.txt`, `git_commit.txt`。

## 2. 数据

| Split | 图像数 | 目标数 |
|---|---|---|
| train | 915 | 3079 |
| val | 113 | 350 |
| test | 114 | 423 |
| 合计 | 1142 | 3852 |

- 13 类, COCO category_id 1–13, 名称与 `label_list.txt` 逐一对齐
- 原始 `processed_detection` 未做任何修改; 训练使用**清洁副本** `processed_detection_clean`(硬链接复制,移除 2 张 0 字节损坏图)
- **test 集全程未参与训练与调参**,仅最终评估
- 已知数据质量项: 11 张 (0.96%) 图像存在 EXIF 旋转 (标注尺寸与像素宽高对调), 按用户决策保持原样仅记录 → 详见 `data_prep_notes.txt`

## 3. 训练配置

- epoch: 100, snapshot_epoch: 5, `--eval` 每 epoch val 评估
- batch_size: 16 (train) / 2 (eval)
- base_lr: 0.002 = 0.001 × 16/8 (线性缩放), LinearWarmup 5 epoch → CosineDecay 100 epoch
- 预训练: Objects365 预训练骨干 (官方默认), 仅加载 backbone
- 完整命令: `train_command.txt`, 配置: `configs/ppyoloe_plus_crn_s_100e_agrivision.yml`

## 4. Smoke Test (2 epoch, 非正式结果)

- 结果: PASS — 干净退出, 无 OOM, loss 4.74→3.25, val mAP@0.5:0.95 0.011→0.029
- 详见 `experiments/smoke_test/SUMMARY.txt`

## 5. 最终指标

### 5.1 VAL 集 (训练时每 epoch 评估, COCO metric)

| 指标 | model_final (epoch 100) | best_model |
|---|---|---|
| mAP@0.5:0.95 | 0.388 | **0.408** (epoch ~69) |
| mAP@0.5 | 0.549 | — |
| AR@100 | 0.702 | — |

完整见 `metrics/val_final_metrics.txt`。

### 5.2 TEST 集 (最终一次独立评估 — 两个 checkpoint 并列报告, 不用于模型选择)

| 指标 | best_model | model_final |
|---|---|---|
| mAP@0.5:0.95 | 0.396 | 0.401 |
| mAP@0.5 | 0.577 | 0.597 |
| mAP@0.75 | 0.438 | 0.432 |
| AR@100 | 0.676 | 0.675 |
| Precision (conf 0.5) | 0.800 | 0.771 |
| Recall (conf 0.5) | 0.501 | 0.596 |
| F1 (conf 0.5) | 0.616 | 0.672 |

> **严格实验规范**: TEST 集仅用于最终一次独立评估, **严禁参与任何训练/调参/模型选择**。
> `best_model` 与 `model_final` 均为训练直接产物, 两权重的 TEST 结果如实并列报告 (不加粗标注推荐),
> 其差异 **不作为选择依据**。checkpoint 选择只能依据 VAL (见 5.1)。

### 5.3 Per-class AP (TEST 集, IoU=0.5:0.95)

| 类别 | best_model | model_final |
|---|---|---|
| Tomato Early blight leaf | 0.199 | 0.253 |
| Tomato Septoria leaf spot | 0.476 | 0.465 |
| Tomato leaf | 0.283 | 0.283 |
| Tomato leaf bacterial spot | 0.202 | 0.217 |
| Tomato leaf late blight | 0.547 | 0.565 |
| Tomato leaf mosaic virus | 0.182 | 0.169 |
| Tomato leaf yellow virus | 0.211 | 0.206 |
| Tomato mold leaf | 0.285 | 0.247 |
| Apple Scab Leaf | 0.542 | 0.613 |
| Apple leaf | 0.600 | 0.551 |
| Apple rust leaf | 0.473 | 0.500 |
| grape leaf | 0.520 | 0.494 |
| grape leaf black rot | 0.636 | 0.647 |

弱类别 (mAP@0.5:0.95 < 0.30): Tomato leaf mosaic virus, Tomato leaf yellow virus, Tomato leaf bacterial spot, Tomato Early blight leaf, Tomato leaf, Tomato mold leaf — 均为数据量偏少/样本重叠高的番茄病害类。

### 5.4 混淆矩阵 (TEST 集, conf=0.5)

- 保存为 `metrics/confusion_matrix_best_model.csv`, `metrics/confusion_matrix_model_final.csv`
- 主要混淆: Tomato leaf ↔ Tomato leaf yellow virus (leaf 被判为 yellow virus 5 例), Tomato bacterial spot ↔ Septoria/mosaic (2-4 例), Apple rust leaf ↔ Apple Scab (1-3 例) — 均为病害形态相似的相邻类别。
- 无跨科混淆 (番茄↔苹果↔葡萄), 说明骨干提取特征有效。

## 6. 模型规模与推理速度

| 项目 | 值 |
|---|---|
| 参数量 | **7.70 M** |
| 模型文件大小 (pdparams) | **29.42 MB** |
| 推理延迟 (batch=1, @640) | **20.5 ms/张** |
| 推理吞吐 (batch=1, @640) | **48.7 FPS** |
| 推理吞吐 (batch=16, @640) | **602 img/s** |
| eval.py 实测 FPS | 17.3 (含数据加载+后处理) |

测试环境: RTX 4090, TensorRT 未启用 (纯 Paddle Inference), FP32。

## 7. 结论与 Baseline 定位

- **checkpoint 选择规范**: `best_model` 仅由 **VAL** 指标确定 (best val mAP@0.5:0.95 = 0.408, epoch ~69, 训练时 COCO eval 自动保存); `model_final` 为 epoch 100 末尾权重。二者均保存并如实报告。
- **TEST 仅用于最终一次独立评估**: 两个 checkpoint 的 TEST 结果 (mAP@0.5:0.95 = 0.396 / 0.401) 并列报告, **不据此选择模型**。
- Baseline 定位: 915 训练图 / 3079 目标的严格 Baseline, 反映小样本 + 类别不平衡下限。
- 后续改进方向 (非 Baseline, 供参考): 更强数据增强 / 平衡采样 / 更多预训练 / 更大模型 (m/l) / 缓解 EXIF 旋转噪声。
- 原始数据零修改, test 零污染, 全流程可复现。

## 8. 产物清单

```
experiments/baseline_v1/
├── BASELINE_SUMMARY.md        ← 本文件
├── environment.txt            环境信息 (已回填)
├── git_commit.txt             PaddleDetection commit
├── config.yml / config_dataset.yml  训练配置
├── train_command.txt          实际训练命令
├── data_prep_notes.txt        数据质量记录 (0字节图 / EXIF 旋转)
├── train.log                  完整 100 epoch 训练日志
├── checkpoints/               epoch{4..94}.pdparams + best_model / model_final
├── logs/                      VDL 曲线
└── metrics/
    ├── val_final_metrics.txt  最终 val 指标
    ├── test_eval.log / test_eval_model_final.log   test 评估日志 (含 per-class AP)
    ├── test_bbox_{best_model,model_final}.json     预测结果
    └── confusion_matrix_{best_model,model_final}.csv
```
