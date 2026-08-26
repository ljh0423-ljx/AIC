# PP-YOLOE+-s Baseline 实验报告 (模板)

> 训练前预填 ①②③④, 训练后填写 ⑤~⑬。本机不训练, 全部数据由 AutoDL 训练产出后回填。

## ① 环境信息
- 训练平台: AutoDL (Linux)
- Python: `<训练后填>`
- PaddlePaddle: 2.6.2 (GPU)
- PaddleDetection: `<填 ppdet.__version__>`
- CUDA: `<填>`  | cuDNN: `<填>`
- GPU: `<填型号>`  | 显存: `<填 GB>`
- GPU 可用: 是

## ② PP-YOLOE 配置
- 模型: PP-YOLOE+-s (depth_mult=0.33, width_mult=0.50)
- 骨干: CSPResNet (官方, 未改)
- 颈部: CustomCSPPAN (官方, 未改)
- 检测头: PPYOLOEHead (官方, 未改)
- 损失: Varifocal Loss + IoU + DFL (官方默认, 未改)
- EMA: 开启 (官方默认, 非本次新增)
- 预训练: Objects365 预训练骨干
- **未增加**: CBAM/Transformer/Focal Loss/注意力/剪枝/蒸馏/量化/任何自定义模块

## ③ 训练参数
- 配置: `configs/ppyoloe_plus_crn_s_100e_agrivision.yml`
- epoch: 100  |  base_lr: 0.001  |  优化器: Momentum(0.9) + L2(5e-4)
- lr 调度: CosineDecay(max_epochs=100) + LinearWarmup(5e)
- batch_size: 8 (train) / 2 (eval)  |  输入: 640 (多尺度 320~768)
- snapshot_epoch: 5  |  seed: 随训练框架默认

## ④ 数据集统计
- 路径: `dataset/processed_detection/` (COCO 格式)
- 3 作物 / 13 类 / 1144 图 / 3861 目标
- train=916 (3084 标注) / val=114 (354) / test=114 (423)
- seed=2026, 已去重, 无跨 split 泄露
- 详见 `dataset/processed_detection/DATASET_FINAL_REPORT.md`

| id | 类别 | 作物 | 性质 |
|---|---|---|---|
| 0 | Tomato Early blight leaf | Tomato | 病害 |
| 1 | Tomato Septoria leaf spot | Tomato | 病害 |
| 2 | Tomato leaf | Tomato | 健康 |
| 3 | Tomato leaf bacterial spot | Tomato | 病害 |
| 4 | Tomato leaf late blight | Tomato | 病害 |
| 5 | Tomato leaf mosaic virus | Tomato | 病害 |
| 6 | Tomato leaf yellow virus | Tomato | 病害(多目标) |
| 7 | Tomato mold leaf | Tomato | 病害 |
| 8 | Apple Scab Leaf | Apple | 病害 |
| 9 | Apple leaf | Apple | 健康 |
| 10 | Apple rust leaf | Apple | 病害 |
| 11 | grape leaf | Grape | 健康 |
| 12 | grape leaf black rot | Grape | 病害(小样本) |

## ⑤ mAP
- mAP@50:95: `<训练后填>`
- mAP@50: `<训练后填>`

## ⑥ Precision
- `<训练后填>`

## ⑦ Recall
- `<训练后填>`

## ⑧ 13 类 AP
| id | 类别 | AP@50 | AP@50:95 |
|---|---|---|---|
| 0 | Tomato Early blight leaf | | |
| ... | (训练后填全 13 行) | | |

## ⑨ 混淆矩阵
- `<训练后用 PaddleDetection 或自写脚本生成, 附图>`

## ⑩ 推理速度
- 单图推理耗时: `<填 ms>` (640 输入, batch=1)
- FPS: `<填>`

## ⑪ 模型参数量 / 模型大小
- 参数量: `<填 M>`
- 模型大小: `<填 MB>` (model_final.pdparams)

## ⑫ Baseline 问题分析
- 重点类 Recall:
  - Tomato leaf yellow virus: `<填>` (目标占比高, 关注过拟合)
  - grape leaf black rot: `<填>` (小样本, 关注 recall)
  - Tomato leaf / Apple leaf / grape leaf (健康类): `<填>` (关注与病害类混淆)
- 训练曲线: `<填 loss/mAP 趋势>`

## ⑬ 下一阶段算法改进建议
- `<训练后基于 baseline 短板给出, 例如: 类别加权/copy-paste增强/小类过采样/轻量化改进等>`
- 注意: 改进需对照本 baseline, 不擅自修改训练集。

---
*模板生成: 2026-08-13 | 训练后回填指标 | 原始数据未修改*
