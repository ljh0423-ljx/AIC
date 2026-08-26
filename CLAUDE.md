# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

AIC2026 农业视觉（AgriVision）目标检测系统：以 **PP-YOLOE+-m** 模型为核心，对番茄/苹果/葡萄三类作物 **13 类叶部病害/健康状态**做检测。数据来自 PlantDoc 公开数据集（COCO 格式）。已交付 CPU/GPU 双后端推理、Gradio Web 检测平台、15 张比赛演示案例，并预留 RK3588 边缘部署接口。

最终模型指标（实测）：VAL mAP@0.5:0.95 = 0.428（模型选择依据），独立 TEST = 0.417，参数量 23.57M / 94.3 MB。

## 仓库边界（重要）

- `PaddleDetection/` 是**第三方框架的本地副本**（PaddleDetection v2.9.0，用于训练/评估）。**不要修改其源码**；数据集/权重路径一律通过 `-o` 命令行覆盖注入，不依赖软链以外的改动。
- 项目自身的代码只在这些目录：`configs/`（训练配置）、`dataset/`（数据）、`inference/`（推理核心）、`web/`（Web 层）、`scripts/`（训练/评估/环境脚本）、`tools/`（数据转换）、`experiments/`（实验产物与报告）、`docs/`（技术文档）。

## 架构（需要跨文件才能看清的“大图”）

### 推理核心 `inference/`
- `config.py`：所有路径基于 `Path(__file__).parent.parent` 动态计算，**不写死盘符**；13 类映射运行时从 `val.json` 读取。
- `detector.py`：后端抽象。`DetectorBackend`（ABC：`load/predict/predict_batch/info/close`）+ 装饰器注册表 `register_backend()` + 工厂 `create_backend()`。当前实现 `PaddleBackend`（注册为 `paddle_cpu`/`paddle_gpu`）；`RKNNBackend`（`rknn_reserved`）仅为占位，构造即 `NotImplementedError`。**Web/后处理/可视化只依赖 `DetectorBackend` 接口，不依赖 Paddle 私有实现**——新增后端只需实现同类接口并注册，其余模块零改动。
- `postprocess.py`：统一 `Detection` 数据结构、置信度过滤、JSON/CSV/TXT 导出、批次统计。
- `batch_infer.py` / `infer.py`：批量引擎与命令行入口（失败隔离、时间戳输出目录、真实 FPS）。

### Web 层 `web/`
- `app.py`（Gradio 6.24.0 界面，约 71 KB）、`backend.py`（`get_detector()` **单例**，模型全程只加载一次；`self_check()` 启动自检）、`config.py`（UI 常量 + 固定模型性能记录，**复用** `inference/config.py` 的路径与类别映射，不重复定义）。
- Web 通过 `inference.detector` 间接调用推理，与后端解耦。设备模式 `AGRI_DEVICE=auto|cpu|gpu`。

### 数据流
图片 → `web/app.py`（Gradio UI）→ `DetectorBackend`（PaddleBackend CPU/GPU）→ 预处理（Resize/Normalize/Permute，与官方 reader 逐位一致）→ PP-YOLOE+-m 前向 → 后处理解码 → 可视化/统计/下载。

### 训练/评估 `configs/` + `scripts/`
- 配置用 `_BASE_` 组合：项目数据集配置（`configs/datasets/agrivision_detection.yml`）+ PaddleDetection 官方 runtime/optimizer/结构/reader 配置。最终模型 `ppyoloe_plus_crn_m_100e_agrivision.yml` 与 baseline（`_s_`）的唯一差异是规模参数 `depth_mult 0.33→0.67`、`width_mult 0.50→0.75` 与 Objects365 预训练权重，**零自定义算法模块**。
- `scripts/*.sh` 是训练/评估/环境的入口，运行时会建立 `PaddleDetection/dataset/processed_detection` 软链到 `dataset/` 下的实际数据。

## 关键约束（违反会破坏项目一致性）

1. **TEST 集全程封闭**：仅用于最终一次独立评估，绝不参与训练、调参、模型选择。模型选择唯一依据是 **VAL**。推理/自检/演示只允许用 VAL 图片与 `val.json`，禁止读取 `test.json` 或 TEST 图片。
2. **13 类映射固定**：类别顺序严格等于 `dataset/processed_detection_exiffix/annotations/val.json` 的 categories 升序，**不重新编号**；模型 0–12 输出索引通过 `clsid2catid` 对齐 COCO 1–13 id。
3. **无官方中文类名**：项目没有中文类别映射，遵循“没有则不自造”。默认只显示英文；`CLASS_NAMES_CN` 留空，除非用户明确填写。
4. **不修改 PaddleDetection 源码**，不改 checkpoint/数据集，不做模型转换（RK3588 仅为预留接口，禁止声称已部署）。
5. **编码问题（Chinese Windows 必踩）**：模型配置文件含中文注释，PaddleDetection 用默认 GBK 读取会报 `'gbk' codec can't decode byte ...`。启动 Web 必须走 `web/run_web.bat`（内部 `chcp 65001` + `set PYTHONUTF8=1`），或先 `set PYTHONUTF8=1` 再 `python web/app.py`。
6. **UI 约定**：`web/app.py` 的功能切换区保持 Gradio 默认文字导航，**不要改成按钮式方框 Tab**（此前已改过又回退）。

## 常用命令

### Web（本地 Windows，Paddle 3.3.0 CPU）
```bash
web\run_web.bat                 # 已强制 PYTHONUTF8=1；默认 http://127.0.0.1:7860
# 等价：PYTHONUTF8=1 python web/app.py
# 参数：--device cpu|gpu|auto（默认 auto）；--port 7860
# 环境：AGRI_DEVICE=auto|cpu|gpu ；GRADIO_SHARE=true 启用公网分享
```

### 推理 CLI（`inference/`）
```bash
python inference/infer.py --image xxx.jpg [--conf 0.5] [--device cpu|gpu] [--output DIR] [--no-visualize]
python inference/infer.py --dir dataset/processed_detection_exiffix/images/val [--recursive] [--limit N]
# 多图：--image a.jpg --image b.jpg
```

### 训练 / 评估（AutoDL Linux，PaddlePaddle 2.6.2 cu118）
```bash
bash scripts/install_env.sh                    # 装 paddlepaddle-gpu + PaddleDetection 依赖（PADDLE_VERSION/CUDA_TAG 可覆盖）
bash scripts/check_env.sh                      # 环境信息 + GPU smoke test
bash scripts/check_gpu_config.sh               # 按显存给出 batch_size/lr 建议
bash scripts/train_baseline.sh                 # 训练（BATCH_SIZE=/BASE_LR= 覆盖；--eval 每 epoch 评估）
bash scripts/eval_baseline.sh [WEIGHTS_PATH]   # 在 TEST 集做最终评估（--classwise）
bash scripts/infer_baseline.sh IMAGE [WEIGHTS] # 单图推理
bash scripts/convert_dataset.sh                # 数据转换（见 tools/yolo_to_coco.py）
```
- 磁盘上的 `.yml` 默认 `bs=8 / lr=0.001`；最终 M 模型实际用 `-o TrainReader.batch_size=16 LearningRate.base_lr=0.002`（lr 按 bs 线性缩放）。复现最终结果须携带这些覆盖参数。

### 依赖
- `inference/requirements_infer.txt`、`web/requirements_web.txt`（本地推理/Web）；训练依赖在 `scripts/install_env.sh` 内安装。

## 两个运行环境（勿混淆）

| 环境 | 用途 | 框架版本 |
|---|---|---|
| AutoDL Linux（RTX 4090 24GB） | 训练 / 评估 / GPU 推理 | PaddlePaddle 2.6.2（cu118）+ PaddleDetection v2.9.0 + Python 3.12 |
| 本地 Windows | Web 展示 / CPU 推理验证 | PaddlePaddle 3.3.0（CPU，仅验证过推理路径，不用于训练/复现） |

## 关键路径

- 最终模型：`experiments/direction_m/M_SCALEUP_100e/checkpoints/best_model.pdparams`
- 最终配置：`configs/ppyoloe_plus_crn_m_100e_agrivision.yml`
- 最终数据集（EXIF-fix，锁定）：`dataset/processed_detection_exiffix/`（train 915 / val 113 / test 114）
- 权威事实库：`PROJECT_FINAL_FACTS.md`（所有实验数据与证据来源）；技术说明书：`docs/PROJECT_TECHNICAL_SPEC.md`

## 测试与验证

仓库**没有自动化单元测试**。验证依赖 `scripts/` 下的 shell 脚本（环境 smoke test、训练、评估）与 `experiments/` 下的实验/验证报告（如 `M_FINAL_TEST_REPORT.md`、`GPU_BATCH_INFERENCE_REPORT.md`）。改动推理/Web 后，用 CLI 推理与 Web 自检来回归验证，不要臆造测试命令。
