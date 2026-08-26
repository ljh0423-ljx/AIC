# AgriVision · 农业病害智能检测系统

面向 AIC2026 农业视觉（AgriVision）目标检测任务，基于 **PP-YOLOE+-m** 的农作物叶部病害自动检测系统，覆盖番茄 / 苹果 / 葡萄三类作物共 **13 类**叶部病害/健康状态，交付 CPU/GPU 双后端推理与 Gradio Web 检测平台。

## 关键结果（实测）

| 项 | 值 |
|---|---|
| 最终模型 | PP-YOLOE+-m（depth_mult=0.67, width_mult=0.75） |
| 参数量 / 模型大小 | 23.57M / 94.3 MB |
| 训练轮数 | 100 epoch |
| VAL mAP@0.5:0.95（模型选择依据） | **0.428** |
| 独立 TEST mAP@0.5:0.95 | **0.417** |
| GPU 批量推理 FPS（8 张 VAL，RTX 4090） | 12.84 |
| CPU 批量推理 FPS（8 张 VAL） | 0.42 |

## 13 类（模型输出顺序 0–12）

`Tomato Early blight leaf`、`Tomato Septoria leaf spot`、`Tomato leaf`、`Tomato leaf bacterial spot`、`Tomato leaf late blight`、`Tomato leaf mosaic virus`、`Tomato leaf yellow virus`、`Tomato mold leaf`、`Apple Scab Leaf`、`Apple leaf`、`Apple rust leaf`、`grape leaf`、`grape leaf black rot`。

## 目录结构

```
configs/        # PP-YOLOE+ 训练配置（_BASE_ 引用 PaddleDetection 官方配置）
dataset/        # 数据集（本仓库仅保留标注/标签；图片需另行下载 PlantDoc）
inference/      # 推理核心：DetectorBackend 抽象 + PaddleBackend（CPU/GPU）+ RKNN 占位
web/            # Gradio Web 检测平台（app.py / backend.py / config.py）
scripts/        # 训练 / 评估 / 环境安装 shell 脚本 + 分析脚本
tools/          # 数据格式转换（yolo_to_coco.py）
experiments/    # 消融实验报告（checkpoint 不随仓库上传）
docs/           # 技术说明书与图素材
```

## 快速开始

### 环境依赖

- **本地推理 / Web**：Windows + PaddlePaddle 3.3.0（CPU）
  ```bash
  pip install -r inference/requirements_infer.txt
  pip install -r web/requirements_web.txt
  ```
- **训练 / 评估**：AutoDL Linux + PaddlePaddle 2.6.2（cu118）+ [PaddleDetection v2.9.0](https://github.com/PaddlePaddle/PaddleDetection)
  ```bash
  bash scripts/install_env.sh
  ```

### 启动 Web 检测平台

```bash
# Windows（推荐，已内置 UTF-8 编码修复）
web\run_web.bat
# 等价于：PYTHONUTF8=1 python web/app.py
# 浏览器访问 http://127.0.0.1:7860
```

### 命令行推理

```bash
python inference/infer.py --image xxx.jpg --conf 0.5
python inference/infer.py --dir <图片目录> --recursive --limit 6
```

### 训练 / 评估

```bash
bash scripts/train_baseline.sh                          # 训练（默认 bs=8/lr=0.001）
bash scripts/eval_baseline.sh [WEIGHTS_PATH]            # 在独立 TEST 集评估
```

## 重要说明

- **PaddleDetection 依赖**：本仓库不包含 `PaddleDetection/` 源码副本。训练/评估需先克隆官方仓库到项目根目录（`PaddleDetection/`），或由 `scripts/install_env.sh` 安装。
- **数据集图片**：`dataset/*/images/` 未随仓库上传（体积约 1.3 GB）。数据来自公开数据集 [PlantDoc](https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset)（CC-BY-4.0），请自行下载并放入 `dataset/processed_detection_exiffix/images/{train,val,test}/`。
- **最终模型**：`experiments/direction_m/M_SCALEUP_100e/checkpoints/best_model.pdparams`（约 90MB，**未随仓库上传**，需自行训练或另存获取）。
- **TEST 集封闭**：TEST 仅用于最终一次独立评估，不参与训练/调参/模型选择；模型选择唯一依据为 VAL。
- **13 类映射**：严格取自 `dataset/processed_detection_exiffix/annotations/val.json`，不重新编号。项目无官方中文类名，默认仅显示英文。

## 文档

- 技术说明书：[docs/PROJECT_TECHNICAL_SPEC.md](docs/PROJECT_TECHNICAL_SPEC.md)
- 事实库（所有实验数据与证据来源）：[PROJECT_FINAL_FACTS.md](PROJECT_FINAL_FACTS.md)
- 实验总清单：[PROJECT_EXPERIMENT_INDEX.md](PROJECT_EXPERIMENT_INDEX.md)

## 许可证

代码与文档由项目作者维护；数据集遵循 PlantDoc 的 CC-BY-4.0 许可。
