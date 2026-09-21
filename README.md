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
knowledge/      # 农业病害知识库（笔记 + 来源清单 + 向量索引）
```

## AI 智能诊断（V1.1 / V1.2）

在检测平台基础上，V1.1/V1.2 新增真实 AI 智能诊断链路（VLM 视觉分析 / RAG 知识检索 / Agent 决策 / 中文语音问答 / 多会话隔离）。

### AI 诊断验收报告（V1.1，位于【⬇️ 结果下载】）

- **生成报告**：上传图片 → 点击「开始 AI 检测」→ 切换到【⬇️ 结果下载】→ 点击「生成验收报告」。
- **报告预览 / 下载 PDF**：编号、生成时间、作物、病害类别、检测置信度、检测框面积代理指标、视觉证据、知识来源、人工复核建议与系统运行状态；中文 A4 排版、图片嵌入、可点击来源链接；**复用当前会话结果，不重复检测、不调用付费 API**。

### 诊断流程图 / 可视化记录图（V1.2，独立顶部 Tab）

- 顶部导航在【⬇️ 结果下载】之后新增 **【🧭 诊断流程图 / 可视化记录图】** Tab。
- **生成流程图**：点击「生成诊断流程图」→ 生成 8 节点流程（原始图片 → PP-YOLOE 检测 → 检测框面积代理评估 → VLM 视觉分析 → RAG 知识检索 → Agent 综合建议 → 语音交互状态 → 验收报告状态）→ 页面预览 → 下载 PNG / PDF。

> 报告与流程图均为「系统运行验收」记录，不等同于病害专业确诊；检测框面积占比为规则代理指标，未检出 ≠ 已确认健康；VLM 分析不等同于病原学确诊。

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
- AI 智能模块指南（V1.0/V1.1 配置与架构）：[docs/V1.0_AI_MODULES_GUIDE.md](docs/V1.0_AI_MODULES_GUIDE.md)
- V1.2 交付报告（诊断流程图 / 可视化记录图）：[docs/V1.2_DELIVERY_REPORT.md](docs/V1.2_DELIVERY_REPORT.md)
- V1.1 交付报告（AI 诊断验收报告）：[docs/V1.1_DELIVERY_REPORT.md](docs/V1.1_DELIVERY_REPORT.md)
- V1.0 交付报告：[docs/V1.0_DELIVERY_REPORT.md](docs/V1.0_DELIVERY_REPORT.md)
- 事实库（所有实验数据与证据来源）：[PROJECT_FINAL_FACTS.md](PROJECT_FINAL_FACTS.md)
- 实验总清单：[PROJECT_EXPERIMENT_INDEX.md](PROJECT_EXPERIMENT_INDEX.md)

## 许可证

代码与文档由项目作者维护；数据集遵循 PlantDoc 的 CC-BY-4.0 许可。
