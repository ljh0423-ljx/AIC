# 农业病害检测系统（PP-YOLOE+-m 推理模块）

基于最终锁定模型 **PP-YOLOE+-m**（EXIF-fix 数据，VAL mAP@0.5:0.95=0.428，TEST=0.417）的本地推理程序。
当前环境：Windows + PaddlePaddle 3.3.0 + CPU。正式训练/TEST 环境保持 AutoDL Paddle 2.6.2，**本模块不改动本地 Paddle 版本**。

> 约束声明：本模块不训练、不调参、不修改 checkpoint、不修改数据集、不访问 TEST、
> 不做模型转换与 RK3588 部署。13 类映射严格取自项目已有的 `val.json`，不重新编号。

---

## 1. 目录结构

```
inference/
├── config.py            # 路径 / 13类映射 / 默认参数（全部 Path，无写死 Windows 路径）
├── detector.py          # 后端抽象 DetectorBackend + PaddleBackend（CPU/GPU），预留 RKNN 接口
├── postprocess.py       # 统一 Detection 结构、置信度过滤、每图/批次统计、JSON/CSV/TXT
├── visualize.py         # 可视化：原图比例、类别(中英可选)+置信度、标题、目标数、真实耗时
├── batch_infer.py       # 批量推理引擎：失败隔离、时间戳目录、真实 FPS
├── infer.py             # 命令行入口
├── requirements_infer.txt
├── README.md
└── outputs/             # 输出根目录（每次运行自动创建 run_YYYYMMDD_HHMMSS/）
```

## 2. 依赖安装

```bash
pip install -r inference/requirements_infer.txt
# 并确保已安装 PaddlePaddle（本项目本地为 3.3.0，CPU；见 requirements 说明）
```

## 3. 使用方法

在工程根目录（AIC2026_AgriVision/）下运行。模型默认加载一次，批量推理不重复加载。

```bash
# 单图推理
python inference/infer.py --image xxx.jpg

# 文件夹推理（默认非递归）
python inference/infer.py --dir dataset/processed_detection_exiffix/images/val

# 指定置信度
python inference/infer.py --image xxx.jpg --conf 0.5

# 多图批量
python inference/infer.py --image a.jpg --image b.jpg --conf 0.5

# 递归 + 限量（小规模验证）
python inference/infer.py --dir dataset/processed_detection_exiffix/images/val --recursive --limit 6

# 其他选项
--output <目录>    指定输出根目录（默认 inference/outputs/）
--device cpu|gpu  推理设备（gpu 为预留接口，无 GPU 自动回退 CPU）
--backend paddle  推理后端（rknn 为预留接口）
--no-visualize    不生成可视化图
```

支持图片格式：`jpg / jpeg / png / bmp / webp`。单张失败不会中断批次，失败文件与原因会记录。

## 4. 输出产物

每次运行在输出根目录下新建 `run_YYYYMMDD_HHMMSS/`（不覆盖历史）。每个运行目录包含：

```
run_20260817_154500/
├── originals/                # 原图副本
├── visualized/               # 检测可视化图（bbox + 类别 + 置信度 + 标题 + 目标数 + 真实耗时）
├── detections/
│   ├── <图名>.json           # 每图 JSON（含 inference_ms）
│   ├── <图名>.txt            # 每图简洁统计（总目标数、每类数量/平均置信度）
│   └── <图名>.csv            # 每图检测明细（image_name,class_id,class_name,confidence,x,y,width,height）
├── predictions.json          # 全批次检测结果
├── predictions.csv           # 全批次检测明细
├── class_statistics.csv      # 每类：出现图片数、目标总数、平均/最高置信度
├── inference_summary.json    # 批次摘要（配置 + 性能 + 失败明细）
└── inference.log             # 运行日志
```

性能统计全部基于真实运行时间：
- `total_images` / `success_images` / `failed_images`
- `total_time`（整个批次墙钟时间，秒）
- `average_time`（total_time / total_images）
- `fps`（total_images / total_time，真实吞吐，不虚构）

## 5. 类别映射（13 类，严格取自项目 val.json）

| COCO id | 英文名 |
|---|---|
| 1 | Tomato Early blight leaf |
| 2 | Tomato Septoria leaf spot |
| 3 | Tomato leaf |
| 4 | Tomato leaf bacterial spot |
| 5 | Tomato leaf late blight |
| 6 | Tomato leaf mosaic virus |
| 7 | Tomato leaf yellow virus |
| 8 | Tomato mold leaf |
| 9 | Apple Scab Leaf |
| 10 | Apple leaf |
| 11 | Apple rust leaf |
| 12 | grape leaf |
| 13 | grape leaf black rot |

> 中文名：项目当前**没有**官方中文映射，遵循"没有则不要擅自创造"约束，
> 默认仅显示英文。如需"中文 + 英文"标注，请在 `config.py` 的 `CLASS_NAMES_CN` 中
> 填写（例如 `{1: "番茄早疫病", ...}`），程序会自动启用双语显示。

## 6. 后端解耦与 RK3588 迁移接口

- 所有功能通过统一接口 `DetectorBackend`（`detector.py`）访问：
  `load() / infer(bgr) / infer_batch(list) / close()`。
- 当前实现 `PaddleBackend`（CPU/GPU）。
- 后续 RK3588 只需实现 `RKNNBackend(DetectorBackend)` 并在 `create_backend()` 注册，
  `infer.py / batch_infer.py / postprocess.py / visualize.py` **无需任何改动**。
- 核心代码全部使用 `pathlib.Path` 动态拼接路径，不写死任何 Windows 盘符/绝对路径；
  迁移 RK3588（Linux/ARM）时直接复制工程即可。
- 数据集路径通过 `-o` 绝对路径覆盖注入（`-o TestDataset.dataset_dir=...`），
  不依赖软链、不修改 PaddleDetection 源码、不使用 test.json。

## 7. 已知限制

- 本地为 CPU：单图约 2.3 s（640×640 前向 ~2.2 s），非实时；实时性需 GPU/RK3588。
- 图片含 EXIF 旋转信息时，模型已按"烘焙后"像素处理（本数据集已修复），
  任意第三方照片若带 EXIF 方向，建议先转正后再推理。
- `PaddlePaddle 3.x` 仅验证了推理路径；训练/复现仍以 AutoDL 2.6.2 为准。
