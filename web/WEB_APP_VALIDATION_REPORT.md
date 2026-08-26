# WEB_APP_VALIDATION_REPORT — 农业病害智能检测系统 Web 功能验证

- **验证日期**: 2026-08-17
- **性质**: Web 展示层开发 + 本地功能验证（**未训练、未调参、未修改 checkpoint、未修改数据集、未访问 TEST、未做模型转换与 RK3588 部署**）
- **结论**: ✅ 全部验证通过。Gradio Web 系统可启动、模型只加载一次、单图/批量/异常处理/可视化/统计/下载/清空均正常，性能统计全部基于真实运行时间。

---

## 1. 启动方式与环境

```bash
# 在工程根目录（AIC2026_AgriVision/）执行
python web/app.py
# 浏览器访问 http://127.0.0.1:7860
```

| 项 | 值 |
|---|---|
| 环境 | Windows 11 · Python 3.12.4 · **PaddlePaddle 3.3.0（CPU）** · **Gradio 6.24.0** |
| 模型 | `experiments/direction_m/M_SCALEUP_100e/checkpoints/best_model.pdparams`（md5 `18bd0e99…`，验证后未变） |
| 配置 | `configs/ppyoloe_plus_crn_m_100e_agrivision.yml`（640×640） |
| 类别映射 | `dataset/processed_detection_exiffix/annotations/val.json`（13 类，**非 TEST**） |
| 数据 | 仅使用 `images/val/` 的 VAL 图片；**未读取 test 目录** |

> **环境修复说明（2026-08-17）**：原环境 gradio 4.44.1 与本机较新的 starlette 1.0.0 /
> fastapi 0.135.2 不兼容（`TemplateResponse` 签名变更导致根页面 500），且 gradio_client 1.3.0
> 对 `additionalProperties: true` 的 schema 解析有 bug。已将 **gradio 升级至 6.24.0**
> （连带 gradio-client 2.6.0、starlette 1.6.0），根页面恢复 HTTP 200；`web/app.py` 内置的
> gradio_client 兼容修补在 6.x 下自动失效（已无需）。**未修改 Paddle 版本、模型与数据。**
> 注意：环境另有 `streamlit 1.32.0 与 protobuf` 的既有依赖冲突警告，与本系统无关。

**启动自检（实测全过）**：模型文件 ✅ / 配置文件 ✅ / 类别映射（13 类）✅ / 推理后端（paddle·cpu·paddle=3.3.0·640×640）✅。
模型**启动时只加载一次**（`web/backend.get_detector()` 单例），所有按钮点击复用同一实例。

> 兼容性：部分环境（代理/VPN/沙箱）会拦截对 127.0.0.1 的预启动探测，程序自动启用
> "本地兼容模式"继续启动（不修改任何第三方库文件）。本次验证环境即属此类，兼容回退生效，
> 服务器成功监听 127.0.0.1:7860。

## 2. 功能清单（界面实现确认）

| 功能 | 状态 |
|---|---|
| 项目标题「农业病害智能检测系统」+ 英文副标题「Agricultural Disease Intelligent Detection System」 | ✅ |
| 模型名称「PP-YOLOE+-m」 | ✅ |
| 单张/多张/拖拽上传（jpg/jpeg/png/bmp/webp） | ✅ |
| 置信度阈值滑块（默认 0.5，标注"推理置信度阈值，非训练参数"） | ✅ |
| 开始检测 / 清空任务 | ✅ |
| 结果可视化（原图比例、bbox+类别+置信度+标题+目标数+真实耗时） | ✅ |
| 原图 vs 检测结果对比（下拉选图） | ✅ |
| 检测结果表格（序号/类别/置信度/x/y/width/height） | ✅ |
| 类别统计图（BarPlot，来自实际推理结果） + 类别统计明细表 | ✅ |
| 批次汇总（总图片/成功/失败/总目标/总耗时/平均耗时/实际 FPS） | ✅ |
| 结果下载（检测图片、JSON、CSV、类别统计、汇总） | ✅ |
| 系统信息区（设备 CPU/GPU、Paddle 版本、模型、输入尺寸、类别数） | ✅（如实显示 CPU，未虚构 GPU/NPU） |
| 模型评测卡（VAL=0.428 选模型依据；TEST=0.417 标注"独立 TEST 最终评估结果"） | ✅（固定记录，不重算） |
| 边缘部署卡（仅"已预留 RKNNBackend 接口，可迁移 RK3588"） | ✅（未声称完成部署） |
| 系统介绍（流程：图片上传→AI 病害识别→结果可视化→统计分析） | ✅ |

## 3. 单图验证（TRAIN_000004_apple_scab.jpg → run_20260817_161411）

| 项 | 结果 |
|---|---|
| 上传/检测 | 1 图，成功 1 |
| 检测目标 | **1 个：Apple Scab Leaf（class_id 9）**，conf 0.5698，bbox [166.68,105.84,104.96,87.61] |
| 单图推理耗时 | **2.44 s**（真实计时） |
| 实际 FPS | **0.41** |
| 可视化 | 原图 vs 检测对比正常；表格 1 行；类别统计正确；下载文件齐全 |

## 4. 批量验证（8 张 VAL 图 → run_20260817_161414）

| 项 | 结果 |
|---|---|
| 图片数 | 8（跨 Apple rust / early blight / mosaic / grape / apple scab / septoria 等） |
| 成功 / 失败 | **8 / 0** |
| 检测目标总数 | **11** |
| 总耗时 / 平均耗时 | **20.24 s / 2.53 s** |
| **实际 FPS** | **0.40** |
| 类别统计 | 6 类（Septoria 4、Tomato leaf 3、Bacterial spot 1、Apple Scab 1、Apple rust 1、grape 1），来自实际推理 |
| 可视化 | gallery 8 图、每图原图/检测图/明细齐全 |
| 下载 | predictions.json/csv、class_statistics.csv、inference_summary.json、inference.log + 8 张可视化图（共 13 个文件） |

## 5. 异常处理验证

| 场景 | 结果 |
|---|---|
| 无检测目标图（run_20260817_161434） | 表格为空、汇总提示"未检出任何目标"、原图/可视化保留，不崩溃 |
| 损坏图片混入批次（run_20260817_161437） | 3 图 → 成功 2、**失败 1**，失败原因"图片无法解码…"记录，批次继续 |
| 空输入 | 友好提示（gr.Warning"请先上传至少一张图片…"）+ 结果区重置 |
| 模型/配置缺失 | 自检显示 FAIL 中文原因（模型文件不存在 / 配置文件不存在 / 类别映射异常） |
| 未知后端 | 自检显示"未知推理后端…预留接口，本阶段未实现" |
| 清空任务 | 上传区、结果区、状态全部重置，可重新上传检测 |

## 6. 输出文件位置

| 项 | 路径 |
|---|---|
| Web 源码 | `web/{config,backend,app}.py` + `__init__.py` |
| Web 依赖/说明 | `web/requirements_web.txt`、`web/README.md` |
| Web 检测输出 | `web/outputs/run_20260817_161411`（单图）、`…_161414`（批量）、`…_161434`（无检测）、`…_161437`（失败隔离） |
| 验证报告 | 本文件 `web/WEB_APP_VALIDATION_REPORT.md` |
| 上传暂存 | `web/tmp/`（验证后已清理） |

## 7. 已知限制

1. **性能**：本地 CPU 单图约 2.4 s、批量 FPS≈0.4，非实时；演示建议小批量（≤10 张）。
2. **上传命名**：Web 上传后以 `upload_001.jpg…` 命名（稳定、无歧义），不保留原文件名。
3. **中文类名**：项目无官方中文映射，默认显示英文；填写 `CLASS_NAMES_CN` 后自动启用双语。
4. **EXIF**：本数据集图片已烘焙处理；第三方带 EXIF 旋转照片建议先转正。
5. **localhost 兼容回退**：仅在预启动探测被拦截时启用（见 §1），正常环境走标准启动。

## 8. 后续 RK3588 迁移接口

- Web 层通过 `inference.detector.DetectorBackend`（当前 `PaddleBackend`）调用推理，**不直接依赖 Paddle**。
- 后续新增 `RKNNBackend(DetectorBackend)` 并在 `create_backend()` 注册后，`web/app.py`、`web/backend.py` **无需改动**。
- 全部路径使用 `pathlib.Path` 动态拼接，不写死 Windows 路径；数据集路径以 `-o` 绝对路径注入，不依赖软链、不改 PaddleDetection 源码。
- 本阶段**未进行**任何模型转换与 RK3588 部署。

## 9. 完整性声明

- ✅ 模型未修改（best_model md5 验证前后一致）
- ✅ 数据集未修改（仅读取 VAL 图片，未访问 test 目录）
- ✅ PaddleDetection 源码零改动；未创建软链；本地 Paddle 版本未升级/降级
- ✅ 未训练、未调参、未重新计算 TEST 指标、未虚构 GPU/NPU/FPS、未声称完成 RK3588 部署
- ✅ 保留 `inference/INFERENCE_VALIDATION_REPORT.md` 与 `inference/outputs/` 全部历史 run；未删除任何现有实验目录
