# WEB_DEMO_FINAL_REPORT — 比赛答辩版农业病害智能检测 Web 系统

- **日期**: 2026-08-17
- **阶段**: Web 界面优化 + 农业场景展示 + 演示数据 + Backend 抽象（**未训练、未调参、未修改 checkpoint/数据集、未访问 TEST、未做 RK3588 转换/部署**）
- **最终模型（锁定）**: PP-YOLOE+-m `experiments/direction_m/M_SCALEUP_100e/checkpoints/best_model.pdparams`
- **配置**: `configs/ppyoloe_plus_crn_m_100e_agrivision.yml`
- **环境**: Windows 11 · Python 3.12.4 · PaddlePaddle 3.3.0（CPU）· Gradio 6.24.0

---

## 1. Web 界面完成内容

| 项 | 说明 |
|---|---|
| 首页标题 | **🌾 AI 农业病害智能检测系统**（渐变标题）+ 英文副标题，突出展示 |
| 真实信息徽章 | 模型 **PP-YOLOE+-m** · **13 类病害检测** · 推理设备（CPU，实测）· 后端（paddle_cpu） |
| 左侧操作区 | 单张/多张/拖拽上传、置信度阈值（默认 0.5，标注非训练参数）、🚀 开始检测、🗑 清空任务 |
| 结果统计条 | 图片数(成功/失败)、总目标数、平均置信度、平均耗时、**批次真实 FPS**（实时更新，无虚假指标） |
| 结果页签 | 可视化结果（gallery + 原图 vs 检测对比 + 单图明细）· 检测结果表格（序号/类别/class_id/置信度/bbox）· 类别统计（BarPlot + 明细）· 批次汇总 · 结果下载（图片/JSON/CSV/统计）· **农业病害场景** |
| 信息卡片 | 系统信息（设备/版本/输入尺寸/类别数，如实显示 CPU，未虚构 GPU/NPU）· 模型评测（VAL=0.428 选模型依据；TEST=0.417 标注"独立 TEST 最终评估结果"）· 边缘部署（仅"已预留 RKNNBackend 接口"）· 系统介绍 |
| 数据真实约束 | 未加入任何虚假准确率/识别率/节省成本指标；所有置信度/FPS 均来自实际推理 |

## 2. 农业场景展示内容

新增 **"农业病害场景"** 页签，展示 15 个 VAL 演示案例（番茄 8 / 苹果 4 / 葡萄 3）：

- **概览画廊**：15 张检测图，每张标注 `作物 · demo_id · 目标数 · 最高conf`；
- **逐案例查看**：下拉选择案例 → 原图 vs 检测结果并排 + 详情（场景类型、GT 类别、检测目标与真实置信度/bbox）；
- 场景标签：番茄病害检测 / 苹果叶部病害检测 / 葡萄叶部病害检测（作物级通用描述）；
- 类别名称**严格使用项目 val.json 英文映射**，未擅自创造官方中文类名。

## 3. 演示数据数量与来源

- 目录：`web/demo_data/`（15 张，**仅复制**，未修改原图）+ `web/demo_manifest.json/csv`
- 来源：`dataset/processed_detection_exiffix/images/val/`（**仅 VAL，未使用 TEST**）
- 生成脚本：`web/make_demo_data.py`
- 覆盖：单目标 9 / 多目标 5 / 密集 1；番茄 8、苹果 4、葡萄 3；类别 12/13

## 4. 演示案例实际检测结果（真实推理，conf=0.5）

结果目录：`web/demo_outputs/run_20260817_173307`（正式推理程序 `inference/infer.py` 生成）

| demo_id | 作物/场景 | GT类别(数) | 实际检测结果（置信度） | 耗时(ms) |
|---|---|---|---|---|
| tomato_01_early_blight | 番茄/单目标 | Early blight(1) | Septoria leaf spot 0.61; bacterial spot 0.52 | 1630 |
| tomato_02_septoria | 番茄/多目标 | Septoria(3) | Septoria 0.67; Septoria 0.64 | 1661 |
| tomato_03_septoria | 番茄/单目标 | Septoria(1) | Septoria 0.53 | 1637 |
| tomato_04_mosaic_dense | 番茄/密集 | mosaic virus(12) | Tomato leaf 0.57/0.54/0.51 | 1763 |
| tomato_05_yellow_virus | 番茄/多目标 | yellow virus(5) | yellow virus 0.55/0.55/0.53/0.51 | 1723 |
| tomato_06_mold | 番茄/多目标 | mold(4) | late blight 0.59; mold 0.56 | 1725 |
| tomato_07_bacterial_spot | 番茄/单目标 | bacterial spot(1) | Septoria 0.60 | 1662 |
| tomato_08_late_blight | 番茄/多目标 | late blight(2) | late blight 0.90/0.89 | 1640 |
| apple_01_scab | 苹果/多目标 | Scab(6) | Apple Scab Leaf 0.57 | 1652 |
| apple_02_rust | 苹果/单目标 | rust(1) | Apple rust leaf 0.84 | 1751 |
| apple_03_rust | 苹果/单目标 | rust(1) | Apple rust 0.82; black rot 0.52 | 1652 |
| apple_04_healthy_leaf | 苹果/单目标 | Apple leaf(1) | Apple leaf 0.94 | 1657 |
| grape_01_healthy_leaf | 葡萄/单目标 | grape leaf(1) | grape leaf 0.93 | 1671 |
| grape_02_black_rot | 葡萄/单目标 | black rot(1) | grape leaf black rot 0.94 | 1704 |
| grape_03_black_rot | 葡萄/单目标 | black rot(1) | grape leaf black rot 0.90 | 1651 |

> 以上均为模型真实输出，**未人为提高置信度**；部分案例存在类间混淆（Septoria↔bacterial spot 等），为模型真实能力表现，如实展示。
> 汇总：15 张 / 成功 15 / 失败 0 / 目标 25 / **平均耗时 1.697 s/张 / 实际 FPS 0.59**。

## 5. 真实平均 FPS

- 演示批量（15 张）：总耗时 **25.46 s**，平均 **1.697 s/张**，**FPS = 0.59**（真实计时）。
- Web 直接测试（8 张，系统预热后）：FPS 0.41~0.66；单图 ~2.3 s。
- **所有 FPS 均由 time.perf_counter 计算，未写死/未虚构；未将 CPU FPS 冒充 RK3588 指标。**

## 6. Backend 架构

- **统一接口** `DetectorBackend`（`inference/detector.py`）：`load() / predict() / predict_batch() / info() / close()`，兼容别名 `infer()/infer_batch()`；输出统一 `postprocess.Detection`。
- **注册机制**：`register_backend(name)` 装饰器 + `create_backend(kind, ...)` 注册表驱动。
- **已注册后端**：
  - `paddle_cpu` / `paddle_gpu`（PaddleBackend，默认；`paddle` 别名按 device 解析）
  - `rknn_reserved`（RKNNBackend 占位，构造即 `NotImplementedError`，中文提示）
- **解耦边界**：Web/UI 仅依赖 DetectorBackend 接口与 `Detection` 结构；`batch_infer.py` 通过 `detector.info()` / `detector.catid2name` 读取统一信息，不访问后端私有字段。

## 7. RK3588 后续迁移接口（未执行）

- 迁移仅需：实现 `RKNNBackend(DetectorBackend)`（Paddle→ONNX→RKNN-Toolkit2 INT8）并 `register_backend("rknn", ...)`；
- **无需改动**：`web/` 界面、`postprocess.py`、`visualize.py`、`batch_infer.py` 统计模块；
- 当前 `rknn_reserved` 为清晰占位（NotImplemented），**未声称完成 RK3588 部署**。

## 8. 验证结果

| 验证项 | 结果 |
|---|---|
| 启动 Web / 页面正常 | ✅ 根页面 HTTP 200，自检全过，模型只加载一次 |
| demo_data 批量推理 | ✅ 15/15 成功，25 目标，FPS 0.59 |
| 原图/检测图/类别统计/结果表格 | ✅ 均正常（含每图 JSON/TXT/CSV 与批次文件） |
| JSON/CSV 下载 | ✅ 预测/类别统计/汇总文件齐全 |
| 单图拖拽推理 | ✅ 单图正常（Apple Scab 0.57 等） |
| 错误图片不中断批次 | ✅ 损坏图 → failed=1，批次继续 |
| TEST 目录/test.json 未被访问 | ✅ demo 数据仅 VAL；推理日志无 test.json 引用 |
| best_model.md5 未变化 | ✅ `18bd0e99329be2113f57280188c227b6` |
| 原始 VAL 数据未修改 | ✅ 原图 mtime 未变；demo_data 为复制件（大小一致） |
| PaddleDetection 源码未修改 | ✅ 本阶段仅新增 inference/ 与 web/ 文件 |
| 未做模型转换/RK3588 部署 | ✅ 未执行 |

## 9. 已知限制

1. 本地 CPU 推理约 0.4~0.7 FPS（非实时），演示以单图/小批量为宜；
2. 上传文件名统一为 `upload_001.jpg` 等（Gradio 不保留原名）；
3. 项目无官方中文类名映射，类别默认英文（`CLASS_NAMES_CN` 填写后自动启用双语）；
4. 本执行环境（沙箱）拦截对 127.0.0.1 的 httpx 预启动探测，Web 已内置兼容回退自动处理；用户正常终端不受影响；
5. 演示数据若重新生成需重新执行 `make_demo_data.py` + 推理，并重启 Web 生效。

## 10. 新增/修改文件清单

- 新增：`web/make_demo_data.py`、`web/make_demo_report.py`、`web/demo_data/`（15 图 + manifest）、`web/demo_outputs/run_20260817_173307/`、`web/DEMO_CASES_REPORT.md`、`web/WEB_DEMO_FINAL_REPORT.md`
- 修改：`inference/detector.py`（接口扩展 + 注册表 + RKNN 占位 + list_backends）、`inference/batch_infer.py`（info/catid2name 最小适配）、`web/backend.py`（info() 解耦）、`web/app.py`（UI 优化 + 农业场景 + 统计条 + 比赛演示模式 + 农业结果卡 + 边缘部署状态 + 启动兼容回退）、`web/_direct_test.py`（验证脚本）

---

# 附录：比赛答辩最终版功能增强（2026-08-17）

## 11. 比赛演示模式

- **入口**：结果区新增 **"🎬 比赛演示模式"** Tab，使用 `web/demo_data/` 的 15 张 VAL 演示图（**仅 VAL，未读取/复制 TEST**）。
- **演示顺序**：番茄 → 苹果 → 葡萄 → 密集场景（`tomato_04_mosaic_dense` 最后）。
- **功能**：◀ 上一张 / 下一张 ▶ / ▶ 自动播放（每 2s 前进一张）/ ⏹ 停止演示 / 🔄 重新检测当前案例。
- **每个案例展示**：案例编号、作物、场景类型、原图、检测结果图、检测目标数、类别、真实 confidence、真实 bbox、**实际单图推理耗时**（来自 demo_outputs 真实记录）。
- **自动播放**：全部使用已有真实结果，**不虚构耗时/FPS**；模型只初始化一次。
- **重新检测当前案例**：仅使用当前 VAL demo 图片，复用已加载模型（不重复加载），显示本次真实推理耗时（如 2684 ms）。

## 12. 农业结果卡（🌱 农业视觉检测结果）

- **位置**：每次检测后、结果页签上方，动态生成。
- **内容**（全部来自真实模型输出，不虚构诊断）：
  - 场景标签（番茄病害检测 / 苹果叶部病害检测 / 葡萄叶部病害检测，按 13 类映射推断）；
  - 作物类别（按 class_id 1-8 番茄 / 9-11 苹果 / 12-13 葡萄推断；多作物混合或无法判断显示"未确定"）；
  - 检测到的病害类别（按目标数从高到低）+ 目标总数 + 最高/平均置信度；
  - 无目标时明确显示"当前置信度阈值下未检测到目标"；
  - **免责声明**："以上为 AI 视觉检测结果，不等同于专业植保诊断"。
- **约束**：不使用"确诊/治疗方案"等表述；不根据 confidence 判断严重程度；不虚构病害风险等级/经济损失/产量影响/防治建议。

## 13. RK3588 边缘部署状态预留（⚡ 边缘部署状态）

- **位置**：首页右侧系统信息区。
- **内容**（从 DetectorBackend/注册表读取真实信息，不硬编码）：
  - 当前后端：**paddle_cpu**（CPU）· 状态：**当前运行**；
  - Paddle GPU：接口已预留（`paddle_gpu` 已注册）；
  - RK3588 / RKNN：接口已预留，**尚未部署**（`rknn_reserved` 已注册占位）；
  - 已注册后端：`paddle_cpu、paddle_gpu、rknn_reserved`。
- **约束**：不显示"已部署/NPU加速完成/RK3588 FPS"；选择/使用 `rknn_reserved` 仅提示"当前阶段未实现 RKNN 后端"，**不会尝试模型转换**。

## 14. 最终验证结果（12 项）

| # | 验证项 | 结果 |
|---|---|---|
| 1 | 正常打开首页 | ✅ 根页面 HTTP 200，自检全过，模型只加载一次 |
| 2 | 进入比赛演示模式并浏览 15 案例 | ✅ 演示顺序与案例完整（番茄→苹果→葡萄→密集） |
| 3 | 上一张/下一张/自动播放/停止/重新检测 | ✅ 全部正常；重新检测返回真实耗时（2684 ms） |
| 4 | 随机检测 1 张 VAL 图，农业结果卡随真实结果变化 | ✅ Apple Scab 单图 → 结果卡显示"苹果 + Apple Scab Leaf 1目标 + 最高/平均置信度 0.570" |
| 5 | 无检测目标时结果卡显示"未检测到目标" | ✅ 已断言验证 |
| 6 | RK3588 状态准确且不触发模型转换 | ✅ 显示"接口已预留、尚未部署"；rknn_reserved 仅占位提示 |
| 7 | JSON/CSV 下载仍正常 | ✅ 批量 13 个下载文件（predictions/class_stats/汇总/可视化） |
| 8 | best_model.pdparams MD5 不变 | ✅ `18bd0e99329be2113f57280188c227b6` |
| 9 | 原始 VAL 数据 mtime 不变 | ✅ `Aug 17 12:35`（本次开发期间未变） |
| 10 | test.json / test 目录零访问 | ✅ test.json mtime `Aug 17 12:38` 未变；演示数据全部来自 images/val（脚本断言 0 TEST 路径） |
| 11 | PaddleDetection 核心源码零修改 | ✅ 本次 0 个 .py 文件被修改 |
| 12 | 已有实验目录零删除/零覆盖 | ✅ experiments 16 目录保留；inference/INFERENCE_VALIDATION_REPORT.md、web/DEMO_CASES_REPORT.md 保留；inference 4 run / demo 1 run 未覆盖 |

**实际耗时/FPS（真实计时）**：演示批量 15 张 25.46 s / FPS 0.59；Web 直接批量 8 张 21.86 s / FPS 0.37；单图 ~2.3 s；重新检测单图 2684 ms。**未虚构 FPS，未将 CPU FPS 冒充 RK3588**。

