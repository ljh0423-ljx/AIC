# AUTODL_WEB_FIX_REPORT — 迁移问题修复报告

- **修复日期**: 2026-08-18
- **项目根目录**: `/root/autodl-tmp/AIC2026_AgriVision/`
- **修复范围**: 仅修复 AUTODL_ENV_CHECK.md 发现的 Web 运行环境与演示资源路径问题。
- **约束遵守**: **未训练 / 未重新训练 / 未调参 / 未修改 best_model / 未修改原始数据集 / 未访问 TEST / 未重新评估 TEST / 未做模型转换 / 未做 RK3588 部署 / 未修改 PaddleDetection 核心源码 / 未启动 Web / 未执行批量 GPU 推理。**

---

## 一、问题 1：Gradio 缺失 — ✅ 已解决

- **根因**: AutoDL 环境未安装 gradio，`web/app.py` 模块级 `import gradio` 失败，Web 无法启动。
- **处理**: `python -m pip install "gradio==6.24.0"`（与本地已验证版本一致）。
- **验证**:
  - `import gradio` → **gradio 6.24.0** ✅
  - `import web.app` 成功（不再 ModuleNotFoundError）✅
- **核心环境未动**: 安装前后 paddlepaddle-gpu **2.6.2**、numpy **1.26.4**、pandas 3.0.5、matplotlib 3.9.2、opencv-python 4.5.5、Pillow 11.0.0、pycocotools 2.0.8 均未变更 ✅

## 二、问题 2：demo_manifest 的 source_path 为 Windows 绝对路径 — ✅ 已解决

- **根因**: `web/demo_data/demo_manifest.json`（及 CSV）中 15 个案例的 `source_path` 为 Windows 绝对路径
  `D:\Fruit\AIC2026_AgriVision\dataset\processed_detection_exiffix\images\val\...`；
  `web/app.py::_load_demo_cases` 用 Linux 机器绝对路径做子串校验 → **15/15 全部被跳过**，演示页显示"暂无演示数据"。
- **修复（3 处，均为演示资源路径映射，未改检测逻辑/未碰 VAL 原图/未访问 TEST）**:

  | 文件 | 修改 |
  |---|---|
  | `web/demo_data/demo_manifest.json` | 15 条 `source_path` 由 Windows 绝对路径改为**项目根目录内相对路径** `dataset/processed_detection_exiffix/images/val/<file_name>`（全部 15 个文件经校验确实存在于该 VAL 目录，映射真实可信；其余字段原样保留） |
  | `web/demo_data/demo_manifest.csv` | 同步更新 `source_path` 列为相同相对路径（保持一致） |
  | `web/app.py::_load_demo_cases` | 安全校验由"Linux 机器绝对路径子串"改为**便携路径段判断** `"/images/val/" not in f"/{src}"`——兼容 Windows/Unix 任意部署位置，仍严格禁止 `images/test` / TEST 来源 |
  | `web/make_demo_data.py` | 生成器 `source_path` 由 `str(src)`（绝对路径）改为 `str(src.relative_to(_BASE_DIR))`（项目内相对路径），避免未来重新生成时回归 Windows 绝对路径；**未运行该脚本** |

- **关键点**: 未硬编码任何 Windows 或 Linux 机器绝对路径到代码；相对路径映射即"文件名/相对路径映射"，跨机器可移植。
- **验证**: `_load_demo_cases()` → **DEMO_CASES = 15**；`_demo_asset_check()` → **original 15/15、result 15/15、TOTAL 15/15**；重点案例 **grape_03_black_rot（第15张）与 tomato_04_mosaic_dense** 的 original+result 均正常解析到 `web/demo_cache/` 净化副本 ✅
- **残留扫描**: `web/demo_data/` 与 `web/inference` 运行时 Python 代码中 **Windows 盘符残留 = 0** ✅
- 说明: 演示原图/检测结果 JSON/图片均未改动，未重新生成任何检测结果。

## 三、问题 3：matplotlib 中文字体 — ✅ 已解决

- **根因**: 系统无中文字体（fc-list 为空）；且 `fonts-noto-cjk` 提供的 `.ttc` 集合文件 matplotlib 默认不扫描、`addfont` 只取到 "Noto Sans CJK JP"（JP 字形），简体中文图表标题会显示方框。
- **处理（两步）**:
  1. `apt-get install -y fonts-noto-cjk`（20220127+repack1-1）——系统字体层已可用（供系统/浏览器渲染）。
  2. `web/app.py::_setup_matplotlib_font` 增加**回退注册**：系统候选字体检测不到时，注册项目自带
     `PaddleDetection/ppdet/utils/simfang.ttf`（注册后字体名 **FangSong / 仿宋**，简体中文正体，与 visualize.py 叠加层用字一致，随项目上传、无需联网）。
- **验证**:
  - 真实函数调用后 `matplotlib.rcParams["font.sans-serif"][0]` = **FangSong** ✅
  - 用真实 `_class_chart_figure()` 渲染含中文柱状图（番茄早疫病/苹果黑星病/葡萄黑腐病）成功 ✅
  - fontTools cmap 校验 simfang.ttf 覆盖图表用到的全部中文字形（番/茄/早/疫/病/苹/果/黑/星/葡/腐/目/标/数）✅
- **未修改** 模型、检测逻辑、PaddleDetection 源码。

## 四、修复后验证（综合）— ✅ 全部通过

| # | 验证项 | 结果 |
|---|---|---|
| 1 | `python -m py_compile web/app.py web/backend.py inference/*.py` | **COMPILE_OK** |
| 2 | `import gradio` 版本 | **gradio 6.24.0** |
| 3 | 演示资源自检 | **15/15（original 15/15, result 15/15）**，含 grape_03、tomato_04 |
| 4 | `web.backend.self_check()` | 模型文件 ✅ / 配置文件 ✅ / 类别映射 13 类（val.json）✅ / **推理后端 paddle_gpu | device=gpu | paddle=2.6.2 | 输入 640×640** ✅ |
| 5 | AGRI_DEVICE 默认值 | 环境变量未设置（None）→ `get_device_mode()=auto` → `resolve_device(auto)=gpu`（RTX 4090 自动解析为 GPU）✅ |
| 6 | TEST 未被访问 | 静态扫描：web/inference 代码仅注释/错误提示提及 test.json，唯一 "TestDataset" 为 PaddleDetection 配置键名（指向 images/val + val.json），无 TEST 数据访问路径；运行时 `sys.addaudithook` 审计 demo 自检 + backend 自检全过程 **未打开任何 test.json / images/test 文件** ✅ |

## 五、修改文件清单

| 文件 | 类型 | 说明 |
|---|---|---|
| `web/app.py` | 代码 | `_load_demo_cases` 便携校验；`_setup_matplotlib_font` simfang.ttf 回退 |
| `web/make_demo_data.py` | 代码 | source_path 输出改为项目内相对路径 |
| `web/demo_data/demo_manifest.json` | 数据 | 15 条 source_path 相对化 |
| `web/demo_data/demo_manifest.csv` | 数据 | source_path 列相对化 |

（另有系统级变更：pip 安装 gradio 6.24.0 及其依赖；apt 安装 fonts-noto-cjk；重建 matplotlib 字体缓存 `~/.cache/matplotlib`。）

## 六、GPU 后端状态

```
RTX 4090 (24 GB) | paddlepaddle-gpu 2.6.2 (cu118) | CUDA 11.8 / cuDNN 8.6(编译)/9.1(运行时)
PaddleBackend.load() 成功 | backend=paddle_gpu | device=gpu | input 640 | num_classes=13
web.backend.self_check 全部 OK
AGRI_DEVICE 默认 auto → 自动解析为 gpu（无需改代码）
```

## 七、剩余问题 / 提示

1. **服务监听地址**：`web/app.py` 默认 `server_name="127.0.0.1"`，端口 7860。需要外部访问时须改为 `0.0.0.0` 并在 AutoDL 控制台做 7860 端口映射（或使用 `--port` 调整端口）——本阶段未改动。
2. **演示页展示**：15/15 案例已可加载；首次访问时 `resolve_demo_asset_path` 会重建/校验 `web/demo_cache/`（已存在 30 个净化副本，本次验证确认可正常解析）。
3. **matplotlib 中文字体优先级**：当前环境最终选择 FangSong（simfang.ttf）。若未来系统装好 `.ttf` 形式的简体中文字体，`_CHINESE_FONT_CANDIDATES` 会优先命中系统字体。
4. **未做之事**（留待后续阶段）：未启动正式 Web、未执行批量/GPU 推理、未重新生成演示检测结果、未做 RK3588 部署。

*修复完成，等待下一步指令。*
