# AUTODL_ENV_CHECK — AutoDL 迁移后环境验收报告

- **验收日期**: 2026-08-18
- **项目根目录**: `/root/autodl-tmp/AIC2026_AgriVision/`
- **验收性质**: 只读检查。**未训练 / 未重新训练 / 未调参 / 未修改 best_model / 未修改原始数据集 / 未访问 TEST / 未做模型转换 / 未删除任何文件 / 未启动正式 Web / 未执行 GPU 推理**。
- **验收方式**: 目录与文件核对、`py_compile` 语法检查、依赖版本核对、`nvidia-smi`、Paddle CUDA 静态检测、git 审计、`load_config` + `PaddleBackend.load()` 生产链路模型加载（仅加载权重，无 forward）。

---

## 一、目录结构与迁移完整性 — ✅ 通过

| 关键路径 | 状态 | 说明 |
|---|---|---|
| `web/` + `web/app.py` + `web/backend.py` | ✅ | app.py 67 KB / backend.py 6.8 KB |
| `web/demo_data/` | ✅ | 15 张 VAL 原图 + demo_manifest.json/csv |
| `web/demo_outputs/` | ✅ | `run_20260817_173307/` 完整（visualized 15 / detections 45 / originals / 统计文件） |
| `inference/`（config/detector/batch_infer/postprocess/visualize/infer + `__init__.py`） | ✅ | 齐全 |
| `configs/ppyoloe_plus_crn_m_100e_agrivision.yml` | ✅ | 顶层配置存在 |
| `PaddleDetection/` | ✅ | 完整仓库 + editable 安装 |
| `experiments/direction_m/M_SCALEUP_100e/checkpoints/best_model.pdparams` | ✅ | 94.3 MB（94,337,197 B），与清单一致 |
| `dataset/processed_detection_exiffix/annotations/val.json` | ✅ | 87 KB，categories 13 |
| `scripts/`（43 项）/ `tools/` / `experiments/final_evaluation/` | ✅ | 存在（final_evaluation 未触碰，E 类） |

**补充**: `best_model/` 子目录含 `model.pdparams`（94.3 MB）；checkpoints 下另有 `model_final.*` 备用权重与多份中间 checkpoint（均未修改）。

---

## 二、语法检查（py_compile） — ✅ 通过

```
python -m py_compile web/app.py web/backend.py inference/*.py  →  COMPILE_OK
```

---

## 三、依赖版本检查 — ⚠️ 1 项缺失（Web 启动阻塞）

| 依赖 | 要求 | 实际 | 状态 |
|---|---|---|---|
| Python | 3.8–3.12（训练 3.12.3） | 3.12.3 | ✅ |
| paddlepaddle-gpu | ==2.6.2（cu118） | 2.6.2 | ✅ |
| paddledet | — | 0.0.0（editable → `PaddleDetection/`） | ✅ |
| matplotlib | ≥3.3 | 3.9.2 | ✅ |
| pandas | ≥1.5 | 3.0.5 | ✅（版本较新，推理不涉及，风险低） |
| numpy | ≥1.21 | 1.26.4 | ✅ |
| opencv-python | ≥4.5 | 4.5.5.64 | ✅ |
| Pillow | ≥9.0 | 11.0.0 | ✅ |
| pycocotools | ≥2.0 | 2.0.8 | ✅ |
| tqdm | ≥4.60 | 4.66.2 | ✅ |
| protobuf | — | 5.28.3 | ✅ |
| **gradio** | **>=6.0** | **未安装** | ❌ **阻塞** |

> `web/app.py` 模块级 `import gradio` → 当前 `import web.app` 直接 `ModuleNotFoundError`。**启动 Web 前必须安装 gradio（本地版本 6.24.0 可参考）。**

---

## 四、GPU / CUDA — ✅ 通过

```
nvidia-smi:  NVIDIA GeForce RTX 4090 | 24564 MiB | 当前占用 0 MiB | Driver 580.76.05
```

| 检查项 | 结果 |
|---|---|
| GPU 型号 | NVIDIA GeForce RTX 4090 ✅ |
| 显存 | 24 GB（24564 MiB），当前空闲 ✅ |
| `paddle.is_compiled_with_cuda()` | **True** ✅ |
| `paddle.device.cuda.device_count()` | 1 ✅ |
| `paddle.device.cuda.get_device_name(0)` | NVIDIA GeForce RTX 4090 ✅ |
| `paddle.device.get_device()` | `gpu:0` ✅ |
| `paddle.set_device("gpu:0")` | 成功 ✅ |
| paddle 编译期 CUDA | 11.8 ✅ |
| paddle 编译期 cuDNN | 8.6.0（运行时检测 9.1，兼容，正常） |
| compute capability | (8, 9) ✅ |
| `resolve_device("auto")` | → **gpu** ✅（生产路径自动识别） |

---

## 五、Paddle 版本 — ✅ 与训练/TEST 环境一致

- **paddlepaddle-gpu 2.6.2（cu118）** — 与清单 §3「训练/TEST 环境即 2.6.2」一致，为官方验证环境。
- `paddledet 0.0.0` editable 安装指向本工程 `PaddleDetection/`。

---

## 六、PaddleDetection 版本 / 代码兼容性 — ✅ 兼容

- **git HEAD = `b25522a0f4bde8c80603f3ba5e3472059972e3b5`**，与迁移清单 §3 锁定的 commit 完全一致。
- `git diff` 名义上 2011 文件变更，但**全部为 CRLF→LF 行尾差异**（`git ls-files --eol` → `i/lf w/crlf`），非内容改动。
- **真实内容改动仅 3 个文件**（`git diff -w`）：
  - `ppdet/modeling/heads/ppyoloe_head.py`：CGPM / class-weighted VFL(B1) / density calibration(D1)，**均为训练期特性，默认关闭**；
  - `ppdet/modeling/architectures/yolo.py`：S1 CG-ACS 图像级辅助分类头，训练期使用，推理旁路；
  - `ppdet/modeling/heads/__init__.py`：新增 1 行 import（`cgacs_aux_cls_head`）。
- 依赖模块存在：`ppdet/modeling/losses/confusion_margin_loss.py`、`ppdet/modeling/heads/cgacs_aux_cls_head.py` ✅。
- **结论**：当前 PaddleDetection 代码 = 训练 best_model 的那份代码，推理链路兼容由定义成立；推断时上述训练开关全部为默认关闭，不影响前向。

---

## 七、模型配置 `_BASE_` 链解析 — ✅ 全部可解析

顶层 `configs/ppyoloe_plus_crn_m_100e_agrivision.yml` 的 5 个 `_BASE_` 引用全部存在且已随真实代码路径解析成功（`ppdet.core.workspace.load_config` 无报错）：

```
configs/ppyoloe_plus_crn_m_100e_agrivision.yml
  └── ./datasets/agrivision_detection.yml                    ✅（无嵌套 _BASE_）
  └── ../PaddleDetection/configs/runtime.yml                 ✅
  └── ../PaddleDetection/configs/ppyoloe/_base_/optimizer_80e.yml      ✅
  └── ../PaddleDetection/configs/ppyoloe/_base_/ppyoloe_plus_crn.yml   ✅
  └── ../PaddleDetection/configs/ppyoloe/_base_/ppyoloe_plus_reader.yml ✅
```

> 说明：链为单层，无嵌套引用；`configs/` 与 `PaddleDetection/` 相对位置保持即可解析。

---

## 八、模型加载（best_model）— ✅ 成功加载（仅 load，未推理）

走生产代码路径（等价 `web/backend.get_detector()`，绕过 gradio）：

```
resolve_device("auto") → gpu
PaddleBackend.load() 成功（耗时约 3.5s）
  backend = paddle_gpu | device = gpu | paddle = 2.6.2
  input_size = 640 | num_classes = 13
  model_path = .../experiments/direction_m/M_SCALEUP_100e/checkpoints/best_model.pdparams
  anno_path  = .../dataset/processed_detection_exiffix/annotations/val.json
  config_path= .../configs/ppyoloe_plus_crn_m_100e_agrivision.yml
```

- `paddle.load` + `model.set_dict` 成功，**无 missing / unexpected 键**，权重与架构完全匹配。

---

## 九、13 类映射来源 — ✅ 确认来自 `annotations/val.json`

- `inference/config.py`：`ANNO_REL_PATH = "annotations/val.json"`，`DATASET_DIR = dataset/processed_detection_exiffix`（明确注释禁止 test.json / TEST）。
- `build_class_mapping()` 严格从 val.json categories 升序构建，不重新编号；实测加载出 13 类：
  `Tomato Early blight leaf / Tomato Septoria leaf spot / Tomato leaf / Tomato leaf bacterial spot / Tomato leaf late blight / Tomato leaf mosaic virus / Tomato leaf yellow virus / Tomato mold leaf / Apple Scab Leaf / Apple leaf / Apple rust leaf / grape leaf / grape leaf black rot`。
- val.json categories id = **1–13**（COCO 1-based）；`label_list.txt` 为 0–12。二者差 1 为预期设计：`clsid2catid` 将模型输出 0-12 索引对齐到 COCO id 1-13，已由加载结果验证正确。

---

## 十、演示资源检查 — ⚠️ 文件齐全，但 Linux 下会被全部跳过（详见问题 #2）

| 资源 | 核对结果 |
|---|---|
| `web/demo_data/` 原图 | 15/15 存在，与 demo_manifest.json 的 `file_name` 一一对应 ✅ |
| `web/demo_outputs/run_20260817_173307/visualized/` | 15/15 覆盖全部案例（按 file_name stem）✅ |
| `web/demo_outputs/run_20260817_173307/detections/` | 15 案例 × 3 格式 = 45 文件 ✅ |
| `web/demo_outputs/.../predictions.json / class_statistics.csv / inference_summary.json / inference.log` | 存在 ✅ |
| `web/demo_cache/` | 30 文件（original+result 净化副本），目录可写 ✅ |
| manifest 来源安全校验（`_load_demo_cases`） | ❌ 15/15 被跳过（见问题 #2） |

---

## 十一、Web 启动条件

**必须项（当前不满足）：**
1. **安装 gradio >= 6.0** —— 当前缺失，`import web.app` 失败。参考本地 6.24.0：`pip install "gradio>=6.0"`。

**已满足项：**
2. 依赖（除 gradio）全部就绪：paddle 2.6.2 cu118 + matplotlib 3.9.2 + pandas/numpy/cv2/Pillow/pycocotools ✅
3. 模型/配置/类别映射/后端加载链路验证通过 ✅（GPU）
4. `web/backend.py` 当前 device 配置：`os.environ.get("AGRI_DEVICE", "auto")`，默认 **auto 自动检测** → 本机解析为 gpu，**无需改代码**。（注意：迁移清单 §11「需把 backend.py 第 47 行 device=cpu 改 gpu」的描述已过时，实际代码已升级为 auto 检测 + 环境变量切换。）
5. 路径：代码全部基于 `BASE_DIR` 相对路径（`Path`），无硬编码 Windows 盘符；软链 1 个（`PaddleDetection/dataset/processed_detection` → exiffix）指向有效且仅训练用，推理不依赖。
6. `NO_PROXY=127.0.0.1,localhost` 已由 app.py 顶部设置，Linux 无害。
7. 服务默认绑定 `127.0.0.1:7860`；如需外部访问需改为 `0.0.0.0` 并在 AutoDL 控制台做 7860 端口映射（启动参数 `--port` 支持改端口）。

**辅助建议（非阻塞）：**
8. 系统中文字体缺失（见问题 #3）—— 建议 `apt install fonts-noto-cjk`。

---

## 十二、发现的问题汇总

| # | 级别 | 问题 | 影响 | 建议（本阶段未修改） |
|---|---|---|---|---|
| 1 | **严重** | **gradio 未安装** | `web/app.py` 模块级 `import gradio` 失败 → Web 无法启动 | 启动前 `pip install "gradio>=6.0"`（本地 6.24.0 已验证） |
| 2 | **严重** | **demo_manifest.json 的 `source_path` 是 Windows 绝对路径**（`D:\Fruit\AIC2026_AgriVision\...`）；`_load_demo_cases` 用 Linux 数据集路径做子串校验 → **15/15 案例全部被跳过** | 农业病害场景/比赛演示模式页签将显示"暂无演示数据"（尽管 demo_data/demo_outputs 文件齐全） | 在 AutoDL 上重新生成 manifest（运行 `web/make_demo_data.py`）或调整校验逻辑；本阶段未改动 |
| 3 | 次要 | 系统无中文字体（fc-list 无 CJK；`_CHINESE_FONT_CANDIDATES` 候选 Microsoft YaHei/SimHei/Noto CJK 等均不存在） | matplotlib 类别统计柱状图中文标题/轴标签显示为方框；**可视化叠加层不受影响**（用 `PaddleDetection/ppdet/utils/simfang.ttf`，文件存在 ✅） | `apt install fonts-noto-cjk`（清单 §12 已预告） |
| 4 | 提示 | 迁移清单 §11「backend.py 第 47 行 device=cpu 需改一行」**已过时** | 无（实际更好） | 实际代码已用 `AGRI_DEVICE` 环境变量 + `auto` 自动检测，无需改动；以本报告为准 |
| 5 | 提示 | `web/requirements_web.txt` 未列 matplotlib（清单 §11 亦指出） | 环境已装 3.9.2，实际无影响 | 可选：补入 requirements 保持文档一致 |
| 6 | 提示 | PaddleDetection 工作树为 CRLF 行尾 → git 显示 2011 文件"变更"（非内容）；3 个真实改动文件均为训练期特性、默认关闭 | 不影响推理 | 无需处理；如需 git 干净可用 `git add --renormalize`（本阶段未执行） |
| 7 | 提示 | pandas 3.0.5 较新（要求 ≥1.5）；cuDNN 编译期 8.6.0 vs 运行时 9.1 | 推理链路实测正常，无影响 | 观察即可 |

---

## 十三、总体结论

- **模型/推理核心链路完全就绪**：目录完整、语法通过、依赖（除 gradio）齐全、Paddle 2.6.2 cu118 + RTX 4090 + CUDA 可用、PaddleDetection commit 与训练环境一致、`_BASE_` 链解析成功、best_model 加载成功、13 类映射来自 val.json 且验证正确。
- **Web 启动前需解决 2 个严重问题**：① 安装 gradio；② demo 演示数据在 Linux 下会被 source_path 校验全部跳过（如需保留演示/农业场景功能）。
- **未执行任何被禁止的操作**：未训练/调参/改模型/改数据/访问 TEST/模型转换/删除文件；未启动正式 Web；未执行 GPU 推理（仅做了模型权重加载验证）。
- 检查完毕，等待下一步指令。
