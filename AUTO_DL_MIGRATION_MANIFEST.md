# AUTO_DL_MIGRATION_MANIFEST — Windows 本地 Web 系统 → AutoDL Linux + RTX 4090 GPU 迁移清单

- **审计日期**: 2026-08-17
- **审计性质**: 只读清单审计（**未修改任何代码/模型/数据，未训练，未访问 TEST，未做模型转换**）
- **迁移目标**: 把本地已验收的完整农业病害检测 Web 系统迁移到 AutoDL Linux，用 RTX 4090 GPU 推理，独立启动完整 Gradio Web
- **最终模型（锁定）**: `experiments/direction_m/M_SCALEUP_100e/checkpoints/best_model.pdparams`（94.3 MB）

---

## 0. 依赖树（按实际 import 追踪）

```
web/app.py
 ├─ gradio(>=6.0)  pandas  numpy  matplotlib(AutoDL 需安装)  cv2  PIL  json  shutil  threading
 ├─ inference.config          → BASE_DIR/configs/…/模型/数据集 路径
 ├─ inference.batch_infer     → run_batch/read_image/discover_images
 ├─ inference.detector        → create_backend/DetectorBackend/DetectorError
 ├─ web.backend               → get_detector()（DetectorBackend 单例）
 └─ web.config                → UI 常量 + 固定模型记录(VAL/TEST/参数量)

inference/detector.py
 ├─ paddle(懒加载, 版本检测)  ppdet.core.workspace(load_config/create/merge_config)
 ├─ ppdet.data.transform.operators(Resize/NormalizeImage/Permute)
 ├─ ppdet.metrics.coco_utils(get_infer_results)
 ├─ ppdet.utils.check(check_version)
 ├─ inference.config / inference.postprocess
 └─ 模型加载: best_model.pdparams + 模型配置(含 _BASE_ 链) + val.json(13类映射)

inference/visualize.py → PIL + cv2 + 字体(simfang.ttf / Windows 字体)
inference/postprocess.py → numpy  csv  dataclasses
inference/infer.py → argparse  + inference.*（CLI，可选）
web/app.py 图表 → matplotlib（AutoDL 需安装 + 中文字体）
```

**模型配置 `_BASE_` 链（必须同步迁移的相对文件）**：
`configs/ppyoloe_plus_crn_m_100e_agrivision.yml`
→ `./datasets/agrivision_detection.yml`
→ `../PaddleDetection/configs/runtime.yml`
→ `../PaddleDetection/configs/ppyoloe/_base_/optimizer_80e.yml`
→ `../PaddleDetection/configs/ppyoloe/_base_/ppyoloe_plus_crn.yml`
→ `../PaddleDetection/configs/ppyoloe/_base_/ppyoloe_plus_reader.yml`
> 这些相对路径在 AutoDL 上只要保持 `configs/` 与 `PaddleDetection/` 的相对位置不变即可解析（本项目自动以 `BASE_DIR` 定位，无硬编码 Windows 路径）。

---

## 1. 必须上传清单（A 类）— 缺一 Web 无法启动

| 本地相对路径 | 作用 | 依赖代码 | 是否必须 | 大小 | AutoDL 目标路径 |
|---|---|---|---|---|---|
| `web/app.py` | Gradio 主界面 + 事件 | 全部 Web 功能 | 必须 | 65 KB | `AIC2026_AgriVision/web/app.py` |
| `web/backend.py` | DetectorBackend 单例 + 自检 | app.py | 必须 | 5 KB | 同左 |
| `web/config.py` | UI 常量 + 固定模型记录 | app.py | 必须 | 3 KB | 同左 |
| `web/__init__.py` | 包标记 | import | 必须 | <1 KB | 同左 |
| `inference/config.py` | 全部路径（BASE_DIR 相对） | 所有 inference.* | 必须 | 5 KB | `AIC2026_AgriVision/inference/config.py` |
| `inference/detector.py` | PaddleBackend + 注册表 | app/backend | 必须 | 16 KB | 同左 |
| `inference/batch_infer.py` | 批量推理引擎 | app.py | 必须 | 14 KB | 同左 |
| `inference/postprocess.py` | Detection 结构/统计/JSON/CSV | 多处 | 必须 | 7 KB | 同左 |
| `inference/visualize.py` | 可视化 + 中文字体查找 | 检测结果图 | 必须 | 7 KB | 同左 |
| `inference/infer.py` | CLI 推理（可选但建议） | — | 必须* | 6 KB | 同左 |
| `inference/__init__.py` | 包标记 | import | 必须 | <1 KB | 同左 |
| `configs/ppyoloe_plus_crn_m_100e_agrivision.yml` | 最终模型配置 | detector.load | 必须 | 2 KB | `AIC2026_AgriVision/configs/…` |
| `configs/datasets/agrivision_detection.yml` | 数据集/类别配置（_BASE_） | 上者 | 必须 | 1 KB | 同左 |
| `experiments/direction_m/M_SCALEUP_100e/checkpoints/best_model.pdparams` | **最终模型权重** | detector 权重加载 | 必须 | 94.3 MB | `AIC2026_AgriVision/experiments/direction_m/M_SCALEUP_100e/checkpoints/best_model.pdparams` |
| `dataset/processed_detection_exiffix/annotations/val.json` | **13 类映射**（唯一数据源） | build_class_mapping | 必须 | ~87 KB | `AIC2026_AgriVision/dataset/processed_detection_exiffix/annotations/val.json` |
| `PaddleDetection/ppdet/` | PaddleDetection Python 包 | detector 全部 ppdet 调用 | 必须 | 21 MB | `AIC2026_AgriVision/PaddleDetection/ppdet/` |
| `PaddleDetection/configs/runtime.yml` | use_gpu 等运行时 | _BASE_ 链 | 必须 | 1 KB | 同左 |
| `PaddleDetection/configs/ppyoloe/_base_/optimizer_80e.yml` | 优化器配置（_BASE_） | _BASE_ 链 | 必须 | 1 KB | 同左 |
| `PaddleDetection/configs/ppyoloe/_base_/ppyoloe_plus_crn.yml` | 网络结构配置 | _BASE_ 链 | 必须 | 2 KB | 同左 |
| `PaddleDetection/configs/ppyoloe/_base_/ppyoloe_plus_reader.yml` | Reader 配置 | _BASE_ 链 | 必须 | 1 KB | 同左 |
| `PaddleDetection/ppdet/utils/simfang.ttf` | 可视化中文字体（Linux 可用） | visualize.find_font | 必须 | 10 MB | 同左 |
| `web/requirements_web.txt` | Web 依赖清单 | 安装 | 必须 | <1 KB | 同左 |
| `inference/requirements_infer.txt` | 推理依赖清单 | 安装 | 必须 | <1 KB | 同左 |

\* `inference/infer.py` 为 CLI 入口，Web 不 import 它；为便于 AutoDL 上调试建议一并上传。

> **PaddleDetection 上传说明**：上传 `ppdet/` + 上述 4 个配置 + `simfang.ttf`（约 22 MB）即可运行；也可直接上传整个 `PaddleDetection/`（109 MB）保持仓库完整；或在 AutoDL 上 `git clone` 固定 commit `b25522a0` 并补齐本地改动（本项目 ppdet 无源码改动）。

---

## 2. 建议上传清单（B 类）— 保持当前比赛版全部功能与演示效果

| 本地相对路径 | 作用 | 依赖代码 | 大小 | AutoDL 目标路径 |
|---|---|---|---|---|
| `web/demo_data/`（15 张 VAL 原图 + demo_manifest.json/csv） | 农业病害场景/比赛演示模式原图 | _load_demo_cases / resolve_demo_asset_path | 2.0 MB | `AIC2026_AgriVision/web/demo_data/` |
| `web/demo_outputs/run_20260817_173307/visualized/` | 15 张检测结果图 | _demo_gallery / _demo_case_view | 1.8 MB | 同左（整个 run 目录） |
| `web/demo_outputs/run_20260817_173307/detections/` | 每案例检测 JSON（目标数/置信度/耗时） | _load_demo_cases | 75 KB | 同左 |
| `web/demo_outputs/run_20260817_173307/predictions.json` `class_statistics.csv` `inference_summary.json` `inference.log` | 演示批次统计（可选完整性） | — | <0.5 MB | 同左 |
| `web/demo_cache/` | 净化缓存副本（**可再生**，上传可省首次生成） | resolve_demo_asset_path | 3.8 MB | 同左 |
| `dataset/processed_detection_exiffix/`（全量 414 MB） | 未来演示数据重生成 / CLI 使用（Web 运行仅需 val.json） | — | 414 MB | 同左 |
| `experiments/direction_m/M_SCALEUP_100e/checkpoints/model_final.pdparams` | 备用最终权重 | — | 94.3 MB | 同左 |
| `web/make_demo_data.py` `web/make_demo_report.py` `web/_direct_test.py` | 演示生成/验证工具 | — | 16 KB | 同左 |
| `scripts/`（train/eval/infer 等） | 运维/开发脚本 | — | <1 MB | 同左 |
| `tools/yolo_to_coco.py` | 数据转换工具 | — | 4 KB | 同左 |
| `README_AUTODL.md` `PROJECT_FINAL_STATUS.md` `WEB_DEMO_FINAL_REPORT.md` `DEMO_CASES_REPORT.md` `LOCAL_DEPLOYMENT_AUDIT.md` `INFERENCE_VALIDATION_REPORT.md` `WEB_APP_VALIDATION_REPORT.md` | 文档/报告 | — | <1 MB | 同左 |

---

## 3. AutoDL 已有、可复用清单（C 类）

| 资源 | 说明 |
|---|---|
| **PaddlePaddle 2.6.2 (cu118)** | 训练/TEST 环境即 2.6.2；若沿用原训练实例则无需重装；若新实例需 `pip install paddlepaddle-gpu==2.6.2 -i https://www.paddlepaddle.org.cn/whl/linux/linux-gpu-cu118.html` |
| **CUDA / cuDNN**（11.8 + cuDNN 9.1） | 随 Paddle 环境/实例自带 |
| `ppyoloe_crn_m_obj365_pretrained.pdparams`（`/root/.cache/paddle/weights/`） | 训练预训练权重；**Web 推理不需要**（配置 `pretrain_weights` 仅训练使用，Web 以 `-o weights=best_model` 覆盖） |
| PaddleDetection git 仓库 @ `b25522a0`（若原实例仍在） | 可复用代替上传（需确认 ppdet 未改动） |

---

## 4. 不需要上传清单（D 类）— 缓存/日志/临时/CPU 验证产物

| 本地路径 | 说明 |
|---|---|
| `web/outputs/`（61 MB，run_*） | 前端验证 run，可再生 |
| `inference/outputs/`（15 MB，run_*） | CLI 验证 run，可再生 |
| `web/tmp/`（24 MB） | 上传暂存，运行时自动生成 |
| `web/demo_cache/`（3.8 MB） | 演示净化缓存，首次访问自动从 demo_data+demo_outputs 重建 |
| 所有 `__pycache__/`、`*.pyc`、`scripts/.ipynb_checkpoints/` | 运行时自动生成 |
| `web/app.py.backup_*`（多个备份） | 本地回滚备份，无需上传 |
| `web/_sse_out.txt` `web/_run.log` `web/_minimal_test.py`（如残留） | 调试临时文件 |
| `inference/fixtures/corrupt_test.jpg` | 测试损坏图夹具 |
| `experiments/` 下除 best_model/model_final 外的全部内容（5.7 GB checkpoints、日志、VDL、archive） | 训练/实验产物，Web 不需要 |
| 各实验 `logs/vdlrecords.*.log`、`train.log`、`run.out` | 训练日志 |

---

## 5. 禁止覆盖清单（E 类）

| 路径 | 说明 |
|---|---|
| `experiments/final_evaluation/` | **TEST 独立评估记录**（含 TEST=0.417 等），不迁移、不覆盖 |
| `experiments/direction_m/M_SCALEUP_100e/checkpoints/` 除 best_model/model_final 外全部 | 中间 checkpoint，不覆盖 AutoDL 训练产物 |
| `dataset/processed_detection/`（原始数据） | 原始数据集，Web 使用 exiffix 副本，不覆盖 |
| `test.json` / `images/test/` | **严禁上传/访问 TEST** |
| AutoDL 上已有的同名实验/训练输出 | 迁移时选择"跳过已存在"避免覆盖 |

---

## 6. 最小可运行 Web 系统上传包（可启动完整 Web + GPU 推理）

```
AIC2026_AgriVision/
├── web/
│   ├── __init__.py  app.py  backend.py  config.py
│   └── requirements_web.txt  README.md
├── inference/
│   ├── __init__.py  config.py  detector.py  batch_infer.py
│   ├── postprocess.py  visualize.py  infer.py
│   └── requirements_infer.txt
├── configs/
│   ├── datasets/agrivision_detection.yml
│   └── ppyoloe_plus_crn_m_100e_agrivision.yml
├── experiments/direction_m/M_SCALEUP_100e/checkpoints/
│   └── best_model.pdparams                      # 94.3 MB
├── dataset/processed_detection_exiffix/annotations/
│   └── val.json                                 # 13 类映射
└── PaddleDetection/
    ├── ppdet/                                   # 21 MB（整个包）
    ├── configs/runtime.yml
    ├── configs/ppyoloe/_base_/
    │   ├── optimizer_80e.yml  ppyoloe_plus_crn.yml  ppyoloe_plus_reader.yml
    └── ppdet/utils/simfang.ttf
```
> 说明：不含 demo_data/demo_outputs → 农业病害场景/比赛演示模式页签显示"暂无演示数据"（优雅降级，不影响核心检测）。

## 7. 完整比赛版 Web 系统上传包（保持 15 张演示/农业场景等全部功能）

在最小包基础上追加：

```
AIC2026_AgriVision/
├── web/
│   ├── demo_data/                     # 15 张原图 + demo_manifest.json/csv（2 MB）
│   ├── demo_outputs/run_20260817_173307/
│   │   ├── visualized/                # 15 张检测结果图（1.8 MB）
│   │   ├── detections/                # 每案例 JSON（75 KB）
│   │   └── predictions.json  class_statistics.csv  inference_summary.json  inference.log
│   └── demo_cache/                    # 可选（3.8 MB，可再生）
├── dataset/processed_detection_exiffix/    # 全量 414 MB（未来演示重生成/CLI）
├── experiments/direction_m/M_SCALEUP_100e/checkpoints/
│   └── model_final.pdparams           # 可选（94.3 MB）
├── web/make_demo_data.py  web/make_demo_report.py  web/_direct_test.py
├── scripts/  tools/yolo_to_coco.py
└── *.md（README_AUTODL / PROJECT_FINAL_STATUS / WEB_DEMO_FINAL_REPORT /
    DEMO_CASES_REPORT / LOCAL_DEPLOYMENT_AUDIT / INFERENCE_VALIDATION_REPORT /
    WEB_APP_VALIDATION_REPORT）
```

---

## 8. FileZilla 上传建议

1. 用 SFTP 连接 AutoDL；目标根目录 `/root/autodl-tmp/AIC2026_AgriVision/`。
2. 按"目录对照"逐项上传：先传 `web/`、`inference/`、`configs/`、`PaddleDetection/`（或用 C 类复用）、`experiments/direction_m/M_SCALEUP_100e/checkpoints/best_model.pdparams`、`dataset/…/annotations/val.json`。
3. 大数据：`best_model.pdparams`（94 MB）、`dataset/…exiffix`（可选 414 MB）。**绝不上传** experiments 下 5.7 GB checkpoints 与 TEST 相关。
4. 同名文件选择**跳过/覆盖策略**：`web/`、`inference/`、`configs/` 覆盖（以本地为准）；`experiments/`、`dataset/processed_detection/`、`experiments/final_evaluation/` **跳过已存在（不覆盖）**。
5. 上传后保持目录层级与本地一致（`configs/` 与 `PaddleDetection/` 的相对位置不变，`_BASE_` 链才能解析）。

---

## 9. 上传后 Claude Code 第一轮检查命令

```bash
cd /root/autodl-tmp/AIC2026_AgriVision
# 1) 语法/依赖自检
python -m py_compile web/app.py web/backend.py inference/*.py && echo COMPILE_OK
python -c "import gradio, matplotlib, pandas, cv2, pycocotools; print('deps ok')"
# 2) 关键文件存在性
ls -la experiments/direction_m/M_SCALEUP_100e/checkpoints/best_model.pdparams
ls -la dataset/processed_detection_exiffix/annotations/val.json
ls -la configs/ppyoloe_plus_crn_m_100e_agrivision.yml
ls -la PaddleDetection/ppdet/utils/simfang.ttf
# 3) 启动自检（模型加载 + 13 类映射 + 后端）
python -c "import sys; sys.path.insert(0,'.'); from web import backend; print(backend.self_check())"
# 4) 演示资源自检（若有 demo_data/demo_outputs）
python -c "import sys; sys.path.insert(0,'.'); from web import app as a; a._demo_asset_check()"
# 5) GPU 检测一次（需已切换 device，见 §11）
python -c "import sys; sys.path.insert(0,'.'); from web import app as a; a._detect(['dataset/processed_detection_exiffix/images/val/TRAIN_000004_apple_scab.jpg'], 0.5)"
```

---

## 10. 启动 Web 所需命令

```bash
cd /root/autodl-tmp/AIC2026_AgriVision
nohup python web/app.py > web/_run.log 2>&1 &
# 访问 http://<AutoDL内网IP>:7860 （AutoDL 可做端口转发 7860）
```
> 服务器默认绑定 `127.0.0.1:7860`；如需外部访问可改为 `server_name="0.0.0.0"`（AutoDL 控制台做 7860 端口映射）。

---

## 11. GPU 推理所需环境依赖（AutoDL）

| 依赖 | 版本建议 | 说明 |
|---|---|---|
| Python | 3.8–3.12（训练用 3.12.3） | — |
| **paddlepaddle-gpu** | **==2.6.2（cu118）** | 与训练/TEST 一致；`pip install paddlepaddle-gpu==2.6.2 -f https://www.paddlepaddle.org.cn/whl/linux/linux-gpu-cu118.html` |
| gradio | >=6.0（本地 6.24.0） | Web 界面 |
| **matplotlib** | 任意（≥3.3） | **requirements 中缺失，必须安装**（柱状图） |
| pandas / numpy | pandas>=1.5 / numpy>=1.21 | 统计 |
| opencv-python / Pillow | >=4.5 / >=9.0 | 图像 |
| pycocotools / tqdm | >=2.0 / >=4.60 | COCO 指标 / 进度 |
| **中文字体** | fonts-noto-cjk（`apt install fonts-noto-cjk`） | **matplotlib 柱状图中文**；可视化叠加层用 `PaddleDetection/ppdet/utils/simfang.ttf` 即可 |

**GPU 切换（当前代码默认 CPU，AutoDL 上需 1 行修改）**：
`web/backend.py` 第 47 行 `device="cpu"` → `device="gpu"`（后端 `paddle_gpu` 已注册，检测器支持 GPU）。改后 AutoDL 4090 上 FPS 约 16（与 TEST 评估一致）。也可改为读取环境变量便于切换。

---

## 12. 预计需要在 Linux 下重新生成/注意的文件

| 文件/目录 | 说明 |
|---|---|
| `web/demo_cache/` | 首次访问时由 `resolve_demo_asset_path` 自动重建（上传则免生成） |
| `web/outputs/`、`web/tmp/`、`inference/outputs/` | 运行时自动生成（每个检测一个时间戳 run） |
| `__pycache__/`、matplotlib 字体缓存 | 自动生成 |
| **matplotlib 中文字体** | Linux 无 Microsoft YaHei；需 `fonts-noto-cjk` 或把 `simfang.ttf` 注册进 matplotlib fontManager（当前 `_setup_matplotlib_font` 会自动尝试 Noto CJK；若未安装需 `apt install fonts-noto-cjk`） |
| `PaddleDetection/dataset/processed_detection` 软链 | **Web 推理不需要**（检测器用 `-o dataset_dir=绝对路径` 覆盖）；仅训练脚本需要 |
| `NO_PROXY=127.0.0.1` 环境设置 | app.py 已内置，Linux 无害 |
| `weights/`、`outputs/`（根目录空目录） | 无实际内容，可不上传 |

---

## 13. 关键说明与风险提示

1. **13 类映射唯一来源** = `dataset/processed_detection_exiffix/annotations/val.json`（非 TEST）。
2. **demo 演示不依赖 TEST**：demo_data/demo_outputs 全部来自 VAL；TEST 文件严禁上传。
3. **GPU 需改一行代码**（`web/backend.py` device="cpu"→"gpu"）；其余代码无需改动即可在 Linux 运行（BASE_DIR 相对路径、无软链依赖、无 Windows 硬编码路径）。
4. **Paddle 版本**：AutoDL 用 2.6.2（官方验证环境）；本地 3.3.0 仅用于本地验证，不迁移。
5. **最小包不含 demo_data/demo_outputs** 时，农业病害场景/比赛演示模式页签优雅显示"暂无演示数据"；完整包需带上这两目录才能保留 15 张演示与农业场景全部效果。

*审计完成：仅生成本清单，未打包、上传、修改或删除任何文件。*
