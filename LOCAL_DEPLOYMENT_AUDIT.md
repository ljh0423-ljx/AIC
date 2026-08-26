# LOCAL_DEPLOYMENT_AUDIT — 最终模型本地部署可行性审计

- **审计日期**: 2026-08-17
- **审计性质**: 只读审计 + 单张 VAL 图 CPU 推理验证（未训练、未调参、未访问 TEST、未修改 checkpoint/数据/实验结果、未做模型转换与硬件部署）
- **审计环境**: 本地 Windows 11（无 NVIDIA GPU）
- **最终模型（锁定）**: PP-YOLOE+-m `experiments/direction_m/M_SCALEUP_100e/checkpoints/best_model.pdparams`（VAL mAP@0.5:0.95=0.428 / TEST=0.417）

---

## 0. 一句话结论

**本地已具备"可运行的 CPU 单图推理"能力**：通过 `-o` 路径覆盖方案完全绕开 `PaddleDetection/dataset/processed_detection` 软链缺失问题，`tools/infer.py` 在本地 Paddle 3.3.0 上成功完成「图片 → PP-YOLOE+-m → 检测结果 → 可视化输出」完整链路（单图 ~2.3 s，纯前向 ~2.23 s，≈0.45 FPS）。本地无 GPU、版本与正式环境不一致、边缘端需走 RKNN/量化转换 —— 均为"可解决"而非"不可行"的障碍。**本次按指令未升级/降级 Paddle、未做模型转换、未做硬件部署。**

---

## 1. 最终产物路径确认（全部实测存在）

| 产物 | 路径 | 状态 |
|---|---|---|
| 最终 checkpoint | `experiments/direction_m/M_SCALEUP_100e/checkpoints/best_model.pdparams`（94.3 MB，md5 `18bd0e99329be2113f57280188c227b6`） | ✅ |
| 最终配置文件 | `configs/ppyoloe_plus_crn_m_100e_agrivision.yml`（md5 `c107b8797d866ed7b850f73708440af9`） | ✅ |
| 13 类 label 映射 | `dataset/processed_detection_exiffix/label_list.txt`（id 0–12）；COCO JSON 中 category_id 1–13 | ✅ |
| 推理脚本 | `PaddleDetection/tools/infer.py`（`tools/eval.py` 亦可） | ✅ |
| 数据集（EXIF-fix） | `dataset/processed_detection_exiffix/`（train 915 / val 113 / test 114） | ✅ |
| 数据软链 | `PaddleDetection/dataset/processed_detection` | ❌ 本地缺失（AutoDL 才创建）→ 已用 §5 方案解决 |

**13 类映射（本审计实测 val.json categories）**：1 Tomato Early blight leaf / 2 Tomato Septoria leaf spot / 3 Tomato leaf / 4 Tomato leaf bacterial spot / 5 Tomato leaf late blight / 6 Tomato leaf mosaic virus / 7 Tomato leaf yellow virus / 8 Tomato mold leaf / **9 Apple Scab Leaf** / 10 Apple leaf / 11 Apple rust leaf / 12 grape leaf / 13 grape leaf black rot（与 `PROJECT_FINAL_STATUS.md` §4 一致）。

---

## 2. 本地环境快照

| 项 | 值 |
|---|---|
| OS | Windows 11 Pro（10.0.26200） |
| Python | 3.12.4（Anaconda, AMD64） |
| PaddlePaddle | **3.3.0**（`compiled_with_cuda: True` 但 `cuda_device_count: 0` → 纯 CPU） |
| numpy / cv2 / pycocotools | 1.26.4 / 4.8.1 / ok |
| matplotlib / PIL / tqdm / yaml / visualdl / shapely / numba | 3.8.4 / 10.4.0 / 4.66.4 / 6.0.1 / 2.5.3 / 2.1.2 / 0.59.1 |
| PaddleDetection | v2.9.0（git HEAD `b25522a0`，本地完整源码） |
| GPU | 无（nvidia-smi 不存在） |

**与正式训练/评估环境对照**：AutoDL Linux / Python 3.12.3 / **Paddle 2.6.2 (cu118)** / RTX 4090 24GB。

---

## 3. 本地 Paddle 3.3.0 vs 正式环境 2.6.2 —— 兼容性风险评估

**按指令：本轮不升级、不降级。** 以下为实测证据与风险定性。

### 3.1 已实测「可用」的证据（本地 3.3.0）
1. `ppdet` 完整导入，`load_config` 正常（需 `PYTHONUTF8=1`，见 §5.2）。
2. 模型结构构建 + `best_model.pdparams` 加载：`set_dict` **0 缺失 / 0 意外键**。
3. `tools/infer.py` 完整推理链（reader 预处理 → 前向 → 后处理 → 可视化 → bbox 输出）**一次跑通**。
4. `ppdet.utils.check.check_version`（要求 ≥2.2）：3.3.0 通过（major 3 > 2 直接放行）。

### 3.2 风险清单
| 风险 | 说明 | 缓解 |
|---|---|---|
| 版本未经正式验证 | 官方 FPS/指标均来自 2.6.2（GPU）；3.3.0 仅验证了推理路径 | 正式复现/评测仍在 AutoDL 2.6.2 进行 |
| 算子级行为差异 | 3.x 对部分算子默认行为/数值精度有调整，本次单图前向不能覆盖全部算子 | 以 2.6.2 结果为基准；后续批量比对再做 |
| 训练侧 API 不兼容 | ppdet 2.9 的训练接口在 3.x 下大概率异常 | 本地不训练（本就不允许） |
| Windows 特有 | ① yaml 读取用 GBK 默认编码导致 UnicodeDecodeError → 需 `PYTHONUTF8=1`；② `runtime.yml` 默认 `use_gpu: true` → 无 GPU 需 `-o use_gpu=False`；③ 可视化字体 `~` 未展开（详见 §6.3） | 均为运行参数，不涉及代码改动 |

### 3.3 结论
本地 Paddle 3.3.0 **足以支撑最终模型的 CPU 推理验证**（已实证），但**不应**作为正式评测环境；正式环境锁定 AutoDL + 2.6.2（与 TEST 完全一致）。本轮不改变版本。

---

## 4. 推理脚本与参数约定（实测可用）

`tools/infer.py` 的 `ArgsParser` **只有 `-c` / `-o`**（无 `-w`），权重通过 **`-o weights=<绝对路径>`** 传入（与最终 TEST 评估 `tools/eval.py -o weights=…` 的官方用法一致）。`merge_config(FLAGS.opt)` 支持点分覆盖任意配置字段。

---

## 5. 本地数据集路径缺失 —— 解决方案（已实测通过）

### 5.1 问题
`configs/datasets/agrivision_detection.yml` 中 `dataset_dir: dataset/processed_detection` 相对 `PaddleDetection/` 根解析；本地该路径**不存在**（AutoDL 上由训练脚本创建软链）。若直接用默认配置，`ImageFolder.get_anno()` 找不到标注 → `get_categories('coco', None)` 会退回 **COCO17 默认 80 类**，标签全错。

### 5.2 方案（选定：`-o` 路径覆盖，零修改原始数据、零修改 PaddleDetection 源码）
不创建软链/junction，不改任何数据文件，仅通过 `-o` 把 `TestDataset` 指向 EXIF-fix 副本的**绝对路径**：

```bash
cd D:/Fruit/AIC2026_AgriVision/PaddleDetection
set PYTHONUTF8=1
python tools/infer.py \
  -c ../configs/ppyoloe_plus_crn_m_100e_agrivision.yml \
  -o weights=D:/Fruit/AIC2026_AgriVision/experiments/direction_m/M_SCALEUP_100e/checkpoints/best_model.pdparams \
     TestDataset.dataset_dir=D:/Fruit/AIC2026_AgriVision/dataset/processed_detection_exiffix \
     TestDataset.anno_path=annotations/val.json \
     use_gpu=False \
  --infer_img=<待推理图片绝对路径> \
  --output_dir=D:/Fruit/AIC2026_AgriVision/outputs/local_cpu_infer \
  --save_results=True
```

**要点**：
- `TestDataset.anno_path=annotations/val.json`：类别映射取自 **VAL**（非 TEST，符合"不访问 TEST"约束），与 test.json 的 13 类完全一致。
- `use_gpu=False`：覆盖 `runtime.yml` 默认 `use_gpu: true`。
- 本方案同时可推广到 `eval.py`（把 `EvalDataset.dataset_dir` 等同样 `-o` 覆盖）。
- 备选方案（本次**未采用**）：`PaddleDetection/dataset` 下建目录 junction 指向 exiffix —— 效果相同但会改动 PaddleDetection 目录结构，故弃用。

---

## 6. CPU 单图推理验证（VAL 图，未触碰 TEST）

### 6.1 验证对象
- 输入图：`dataset/processed_detection_exiffix/images/val/TRAIN_000004_apple_scab.jpg`（300×360，普通多目标叶片图，非 EXIF 旋转图、非错误案例）
- GT（val.json）：**6 个 Apple Scab Leaf**（id=19）

### 6.2 完整链路（逐环节确认）
`tools/infer.py` → 权重加载（best_model）→ `TestDataset`（val.json 类别）→ Decode → Resize 640×640 → NormalizeImage(mean/std 0/1) → Permute → **PP-YOLOE+-m 前向** → 后处理（score_threshold 0.5）→ 可视化 → 结果落盘。**全程无报错，PASS。**

### 6.3 推理耗时（CPU，640×640 单图）
| 环节 | 耗时 |
|---|---|
| 纯模型前向（实测） | **2.230 s** |
| 预处理（decode/resize/norm） | 0.051 s |
| 官方 infer.py 端到端（含后处理/可视化，日志 2.30~2.53 s/it） | ~2.3–2.5 s |
| 折算吞吐 | **≈0.42–0.45 FPS**（单线程 CPU） |

### 6.4 检测结果（`outputs/local_cpu_infer/bbox.json`）
| 项 | 值 |
|---|---|
| 类别 | **Apple Scab Leaf**（category_id 9，与 GT 类别一致 ✅） |
| 置信度 | **0.5698** |
| bbox (x,y,w,h) | **[166.68, 105.84, 104.96, 87.61]** |
| 检出数 | 1（GT 6 → 在 conf≥0.5 工作点检出 1/6；与该模型 Apple Scab 弱类 recall 已知偏弱的表现一致，非异常） |

### 6.5 输出物
| 文件 | 说明 |
|---|---|
| `outputs/local_cpu_infer/TRAIN_000004_apple_scab.jpg` | 可视化结果（实测含 2193 个红色框线像素 + 类别/置信度文字，确认已画框） |
| `outputs/local_cpu_infer/bbox.json` | 官方格式 bbox 预测 |

### 6.6 副作用记录（已清理/已说明）
- 首次运行自动下载可视化字体 `simfang.ttf`（仓库已自带 `ppdet/utils/simfang.ttf`，但 visualizer 走 `~/.cache/paddle/` 路径）；Windows 下 `~` 未展开，在 `PaddleDetection/~/.cache/paddle/` 生成了 10 MB 缓存 —— **本次审计已将其移除**，下次运行若该缓存不存在会自动重新下载（一次性）。
- 控制台首行的 GBK 乱码信息来自 `import paddle` 的原生库输出，无害。

---

## 7. 边缘设备部署可行性分析（仅分析，未执行转换/部署）

### 7.1 目标设备与现状
| 设备 | 算力形态 | 对 PP-YOLOE+-m（23.57M, 94.3MB fp32, 640×640）的可行性 |
|---|---|---|
| **RK3588 / NPU** | 3 核 NPU，INT8 ~6 TOPS | **最有前景**：RKNN 官方模型库含 PP-YOLOE 示例，INT8 量化后 `ppyoloe_s` @640 ≈ **32.9 FPS**（NPU）；m 参数量约为 s 的 3.06×、FLOPs 约 4.5×，按比例粗估 **INT8 下 ~7–11 FPS**（需实测，仍属实时边缘推理区间） |
| RK3588 / CPU | 4×A76 + 4×A55 | 仅作后备：预计 1–2 s/帧量级，无法实时 |
| **树莓派 4 / 5** | CPU-only（无 NPU） | 仅离线/演示：Pi5（4×A76）预计 ≥1 s/帧，Pi4 更慢；可考虑降分辨率或轻量模型 |

### 7.2 RK3588 部署路径（未来步骤，本次未执行）
```
.pdparams (state dict) ──export_model──▶ 推理模型(.pdmodel/.pdiparams)
      ──Paddle2ONNX──▶ .onnx ──RKNN-Toolkit2 (INT8 量化 + 校准)──▶ .rknn
      ──部署：RKNN Runtime (Python/C) 或 FastDeploy rknpu2 工具
```
关键点（据 2026 年公开资料）：
- RKNN Model Zoo 提供官方 `ppyoloe` 示例（`examples/ppyoloe`，含 `ppyoloe_m.onnx` 的转换/部署脚本），量化默认 INT8，FP16 免校准但达不到 NPU 峰值。
- INT8 需**校准数据集**（100–500 张）；可用本项目 **VAL 集**（113 张，非 TEST）做校准素材。
- 后处理（decode + NMS）官方在板端 CPU 实现，需移植还原。
- Paddle Lite 官方已支持瑞芯微 TIM-VX NPU（ARM Linux），提供 C++/Python API；也可走 FastDeploy 直接转 RKNN。
- 转换前需先 `export_model.py` 把 state dict 转为推理模型，并核对归一化与输入布局（NHWC/RGB 顺序）与部署配置一致。

### 7.3 风险与前置条件
| 风险/前提 | 说明 |
|---|---|
| 算子兼容 | PP-YOLOE+ 的 CSPRepResNet / SiLU / ESE 注意力算子主流 RKNN 支持，但需转换验证；PP-YOLOE（非 +）已有官方示例，`+` 版本需实测 |
| 量化精度 | INT8 对 AP 可能有小幅损失（该模型"量化友好"），需量化后用 VAL 复测 | 
| 模型是 state dict | 必须先 export，未锁定这一步骤本身 |
| **约束** | 本次明确**不做模型转换、不做硬件部署**；上述仅为可行性结论 |

**建议**：若后续启动边缘部署，优先 RK3588 NPU + INT8（RKNN 路线），目标 7–11 FPS @640；树莓派仅作离线演示。

---

## 8. 审计结论与后续建议

1. **本地推理**：✅ 可行（CPU，~0.45 FPS），`-o` 路径覆盖方案已验证且可复用；正式环境仍以 AutoDL 2.6.2 为准。
2. **版本**：暂不升级/降级 Paddle；保持 3.3.0 仅用于本地验证，正式评测用 2.6.2。
3. **边缘**：RK3588/NPU INT8 是唯一现实路径（预估 7–11 FPS @640）；树莓派仅演示。转换与部署列为后续独立任务。
4. **本次未做**：训练 / 调参 / TEST 访问 / checkpoint、数据、实验结果修改 / 模型转换 / 硬件部署 —— 全部遵守。

---

## 9. 本次审计产出物
| 产出 | 路径 |
|---|---|
| 本报告 | `LOCAL_DEPLOYMENT_AUDIT.md` |
| 推理可视化 | `outputs/local_cpu_infer/TRAIN_000004_apple_scab.jpg` |
| 推理 bbox 结果 | `outputs/local_cpu_infer/bbox.json` |

参考来源：RKNN Model Zoo（[ppyoloe 示例](https://github.com/edgeble/rknn_model_zoo/blob/main/examples/ppyoloe/README.md)、[量化说明](https://deepwiki.com/airockchip/rknn_model_zoo/7.2-quantization)）、[Paddle-Lite 官方仓库（瑞芯微 NPU 支持）](https://github.com/PaddlePaddle/Paddle-Lite)、[RKNN-Toolkit2 部署实践](https://blog.gitcode.com/14aa8fb8176e3dba0af078de7743b78e.html)。
