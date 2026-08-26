# INFERENCE_VALIDATION_REPORT — 农业病害检测系统本地推理验证

- **验证日期**: 2026-08-17
- **性质**: 推理程序开发 + 推理验证（**未训练、未调参、未修改 checkpoint、未修改数据集、未访问 TEST、未做模型转换与 RK3588 部署**）
- **结论**: ✅ 全部验证通过。最终模型 PP-YOLOE+-m 在本机（Windows + Paddle 3.3.0 + CPU）完成「图片 → 模型 → 检测结果 → 可视化 → 统计产物」完整链路，单图/批量/文件夹/失败隔离均正常，性能统计全部基于真实运行时间。

---

## 1. 验证对象与关键路径

| 项 | 值 |
|---|---|
| 最终模型 | `experiments/direction_m/M_SCALEUP_100e/checkpoints/best_model.pdparams`（94.3 MB，md5 `18bd0e99329be2113f57280188c227b6`，**验证后未变**） |
| 模型配置 | `configs/ppyoloe_plus_crn_m_100e_agrivision.yml`（PP-YOLOE+-m，640×640） |
| 推理框架 | `PaddleDetection/`（v2.9.0，**源码零改动**，路径以 `-o` 绝对路径覆盖注入） |
| 13 类映射 | `dataset/processed_detection_exiffix/annotations/val.json`（**非 TEST**，不重新编号） |
| 数据集 | `dataset/processed_detection_exiffix/`（train 915 / val 113 / test 114；本次只读 VAL 图片） |
| 新模块 | `inference/`（config / detector / postprocess / visualize / batch_infer / infer + README + requirements + outputs） |

**环境**：Windows 11（10.0.26200）· Python 3.12.4 · **PaddlePaddle 3.3.0**（CPU，cuda_device_count=0）· cv2 4.8.1 · numpy 1.26.4 · pycocotools · Pillow。
正式训练/TEST 环境仍为 AutoDL Paddle 2.6.2（RTX 4090）；**本地未改 Paddle 版本**。

---

## 2. 运行方式（在工程根目录执行）

```bash
# 单图
python inference/infer.py --image <图片.jpg>
# 文件夹
python inference/infer.py --dir <目录> [--recursive]
# 指定置信度
python inference/infer.py --image <图片> --conf 0.5
# 多图批量
python inference/infer.py --image a.jpg --image b.jpg --conf 0.5
# 递归 + 限量
python inference/infer.py --dir <目录> --recursive --limit 20
```

支持 `jpg/jpeg/png/bmp/webp`；结果目录按时间戳自动创建 `inference/outputs/run_YYYYMMDD_HHMMSS/`，不覆盖历史。

---

## 3. 单图验证（TRAIN_000004_apple_scab.jpg）

| 项 | 结果 |
|---|---|
| 图片 | `images/val/TRAIN_000004_apple_scab.jpg`（300×360，GT = 6×Apple Scab Leaf，非 EXIF 图） |
| 检测 | **1 个目标：Apple Scab Leaf（class_id 9）**，confidence **0.5698**，bbox **[166.68, 105.84, 104.96, 87.61]** |
| 结果一致性 | 与本系统此前用官方 `tools/infer.py` 的验证输出**逐位一致**（0.569842 / 相同 bbox） |
| 推理耗时（单图） | **2304 ms**（纯模型前向 ~2230 ms + 预处理/后处理） |
| 输出 | 原图副本 / 可视化（红色 bbox + 标签 + 标题 + 目标数 + 真实耗时）/ JSON / TXT / CSV |

> 说明：conf≥0.5 检出 1/6，与该模型 Apple Scab 类在 conf=0.5 下 recall 偏弱的已知表现一致，属预期行为，非程序异常。

---

## 4. 批量验证（8 张 VAL 图）

| 项 | 结果 |
|---|---|
| 图片数 | 8（跨 Apple rust / Tomato early blight / mosaic / grape / apple scab / early blight / septoria 等） |
| 成功 / 失败 | **8 / 0** |
| 检测目标总数 | 11（逐图 1,0,3,1,1,2,2,1） |
| 总耗时 | **18.84 s** |
| 平均耗时 | **2.355 s/张** |
| **FPS（真实）** | **0.42**（= 8 / 18.84） |
| 模型加载 | 全程 **仅加载 1 次** |

检测合理性抽查：Apple rust GT 图 → Apple rust leaf 0.836 ✓；grape leaf GT 图 → grape leaf 0.926 ✓；apple scab GT 图 → Apple Scab Leaf ✓；Septoria 图 → Septoria ✓；mosaic 图在 conf≥0.5 检出 3 个 Tomato leaf（该弱类已知困难，无 mosaic 高置信框，符合模型表现）。

## 5. 失败隔离验证（含 1 张损坏图片）

| 项 | 结果 |
|---|---|
| 输入 | 2 张正常 VAL 图 + 1 张损坏图（`inference/fixtures/corrupt_test.jpg`，垃圾字节） |
| 成功 / 失败 | **2 / 1** |
| 失败记录 | `corrupt_test.jpg → 图片无法解码（可能已损坏或不支持的格式）`，已写入 `inference_summary.json` 的 failed 列表 |
| 行为 | 单张失败**未中断批次**，其余 2 张正常完成 |

## 6. 文件夹推理验证（VAL images 目录，小规模）

| 项 | 结果 |
|---|---|
| 命令 | `python inference/infer.py --dir dataset/processed_detection_exiffix/images/val --limit 4` |
| 发现 / 成功 / 失败 | 4 / 4 / 0 |
| 总耗时 / FPS | 10.85 s / **0.37**（真实） |
| 说明 | 仅读取 VAL 目录，**未访问 test 目录**；目录中发现-排序-限量逻辑正常 |

## 7. 异常处理核验（均给出明确中文提示，非裸 Traceback）

| 场景 | 提示 |
|---|---|
| 空文件夹 / 无图片 | 「未找到任何可推理的图片…」 |
| 图片损坏 | 「图片无法解码（可能已损坏或不支持的格式）：<名>」 |
| 模型文件缺失 | 「模型文件不存在：…」 |
| 配置文件缺失 | 「配置文件不存在：…」 |
| 类别标注缺失 | 「类别映射标注文件不存在：…」 |
| 未知后端 | 「未知推理后端：rknn…（预留接口，本阶段未实现）」 |
| 无检测目标 | 正常处理：JSON/TXT 记为 0 目标，保留原图与可视化 |

---

## 8. 输出产物（本次 4 个有效 run）

`inference/outputs/` 下按时间戳生成，均含 `originals/ · visualized/ · detections/{json,txt,csv} · predictions.json/csv · class_statistics.csv · inference_summary.json · inference.log`：

| run 目录 | 用途 | 规模 |
|---|---|---|
| `run_20260817_154849` | 单图验证 | 1 图 / 1 目标 / FPS 0.42 |
| `run_20260817_154953` | 批量验证 | 8 图 / 11 目标 / FPS 0.42 |
| `run_20260817_155112` | 失败隔离 | 3 图（2 成功 1 失败）/ FPS 0.59 |
| `run_20260817_155139` | 文件夹验证 | 4 图 / FPS 0.37 |

**每个检测目标的统一字段**：`image_name, class_id, class_name, confidence, x, y, width, height`（COCO 原图坐标）。
**批次统计字段**：每类 `num_images / num_targets / avg_confidence / max_confidence`；性能 `total_images / success_images / failed_images / total_time / average_time / fps`，**全部真实计时，无写死/虚构**。

---

## 9. 已知限制

1. **性能**：本地 CPU 单图约 2.3 s（640×640 前向 ~2.2 s），FPS ≈ 0.4，非实时；实时性需 GPU 或 RK3588。
2. **中文标注**：项目当前无官方中文类名映射，遵循"没有则不要擅自创造"，可视化默认仅显示英文；在 `config.py` 的 `CLASS_NAMES_CN` 填写后自动启用"中文+英文"。
3. **Paddle 3.3.0**：仅验证了本推理路径；训练/复现仍以 AutoDL 2.6.2 为准。
4. **EXIF 方向**：本数据集图片为 EXIF 烘焙后的像素，已正常；任意第三方带 EXIF 旋转的照片建议先转正。
5. **目录**：`--dir` 默认非递归，递归需 `--recursive`。
6. 控制台首行的 GBK 乱码为 `import paddle` 原生库输出，无害。

---

## 10. 后续 RK3588 迁移接口说明（本阶段未实施）

- **统一后端接口** `DetectorBackend`（`inference/detector.py`）：`load() / infer(bgr) / infer_batch(list) / close()`。
- 当前实现 `PaddleBackend`（CPU/GPU）。RK3588 只需新增 `RKNNBackend(DetectorBackend)`（Paddle→ONNX→RKNN-Toolkit2 INT8），并在 `create_backend()` 注册；`infer.py / batch_infer.py / postprocess.py / visualize.py` **无需改动**。
- 全部路径使用 `pathlib.Path` 动态拼接，无写死 Windows 路径；数据集路径以 `-o` 绝对路径注入，不依赖软链、不改 PaddleDetection 源码。
- 部署到 RK3588 前需：`export_model`（state dict → 推理模型）→ Paddle2ONNX → RKNN 量化（校准集可用 VAL，非 TEST）→ 板端集成。**以上均待后续独立任务执行。**

---

## 11. 完整性声明

- ✅ 模型未修改（best_model md5 验证前后一致 `18bd0e99…`）
- ✅ 数据集未修改（val.json 等 mtime 未变；仅读取 VAL 图片，未访问 test）
- ✅ PaddleDetection 源码零改动；未创建软链；本地 Paddle 版本未升级/降级
- ✅ 未训练、未调参、未重新计算 TEST 指标、未虚构 FPS、未宣称完成 RK3588 部署
- ✅ 新建文件均位于 `inference/` 与 `inference/outputs/`，未删除任何现有实验目录
