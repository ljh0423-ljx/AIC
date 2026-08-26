# DEMO_CASES_REPORT — 农业病害智能检测系统演示案例

- 生成日期：2026-08-17 17:35
- 演示数据来源：`dataset/processed_detection_exiffix/images/val/`（**仅 VAL，未使用 TEST**）
- 图片处理：仅复制到 `web/demo_data/`，**未修改原图**
- 推理程序：`inference/infer.py`（正式推理程序，PP-YOLOE+-m best_model）
- 结果目录：`run_20260817_173307`
- 性能：15 张 / 成功 15 / 失败 0 / 总耗时 25.46 s / 平均 1.697 s/张 / **实际 FPS 0.59**

> 以下置信度均为模型真实输出（conf=0.5 阈值过滤），未人为提高。

## 案例清单（15 例）

| demo_id | 作物 | 场景 | 原图文件 | GT类别(数) | 检测结果(置信度) | 单图耗时(ms) |
|---|---|---|---|---|---|---|
| tomato_01_early_blight | 番茄 | 番茄病害检测(single) | `TRAIN_000029_early-blight-of-tomato-tomato-1.jpg` | Tomato Early blight leaf(1) | Tomato Septoria leaf spot 0.61; Tomato leaf bacterial spot 0.52 | 1630 |
| tomato_02_septoria | 番茄 | 番茄病害检测(multi) | `TRAIN_000037_Tomato+Problems+Septoria+Leaf+Spot.jpg` | Tomato Septoria leaf spot(3) | Tomato Septoria leaf spot 0.67; Tomato Septoria leaf spot 0.64 | 1661 |
| tomato_03_septoria | 番茄 | 番茄病害检测(single) | `TRAIN_000040_Septoria_leaf_spot_tomato.jpg` | Tomato Septoria leaf spot(1) | Tomato Septoria leaf spot 0.53 | 1637 |
| tomato_04_mosaic_dense | 番茄 | 番茄病害检测（密集）(dense) | `TEST_000079_9511.img.jpg` | Tomato leaf mosaic virus(12) | Tomato leaf 0.57; Tomato leaf 0.54; Tomato leaf 0.51 | 1763 |
| tomato_05_yellow_virus | 番茄 | 番茄病害检测(multi) | `TEST_000088_tylcv-seminar-1-638.jpg` | Tomato leaf yellow virus(5) | Tomato leaf yellow virus 0.55; Tomato leaf yellow virus 0.55; Tomato leaf yellow virus 0.53; Tomato leaf yellow virus 0.51 | 1723 |
| tomato_06_mold | 番茄 | 番茄病害检测(multi) | `TEST_000099_fungus-univ-of-minnesoeta.jpg` | Tomato mold leaf(4) | Tomato leaf late blight 0.59; Tomato mold leaf 0.56 | 1725 |
| tomato_07_bacterial_spot | 番茄 | 番茄病害检测(single) | `TRAIN_000057_bacterial-spot-of-tomato-7-638.jpg` | Tomato leaf bacterial spot(1) | Tomato Septoria leaf spot 0.60 | 1662 |
| tomato_08_late_blight | 番茄 | 番茄病害检测(multi) | `TRAIN_000069_IMG_5360.jpg` | Tomato leaf late blight(2) | Tomato leaf late blight 0.90; Tomato leaf late blight 0.89 | 1640 |
| apple_01_scab | 苹果 | 苹果叶部病害检测(multi) | `TRAIN_000004_apple_scab.jpg` | Apple Scab Leaf(6) | Apple Scab Leaf 0.57 | 1652 |
| apple_02_rust | 苹果 | 苹果叶部病害检测(single) | `TEST_000018_0605_Rust-induced_leafspot.jpg` | Apple rust leaf(1) | Apple rust leaf 0.84 | 1751 |
| apple_03_rust | 苹果 | 苹果叶部病害检测(single) | `TRAIN_000020_PLPATH-FRU-02-cedar-apple-rust-figure-1.jpg` | Apple rust leaf(1) | Apple rust leaf 0.82; grape leaf black rot 0.52 | 1652 |
| apple_04_healthy_leaf | 苹果 | 苹果叶部病害检测(single) | `TRAIN_000014_stock-photo-green-apple-leaf-clipping-path-258728936.jpg` | Apple leaf(1) | Apple leaf 0.94 | 1657 |
| grape_01_healthy_leaf | 葡萄 | 葡萄叶部病害检测(single) | `TEST_000106_depositphotos_3443387-stock-photo-the-green-grape-leaf-on.jpg` | grape leaf(1) | grape leaf 0.93 | 1671 |
| grape_02_black_rot | 葡萄 | 葡萄叶部病害检测(single) | `TEST_000108_5-29black-rot-chardRR.jpg` | grape leaf black rot(1) | grape leaf black rot 0.94 | 1704 |
| grape_03_black_rot | 葡萄 | 葡萄叶部病害检测(single) | `TEST_000112_Black%20rot%20on%20foliage.jpg` | grape leaf black rot(1) | grape leaf black rot 0.90 | 1651 |

## 场景类型覆盖
- 单目标（single）：9 例
- 多目标（multi，2~9 个）：5 例
- 密集场景（dense，≥10 个 GT）：1 例（`tomato_04_mosaic_dense`，12 个 GT）
- 作物覆盖：番茄 8 / 苹果 4 / 葡萄 3
- 类别覆盖：12/13 类（Tomato 7 类、Apple 3 类、Grape 2 类）

## 说明
- 演示图片均来自 VAL 子集；**未复制/读取任何 TEST 图片**。
- 检测结果全部来自正式推理程序，未做任何后处理/人为调高置信度。
- 少量案例存在类间混淆（如 Septoria↔bacterial spot），为模型真实能力表现，如实展示。