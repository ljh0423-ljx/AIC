# PlantDoc 目标检测数据集 — 最终预处理报告

> 生成时间: 2026-08-13  |  随机种子: 2026
> 原始数据未做任何修改/删除/覆盖; 所有产物在 `D:\Fruit\processed_detection\`

## 1. 原始数据量
- 原始 XML 总数: 2583
- 原始图片总数: 2586
- 原始类别数(全部): 29

## 2. 清洗后数据量
- 解析成功 XML: 2583
- 进入清洗流程: 2583
- 最终有效样本(13类内): 1144

## 3. 删除/排除数量及原因
| 排除原因 | 数量 |
| --- | --- |
| 损坏 XML (无法解析) | 0 |
| 空 XML (无 object) | 11 |
| 缺失图片 | 2 |
| XML size=0 | 4 |
| 图片无法解码 | 0 |
| 全部 bbox 非法 | 5 |
| MD5 重复副本 | 12 |
| 无 13 类内目标 | 1405 |
- 明细见 `quality_issues.csv` 与 `deduplication_log.csv`

## 4. 13 个类别详细统计

class_id → class_name → crop → image_count → object_count

| class_id | class_name | crop | image_count | object_count |
| --- | --- | --- | --- | --- |
| 0 | Tomato Early blight leaf | Tomato | 89 | 213 |
| 1 | Tomato Septoria leaf spot | Tomato | 149 | 430 |
| 2 | Tomato leaf | Tomato | 72 | 396 |
| 3 | Tomato leaf bacterial spot | Tomato | 113 | 278 |
| 4 | Tomato leaf late blight | Tomato | 110 | 219 |
| 5 | Tomato leaf mosaic virus | Tomato | 55 | 261 |
| 6 | Tomato leaf yellow virus | Tomato | 75 | 824 |
| 7 | Tomato mold leaf | Tomato | 90 | 291 |
| 8 | Apple Scab Leaf | Apple | 93 | 171 |
| 9 | Apple leaf | Apple | 91 | 247 |
| 10 | Apple rust leaf | Apple | 88 | 178 |
| 11 | grape leaf | Grape | 69 | 220 |
| 12 | grape leaf black rot | Grape | 64 | 133 |

## 5. 每类图片数 / 6. 每类目标数
(见上表 image_count / object_count 列)

## 7. train/val/test 数量
- train: 916  |  val: 114  |  test: 114

## 8. 各类别在 train/val/test 中的分布
| id | 类别 | train图 | val图 | test图 | 合计 |
| --- | --- | --- | --- | --- | --- |
| 0 | Tomato Early blight leaf | 69 | 8 | 8 | 85 |
| 1 | Tomato Septoria leaf spot | 119 | 15 | 15 | 149 |
| 2 | Tomato leaf | 51 | 6 | 6 | 63 |
| 3 | Tomato leaf bacterial spot | 91 | 11 | 11 | 113 |
| 4 | Tomato leaf late blight | 87 | 11 | 11 | 109 |
| 5 | Tomato leaf mosaic virus | 43 | 6 | 6 | 55 |
| 6 | Tomato leaf yellow virus | 59 | 8 | 8 | 75 |
| 7 | Tomato mold leaf | 72 | 9 | 9 | 90 |
| 8 | Apple Scab Leaf | 75 | 9 | 9 | 93 |
| 9 | Apple leaf | 73 | 9 | 9 | 91 |
| 10 | Apple rust leaf | 70 | 9 | 9 | 88 |
| 11 | grape leaf | 55 | 7 | 7 | 69 |
| 12 | grape leaf black rot | 52 | 6 | 6 | 64 |

## 9. 数据不均衡情况
- 目标数范围: 133 ~ 824
- 不均衡比 (max/min): 6.20
- 最多类: Tomato leaf yellow virus (824)
- 最少类: grape leaf black rot (133)
- ⚠️ Tomato leaf yellow virus 目标数显著偏高(单图多目标), 训练时建议类别加权或采样平衡

## 10. 重复数据处理
- 完全重复图片组: 12
- 排除重复副本: 12 (其中跨 train/test 泄露: 11)
- 同名异内容(保留): 5 组
- 明细见 `deduplication_log.csv`

## 11. bbox 合法性检查
- 非法 bbox(已排除该 box, 不静默修改): 11
- 尺寸不一致(XML size vs 实际, 已记录, 以实际尺寸归一化): 10
- 明细见 `quality_issues.csv`

## 12. 图片尺寸统计
- 宽度: min=115, max=6000, mean=1013
- 高度: min=69, max=6000, mean=861

## 13. 最终数据集路径
```
D:\Fruit\processed_detection/
├── images/train  images/val  images/test   # YOLO 图片
├── labels/train  labels/val  labels/test   # YOLO TXT 标签
├── visual_check/                              # 可视化抽检
├── label_list.txt   dataset.yaml
├── quality_issues.csv   deduplication_log.csv
└── DATASET_FINAL_REPORT.md
```

## 14. 是否可以直接用于 PP-YOLOE 训练

**可以直接用于 PP-YOLOE 训练。**

- 标签为标准 YOLO TXT (`class_id x_center y_center w h`, 归一化)
- `dataset.yaml` 已按 Ultralytics/YOLO 约定生成; PP-YOLOE (PaddleDetection) 可读 COCO,
  也可用同样数据通过 `tools/x2coco.py` 转 COCO 后挂载。
- train/val/test 已分层无泄露, 类别 id 与 label_list.txt 一致。
- 建议训练前:抽样核对 `visual_check/` 中 bbox 是否贴合; 关注 Tomato leaf yellow virus 的类别不平衡。

---
*报告由 preprocess_detection.py 自动生成, 原始数据未修改。*