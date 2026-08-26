# DATASET_CONSISTENCY_REPORT — AIC2026 PlantDoc 13类 数据一致性核对

- 日期: 2026-08-14
- 核对人: AutoDL 训练流程 (Claude), Baseline 深度诊断阶段
- 对象: `dataset/processed_detection` (原始, 只读) vs `dataset/processed_detection_clean` (训练用清洁副本)

## 结论摘要

1. **原始数据零修改**: `processed_detection` 116 项检查中所有文件保持原样 (含 0 字节损坏图)。
2. **916→915 的差异原因**: 原始 train 分片包含 1 张 0 字节损坏图 (`TRAIN_000133_...928225.jpg`), 清洁副本按用户批准的决策移除该图及其 5 个标注, 故 train 由 916→915。**不是统计口径变化, 而是数据清理所致。**
3. **最终训练实际使用**: **train 915 张 / 3079 目标, val 113 张 / 350 目标** (训练日志 `Load [915 samples valid, 0 samples invalid]` 与 `Load [113 samples valid, 0 samples invalid]` 证实)。
4. **test 完全封闭**: 114 张 / 423 目标, 图片清单 MD5 与 JSON MD5 均与原始一致, **零改动**。

---

## 一、数量统计对比 (原始 vs 清洁副本)

| Split | 图片数(原始) | 图片数(clean) | 目标数(原始) | 目标数(clean) | 差异 |
|---|---|---|---|---|---|
| train | 916 | **915** | 3084 | **3079** | 移除 1 图 / 5 标注 |
| val | 114 | **113** | 354 | **350** | 移除 1 图 / 4 标注 |
| test | 114 | 114 | 423 | 423 | 无 |
| **合计** | 1144 | **1142** | 3861 | **3852** | 移除 2 图 / 9 标注 |

- label_list.txt col5 (总实例数) 合计 = 3861 = 原始目标数 ✓ (col4=含该类图像数 1158, col5=实例 3861)
- YOLO txt 与图片一一对应: clean train 915/915, val 113/113, test 114/114 ✓

## 二、被移除的 0 字节损坏图 (共 2 张)

| 分片 | 文件 (原始路径, 0 字节) | COCO image_id | 原标注 | YOLO txt |
|---|---|---|---|---|
| train | `dataset/processed_detection/images/train/TRAIN_000133_apple-tree-branch-blossom-plant-fruit-berry-leaf-flower-food-green-produce-evergreen-flora-sad-shrub-apples-branch-with-apples-flowering-plant-rose-family-acerola-malpighia-woody-plant-land-plant-928225.jpg` | 208 | 5 个 (全为 cat=10 Apple leaf) | 0 字节 |
| val | `dataset/processed_detection/images/val/TRAIN_000104_nature-plant-grape-vine-wine-fruit-food-green-produce-agriculture-grapevine-vines-shrub-grapes-winegrowing-rebstock-flowering-plant-vitis-grape-leaves-green-grapes-swiss-francs-land-plant-grapevine-family-556232.jpg` | 109 | 4 个 (全为 cat=12 grape leaf) | 0 字节 |

- 两张图均 **0 字节** (PIL 无法打开, 系统内无其他备份), cv2.imread 返回 None, 训练遇之必崩。
- 移除后 train 的 cat=10 目标 200→195 (差 5), val 的 cat=12 目标 18→14 (差 4), 与上述标注数逐一对齐 ✓
- 原始目录中两张图与其 0 字节 txt **保留未动**; clean 副本中已不存在。

## 三、MD5 前后对照

### 3.1 图片清单 (basename 排序后 MD5)

| Split | 原始清单 MD5 | clean 清单 MD5 | 一致性 |
|---|---|---|---|
| train | `c971f320aa2e6d13988547b68a24cb75` | `88d9406b09be27ba3820fd02ebcb4581` | 差异恰为被移除的 1 个文件名 |
| val | `e71af4d052c15ba6aa1ba0b778d500dc` | `d87c0224edbf29f2aa280d23fdf119a2` | 差异恰为被移除的 1 个文件名 |
| test | `8e0848f8fd6723bbe635a4fd052622bf` | `8e0848f8fd6723bbe635a4fd052622bf` | **完全一致** |

> 注: 早期用全路径(含目录前缀)计算的 MD5 不同属于路径伪影 (`processed_detection` vs `processed_detection_clean` 前缀不同), 与文件内容无关。

### 3.2 文件内容 (inode 硬链接验证)

clean 副本全部通过 `cp -al` 硬链接复制, 逐 split 比对 inode:

| Split | 原始 inode 数 | clean inode 数 | 共享 inode | 内容一致性 |
|---|---|---|---|---|
| train | 916 | 915 | 915 | 100% 共享 |
| val | 114 | 113 | 113 | 100% 共享 |
| test | 114 | 114 | 114 | 100% 共享 |

→ 所有 retained 图片与原文件同 inode (逐字节相同), 副本删除不影响原件。

### 3.3 标注 JSON MD5

| Split | 原始 JSON MD5 | clean JSON MD5 | 说明 |
|---|---|---|---|
| train | `804416ef1fd1daf5ab0333e9dd35c258` | `df0ab46d4c9ea38bb532a51fc169bd5e` | 重写 (移除 image_id=208 及其标注) |
| val | `6898860d837da317d6c92e6457c77b1f` | `167f568d0736dd11c5123e06f66a6124` | 重写 (移除 image_id=109 及其标注) |
| test | `2008eddc2aa4fcd63ba720c33cc678da` | `2008eddc2aa4fcd63ba720c33cc678da` | **完全一致** |

## 四、最终训练实际用量确认

- **训练日志** (experiments/baseline_v1/train.log) 实测加载行:
  - `[08/14 18:49:02] ... Load [915 samples valid, 0 samples invalid] ... train.json` → train 用 915 张
  - `[08/14 18:57:42] ... Load [113 samples valid, 0 samples invalid] ... val.json` → val 用 113 张
- 训练全程 100 epoch, 无 OOM / 无 ERROR / 无 Traceback, 干净退出。
- 评估软链: `PaddleDetection/dataset/processed_detection → ../../dataset/processed_detection_clean` (train/eval 脚本统一指向 clean)。

## 五、与 data_prep_notes.txt 的一致性核对

| data_prep_notes 记录 | 本次实测 | 一致 |
|---|---|---|
| 原始 916/114/114=1144, 3861 目标 | 916/114/114, 3861 标注 | ✓ |
| clean 915/113/114=1142, 3852 目标 | 915/113/114, 3852 标注 | ✓ |
| 移除 2 张 0 字节图, train id=208 (5个cat10), val id=109 (4个cat12) | 实测一致 | ✓ |
| 原始 processed_detection 保持原样 | 原始目录含 0 字节图与空 txt 未动 | ✓ |
| 硬链接复制零额外空间 | inode 共享率 100% | ✓ |

## 六、其他一致性事实

- 无交集: train/val/test 图片文件名两两无重复 (此前阶段已核, 本报告基于各 split 独立清单)。
- 类别: 13 类, COCO category_id 1~13 与 label_list.txt 名称逐一对齐 (label_list col2 顺序即 0~12 索引)。
- bbox 合法性: 全部 x2>x1, y2>y1, 在图像尺寸内 (此前阶段已核; 本次仅核对数量, 未重复全量几何校验)。
- 已知数据质量项 (非本次改动): 11 张 (0.96%) 图像存在 EXIF 旋转 (标注尺寸与 cv2 实际像素宽高对调), 按用户决策保持原样仅记录, 详见 data_prep_notes.txt。

## 七、最终唯一可信数据统计 (Canonical — Baseline v1 唯一数据基准)

> 以下统计经"原始清单 / 清洁副本 / COCO JSON / 训练日志 / MD5"五路交叉核对, 作为 Baseline v1 及后续所有 Ablation 实验的**唯一可信数据基准**。训练/评估统一使用 `dataset/processed_detection_clean` (软链名 `dataset/processed_detection`)。

### 7.1 分片总览 (图片文件数 = JSON images = YOLO txt 数, 全部 1:1)

| Split | 图片数 | 目标数 | YOLO txt | 说明 |
|---|---|---|---|---|
| train | **915** | **3079** | 915 | 原始 916 → 移除 1 张损坏图 (5 标注) |
| val | **113** | **350** | 113 | 原始 114 → 移除 1 张损坏图 (4 标注) |
| test | **114** | **423** | 114 | 零改动 |
| **合计** | **1142** | **3852** | 1142 | 原始 1144/3861, 差 2 图 9 标注 |

### 7.2 每类目标数 (clean 副本, 与 label_list col5 校验)

| cat | 类别 | train | val | test | clean合计 | label_list col5 | 校验 |
|---|---|---|---|---|---|---|---|
| 1 | Tomato Early blight leaf | 171 | 21 | 21 | 213 | 213 | ✓ |
| 2 | Tomato Septoria leaf spot | 338 | 37 | 55 | 430 | 430 | ✓ |
| 3 | Tomato leaf | 294 | 36 | 66 | 396 | 396 | ✓ |
| 4 | Tomato leaf bacterial spot | 227 | 16 | 35 | 278 | 278 | ✓ |
| 5 | Tomato leaf late blight | 159 | 26 | 34 | 219 | 219 | ✓ |
| 6 | Tomato leaf mosaic virus | 208 | 33 | 20 | 261 | 261 | ✓ |
| 7 | Tomato leaf yellow virus | 700 | 57 | 67 | 824 | 824 | ✓ |
| 8 | Tomato mold leaf | 239 | 32 | 20 | 291 | 291 | ✓ |
| 9 | Apple Scab Leaf | 125 | 30 | 16 | 171 | 171 | ✓ |
| 10 | Apple leaf | 195 | 14 | 33 | 242 | 247 | **-5** (train 损坏图 5 标注) |
| 11 | Apple rust leaf | 149 | 12 | 17 | 178 | 178 | ✓ |
| 12 | grape leaf | 173 | 14 | 29 | 216 | 220 | **-4** (val 损坏图 4 标注) |
| 13 | grape leaf black rot | 101 | 22 | 10 | 133 | 133 | ✓ |

> 校验逻辑: clean 合计与 label_list col5 的差异恰为被移除的 9 个标注 (cat10 -5, cat12 -4), 其余 11 类完全相等 — 进一步证明除损坏图外无任何对象丢失。

### 7.3 被移除图片 (唯一两处差异)

| 原始分片 | 文件 | COCO id | 标注 | YOLO txt |
|---|---|---|---|---|
| train | `TRAIN_000133_...apple-tree...928225.jpg` | 208 | 5×cat10 | 0 字节 (空) |
| val | `TRAIN_000104_...grape-vine...556232.jpg` | 109 | 4×cat12 | 0 字节 (空) |

- **916→915 (train)**: 移除 0 字节损坏图 `TRAIN_000133_...` 及其 5 个标注。
- **114→113 (val)**: 移除 0 字节损坏图 `TRAIN_000104_...` 及其 4 个标注 (同样因 0 字节无法解码, 训练必崩)。
- 两张图在原始 `processed_detection` 中**保留未动**, 仅从训练用 clean 副本移除。

### 7.4 训练实际用量 (日志实证)

- train 加载: `Load [915 samples valid, 0 samples invalid]` ✓
- val 加载: `Load [113 samples valid, 0 samples invalid]` ✓
- test 全程未加载于训练/调参, 仅最终独立评估使用 (114 张 / 423 目标)。

### 7.5 数据使用铁律 (后续实验)

1. Baseline 及 Ablation 一律以 `processed_detection_clean` 为数据源; 禁止改原始 `processed_detection`。
2. 任何新实验副本 (如 EXIF 修复副本) 必须基于 clean 副本派生, 且**只允许在副本上修改**。
3. test 数据禁止用于任何训练/验证/调参/模型选择, 仅允许最终一次独立评估。

## 八、结论

- **无任何擅自修改**: 所有数量差异均可由"移除 2 张 0 字节损坏图 (9 个标注)"这一唯一且已获用户批准的操作解释。
- **test 集零污染**: 图片、标注、JSON 三重 MD5 完全一致。
- **训练用量明确**: 915 张训练图 / 3079 目标, 113 张 val / 350 目标, 与 Canonical 表一致, 无隐藏差异。
- 若未来获得两张损坏图的可恢复备份, 可回填原始目录还原 1144 张; 在获得前, clean 副本即为可复现的训练数据基准。
