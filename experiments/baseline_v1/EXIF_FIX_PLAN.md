# EXIF_FIX_PLAN — 方向1: EXIF 旋转数据质量修复方案 (仅设计, 未执行)

- 日期: 2026-08-14
- 状态: **设计完成, 未对任何数据执行修改**
- 范围: 只在新的实验副本 `processed_detection_exiffix` 上执行; 原始 `processed_detection` 与 `processed_detection_clean` 均不改动。

---

## 1. EXIF 异常图片清单 (11 张, 全量扫描实证)

扫描方法: 遍历 clean 副本全部 1142 张, 读取 EXIF Orientation 标签; 另以 cv2 像素尺寸 vs COCO JSON 尺寸比对。**11 张** Orientation∈{6,8} 且尺寸互换, 与 data_prep_notes 记录一致。另有 2 张 (TRAIN_000246 / TRAIN_000571) Orientation=0 但无尺寸互换, 判定为标签异常非旋转问题, **不纳入修复**。

| # | 分片 | 文件 | 原始像素 (w×h) | JSON 尺寸 (w×h) | Orientation | bbox 数 | 受影响类别 |
|---|---|---|---|---|---|---|---|
| 1 | train | TEST_000185_20130610_110514.jpg | 2448×3264 | 3264×2448 | 6 | 1 | Apple rust leaf |
| 2 | train | TEST_000266_Shoemaker_7068.JPG.jpg | 450×600 | 600×450 | 6 | 1 | Tomato Early blight |
| 3 | train | TRAIN_000172_20130802_111648.jpg | 2448×3264 | 3264×2448 | 6 | 1 | Apple rust leaf |
| 4 | train | TRAIN_000198_20130610_110525.jpg | 2448×3264 | 3264×2448 | 6 | 1 | Apple rust leaf |
| 5 | train | TRAIN_000225_early-blight-in-high-tunnel-tomatoes-19tm7*.jpg | 2091×2978 | 2978×2091 | 6 | 1 | Tomato Early blight |
| 6 | train | TRAIN_000325_tomato-septoria-3.jpg | 1152×1497 | 1497×1152 | 6 | 1 | Tomato Septoria leaf spot |
| 7 | train | TRAIN_000329_tomato-septoria-5.jpg | 1154×1500 | 1500×1154 | 8 | 1 | Tomato Septoria leaf spot |
| 8 | train | TRAIN_000437_flies.jpg | 2592×3888 | 3888×2592 | 6 | 1 | Tomato leaf |
| 9 | train | TRAIN_000494_IMG_2348.jpg | 2448×3264 | 3264×2448 | 6 | 3 | Tomato leaf bacterial spot |
| 10 | val | TRAIN_000054_happier-inside.jpg | 2592×3888 | 3888×2592 | 8 | 7 | Tomato leaf |
| 11 | test | TRAIN_000064_photo-1.jpg | 750×1000 | 1000×750 | 6 | 1 | Tomato leaf bacterial spot |

- 分布: train 9 张 / val 1 张 / test 1 张; 受影响 bbox 共 19 个; 类别集中于 Tomato(1/2/3/4) 与 Apple rust(11)。
- 完整扫描报告: `metrics/exif_anomaly_report.json`。

## 2. 关键设计发现 (决定修复方式)

对全部 11 张做几何验证 (见下方 §4 校验命令的预演):

> **bbox 与 JSON 尺寸均位于"显示 (旋转后) 空间", 与 EXIF 烘焙后的图像空间完全一致。**

- 例: TRAIN_000054_happier-inside.jpg 原始像素 2592×3888 (竖), JSON 3888×2592 (横), 烘焙(旋转)后 3888×2592 == JSON。
- 预演验证 **11/11 全部通过**: `cv2.rotate`(O=6→90°CW, O=8→90°CCW) 烘焙后尺寸与 JSON 完全一致, 且所有 bbox 落在烘焙后图像边界内。

**结论: 修复 = 仅将像素烘焙为显示方向; bbox 坐标与 JSON 宽高均无需变换。**

> 判定依据 (数据一致性): 全数据集 bbox 统一位于"JSON 尺寸空间"; 非 EXIF 图 JSON==像素, EXIF 图 JSON==显示尺寸。因此烘焙像素使其等于 JSON 尺寸即可对齐, 无需触碰标注。该规则优于"逐图猜 H1/H2", 且可被 §4 校验脚本以不变式自动证明。

## 3. 修复流程 (在 `processed_detection_exiffix` 副本上执行)

### 3.1 副本构建
1. `cp -al dataset/processed_detection_clean dataset/processed_detection_exiffix` (硬链接, 零额外空间; 后续替换仅覆盖目标文件, 不影响原 inode 指向的文件)。
2. 拷贝 annotations/ 与 label_list.txt、dataset.yaml 至副本 (保持不动)。

### 3.2 像素烘焙 (仅 11 张, 其中 train 9 + val 1; **test 不动**)
对每张 O=6 → `cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)`; O=8 → `cv2.ROTATE_90_COUNTERCLOCKWISE`。
- 编码: JPEG quality=95, 且 **写入时移除 EXIF Orientation 标签** (避免下游二次旋转)。
- 覆盖写入副本中的同路径文件 (硬链接 inode 仅被本文件替换, 原文件不受影响)。

### 3.3 标注
- COCO JSON: **不改 bbox、不改 width/height** (已在显示空间, §2 已验证)。
- YOLO txt: 归一化坐标不变 (数值不变, 分母逻辑在新像素下由 dataset.yaml/转换脚本按新尺寸解释)。

### 3.4 封闭性
- **test 分片完全不修改**: Baseline 与 Ablation 在**完全相同**的 test 上做最终独立评估, 保证可比性。
- 若未来部署需要, 可单独对 test 的 TRAIN_000064_photo-1.jpg 烘焙 (需同步更新其 bbox/尺寸), 但**不纳入本次 Ablation 范畴**。

## 4. 自动校验脚本设计 `scripts/validate_exiffix.py`

对副本执行, 通过才算修复成功:

1. **图像-标注尺寸一致 (核心不变式)**: 对每一张图, `cv2.imread` 尺寸 == JSON `width/height`。预演结果: 修复后 11 张全等 (原 clean 下 11 张不等, 校验需先确认"修复前恰好 11 张不等, 修复后 0 张不等")。
2. **bbox 合法性**: 全量 bbox `x2>x1, y2>y1`, 且 `x2≤W, y2≤H`; 非法数必须为 0。
3. **EXIF 标签清零**: 11 张烘焙后 Orientation 字段不存在或 ==1; 其余图 Orientation 保持原样 (==1 或缺失)。
4. **烘焙正确性 (抽查 11 张)**: 重新读副本图像 `cv2.rotate` 逆变换回原始方向, 与 clean 原始像素做 `np.abs` 像素差; 平均绝对差应 < 3 (仅 JPEG 重压缩噪声), 证明烘焙没有引入内容错位。
5. **YOLO↔COCO 交叉比对**: 副本 YOLO txt 换算的 bbox 与 COCO bbox 逐条一致 (容差 ≤1px), 0 差异。
6. **数量守恒**: 副本图片数/标注数/YOLO txt 数与 clean 完全一致 (train 915 / val 113 / test 114; 3079 / 350 / 423)。
7. 输出 `VALIDATION_REPORT.txt`, 全部 PASS 才允许进入 Ablation 训练。

## 5. 影响面与风险

| 项 | 说明 |
|---|---|
| 收益 | 消除 11 张的 bbox↔像素错位 (img65 实证: 7 GT 全漏 + 大量误检); 修复后预期提升相关类 recall (尤其 Tomato 类)。 |
| 压缩混淆 | 烘焙用 quality=95, 与原片仅重压缩差异; 11 张内容像素变化极小 (|Δ|≈0-3), 对训练影响可忽略, 且已纳入 §4.4 校验。 |
| 标注一致性 | bbox/JSON 零改动, 无标注误差放大风险; 不变式由校验脚本保证。 |
| test 封闭 | test 保持原样, Ablation 两臂 test 输入完全相同。 |
| 失败回滚 | 副本独立; 任何一步失败丢弃副本重建即可, 原始/clean 不受影响。 |

## 6. 待用户确认的执行步骤 (当前未执行)

1. `cp -al` 建副本 → 2. 烘焙 10 张 (train+val) → 3. 跑 `validate_exiffix.py` → 4. 通过后纳入 ABLATION_PLAN 实验。
