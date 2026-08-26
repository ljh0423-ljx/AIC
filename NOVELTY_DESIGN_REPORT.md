# NOVELTY_DESIGN_REPORT — 半监督/弱监督农业病害检测创新方案可行性分析

- 日期: 2026-08-16
- 阶段: **创新方案可行性分析(只读)。未训练、未调参、未碰 TEST、未修改任何现有数据与历史实验。**
- 数据输入:
  - PlantDoc 检测监督数据:`dataset/processed_detection_exiffix`(A0 数据基础),3 作物 / 13 类 / 1144 图 / 3861 目标,train916 val114 test114,seed=2026
  - PV_CORE 分类数据:`/root/autodl-tmp/PV_CORE_19233_audit/`(审计 PASS),19,233 张无 bbox、带精确类标签、13 类 1:1 同义
- 基线: **A0**(PP-YOLOE+-s, exiffix)VAL mAP@0.5:0.95 = **0.4012**;baseline_v1 TEST = 0.401
- 赛事背景: AIC2026 AI+农业(病虫害识别预测),评审含**数据集测评 + 路演答辩** —— 方案须兼顾硬指标与可答辩的算法创新机制。

---

## 0. 结论速览(Executive Summary)

1. **类别映射 1:1 自动核验通过**:PlantDoc 13 类与 PV_CORE 13 类经 `aic_class_id` 完全一一对应,名称语义一致,可安全混用。
2. **PV_CORE 作为"无标注检测图"的直接可用性有限,但作为"精确标签分类监督源"的价值极高**。核心事实:PV_CORE 全部为 **256×256** 固定分辨率、均匀灰背景、单/双叶居中(前景覆盖 0.33~0.61)、无 bbox、无与 PlantDoc train/val 的内容级重叠。
3. **基线真实瓶颈 = 分类混淆 + 数据稀缺**,不是定位能力。此前 6 个方向(方向2/3/4)全部在同一 916 张训练图上做损失侧/推理侧修补均失败;项目唯一实证增益来自**数据修复**(A0 = +0.0148)。→ **"加数据"是剩余最大的、未被触动的杠杆。**
4. **主推方案 S1(CG-ACS 分类监督辅助检测)**:用 19,233 张精确类标签图强化共享骨干的病害判别特征,直接攻击已测得的分类混淆瓶颈;零推理开销、不依赖伪框质量、消融干净、答辩叙事清晰。
5. **备用方案 S2(CTPL 类锚定教师-学生伪标签)**:用 A0 教师对 PV_CORE 生成伪框、**类标签锁定为已知 PlantVillage 真值**(消除伪标签类噪声),扩充检测监督。风险高于 S1(256×256 伪框质量)。
6. **进阶方案 S3(CRL 一致性正则联合学习)**:分类一致性 + 教师-学生伪框的全量 SSL 组合,天花板最高、控制难度最大,作为 S1→S2 成功后的长期延伸。
7. 推荐实施顺序 **S1 → (通过后) S1+S2 组合 → (可选) S3**;A0 全程严格对照;成功判据沿用项目纪律(预注册、差 0.005 门槛、强类无回退)。

---

## 1. 任务定义与硬约束

### 1.1 任务
以 AIC2026_AgriVision 为基线,PlantDoc 为有 bbox 检测监督,PV_CORE 为 19,233 张无 bbox 分类数据,设计**符合 AIC2026 AI+农业赛道要求、具有明确算法创新性**的半监督/弱监督农业病害目标检测方案。本阶段仅分析。

### 1.2 硬约束(必须遵守)
| # | 约束 | 说明 |
|---|---|---|
| C1 | 保留 A0 baseline 严格对照 | 所有新实验以 A0(exiffix)为唯一基准,单变量消融 |
| C2 | 禁止修改 `processed_detection` | 原始数据不动;训练经软链使用 exiffix/clean 副本 |
| C3 | 禁止修改 `baseline_v1` / `direction2` / `direction3` / `direction4` | 历史实验完整封存;新实验全部落在新目录(建议 `experiments/novelty_*/`) |
| C4 | TEST 完全封闭 | 仅 VAL 评估/选择;TEST 仅经用户明确批准后做最终评估 |
| C5 | 不引入通用模块套壳 | 沿用项目纪律:不新增 CBAM/SE/Transformer 等通用注意力;创新须针对已测瓶颈且可严格消融 |
| C6 | 当前阶段只读分析 | 不训练、不调参、不生成伪标签(伪标签生成属实施阶段) |

---

## 2. 数据核验与统计(本次只读自动核验结果)

### 2.1 PlantDoc ↔ PV_CORE 13 类一一对应(自动核验,全对)

| aic_class_id | PlantDoc 类名(label_list) | PV_CORE 目录 | 作物 | 映射 |
|---:|---|---|---|---|
| 0 | Tomato Early blight leaf | Tomato___Early_blight | Tomato | ✅ |
| 1 | Tomato Septoria leaf spot | Tomato___Septoria_leaf_spot | Tomato | ✅ |
| 2 | Tomato leaf | Tomato___healthy | Tomato | ✅ |
| 3 | Tomato leaf bacterial spot | Tomato___Bacterial_spot | Tomato | ✅ |
| 4 | Tomato leaf late blight | Tomato___Late_blight | Tomato | ✅ |
| 5 | Tomato leaf mosaic virus | Tomato___Tomato_mosaic_virus | Tomato | ✅ |
| 6 | Tomato leaf yellow virus | Tomato___Tomato_Yellow_Leaf_Curl_Virus | Tomato | ✅ |
| 7 | Tomato mold leaf | Tomato___Leaf_Mold | Tomato | ✅ |
| 8 | Apple Scab Leaf | Apple___Apple_scab | Apple | ✅ |
| 9 | Apple leaf | Apple___healthy | Apple | ✅ |
| 10 | Apple rust leaf | Apple___Cedar_apple_rust | Apple | ✅ |
| 11 | grape leaf | Grape___healthy | Grape | ✅ |
| 12 | grape leaf black rot | Grape___Black_rot | Grape | ✅ |

核验方式:PV_CORE `file_manifest.csv` 的 `aic_class_id` ↔ PlantDoc `label_list.txt` 顺序 id;13=13,无缺无多,`pv_category` 与目录名一致(前一阶段审计亦确认)。

### 2.2 两类数据集各类数量与不平衡度

| cls | PlantDoc 名称 | PV_CORE 目录 | **PV 图数** | PD train图 | PD train目标 | PD val目标 | PD test目标 | 合并图数(PV+PDtrain) |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Early blight | Tomato___Early_blight | 1000 | 73 | 171 | 21 | 21 | 1073 |
| 1 | Septoria | Tomato___Septoria_leaf_spot | 1771 | 119 | 338 | 37 | 55 | 1890 |
| 2 | Tomato leaf(健康) | Tomato___healthy | 1591 | 56 | 294 | 36 | 66 | 1647 |
| 3 | bacterial spot | Tomato___Bacterial_spot | 2127 | 91 | 227 | 16 | 35 | 2218 |
| 4 | late blight | Tomato___Late_blight | 1909 | 87 | 159 | 26 | 34 | 1996 |
| 5 | mosaic virus | Tomato___Tomato_mosaic_virus | **373** | 43 | 208 | 33 | 20 | 416 |
| 6 | yellow virus | Tomato___Tomato_Yellow_Leaf_Curl_Virus | **5357** | 59 | 700 | 57 | 67 | 5416 |
| 7 | mold leaf | Tomato___Leaf_Mold | 952 | 72 | 239 | 32 | 20 | 1024 |
| 8 | Apple Scab | Apple___Apple_scab | 630 | 75 | 125 | 30 | 16 | 705 |
| 9 | Apple leaf(健康) | Apple___healthy | 1645 | 73 | 200 | 14 | 33 | 1718 |
| 10 | Apple rust | Apple___Cedar_apple_rust | **275** | 70 | 149 | 12 | 17 | 345 |
| 11 | grape leaf(健康) | Grape___healthy | 423 | 55 | 173 | 18 | 29 | 478 |
| 12 | black rot | Grape___Black_rot | 1180 | 52 | 101 | 22 | 10 | 1232 |
| | **合计** | | **19233** | 925* | 3084 | 354 | 423 | 20158 |

*925 > 916:部分图多类目标(单图含多类),按类计数相加 > 图像总数。

**不平衡度**:
- PV_CORE 图数 max/min = 5357/275 = **19.48×**(比 PlantDoc train 目标 6.93× 更极端)
- PlantDoc train 目标 max/min = 700/101 = **6.93×**;train 图/类 max/min = 119/43 = 2.77×
- 合并后每类图数 345~5416(15.7×)——仍需在分类流中做平衡采样

### 2.3 图像属性差异(决定 PV_CORE 使用方式的关键)

| 属性 | PlantDoc(检测监督) | PV_CORE(分类) |
|---|---|---|
| 分辨率 | W min115/med700/max6000, H min69/med600/max6000 | **固定 256×256**(13 类全采样验证) |
| 每图目标数 | train 均值 3.37, max 42;密集图(≥5目标)= 218 张 | 无 bbox;图像为单/双叶居中 |
| 背景 | 真实田间/网页/多源照片背景 | **均匀灰背景**(各角像素近似一致) |
| 前景覆盖 | 多目标散布 | 0.33~0.61(叶片占满画面,行/列覆盖≈1.0) |
| 类标签 | 有,bbox 级 | **有,图像级且精确**(PlantVillage 真值) |
| 内容重叠 | — | 与 PlantDoc train/val **0 对 hamming≤6**(本阶段核验);TEST 侧核验列为实施门槛 |

### 2.4 内容级去重(泄露检测)
- **PV_CORE ↔ PlantDoc train/val:dHash(hamming≤6)= 0 对**,训练侧无内容级重叠,无泄露。
- **PV_CORE ↔ PlantDoc TEST:本阶段不读 TEST(约束 C4);实施阶段第一条强制门槛**——用相同 dHash 对 TEST 全集做交叉核验,PASS 后方可使用 PV_CORE。

### 2.5 数据卫生备注(不影响任何实验)
- 原始 `dataset/processed_detection` 目录含 **2 个 0 字节孤儿空文件**(train `TRAIN_000133_...928225.jpg`、val `TRAIN_000104_...556232.jpg`),其 COCO 标注引用仍存在(陈旧引用),但 **clean/exiffix 变体均不包含、不引用这两图**,全部历史训练/评估(经软链读 exiffix)从未接触它们。
- 结论:原始目录的 2 个空文件仅属归档卫生问题,**不得删除**(约束 C2);新方案不使用原始目录,无影响。

---

## 3. 基线瓶颈分析(既有诊断结论回顾)

### 3.1 现状
| 项 | 值 |
|---|---|
| A0 VAL mAP@0.5:0.95 | 0.4012 |
| baseline_v1 TEST | 0.401 |
| Tomato 8类 mAP(占74%目标) | **0.248**(Apple 0.646 / Grape 0.560) |
| 定位类无关召回 / 逐类召回 | 58.0% / 41.1% |
| **定位对但类错** | 59/203 定位成功 = **29%** |

### 3.2 真实瓶颈(证据)
1. **组内病害分类混淆(主瓶颈)**:89 次组内混淆 vs 9 次跨作物;最大混淆边 `bacterial spot→Septoria` 占 GT **31.2%**、`mosaic→yellow` 15.2%、`Scab→rust` 13.3%;混淆全发生在同一作物内、共享相似叶面症状。
2. **密集场景置信度压低**:R@conf0.5 = 47.1% vs R@conf0.1 = 97.1%;漏检框实际 IoU 均值 0.76(框质量高,纯校准问题);yellow/mosaic/mold 密集图全图无 ≥0.5 检测。
3. **小数据 916 张**:弱类每类 train 图仅 43~73 张(mosaic 43、black rot 52、EB 73)。
4. 小目标非瓶颈(VAL 仅 4 个小目标)。

### 3.3 为什么"加数据"是剩余最大杠杆(项目经验)
- **已失败方向(全在同一 916 图上做修补)**:方向2 CGPM/CCDH(分类边界损失)——无效;方向3 class-weighted VFL——Recall 升 mAP 不升;方向4 D1 密集感知置信度校准——**0.4061,差门槛 0.0001 FAIL**,且强类回退;D2 推理侧重打分——0.4008 < A0。
- **唯一实证增益来自数据**:A0(exiffix EXIF 修复)**+0.0148**,是全项目最大单次增益。
- 推论:损失侧/推理侧已近饱和,**尚未触碰的大变量是训练数据量(916 图)+ 分类判别监督(19,233 张精确标签)**。

---

## 4. PV_CORE 作为无标注检测图的可行性分析

### 4.1 利
- 类标签**精确且同义**(PlantVillage 真值,1:1 映射),无标注噪声。
- 规模大:19233 vs 916(21×),覆盖弱类(mosaic 43→416、black rot 52→1232、EB 73→1073)。
- 与 PlantDoc 同源病害形态(PlantVillage 家族),特征迁移友好。
- 训练侧无内容级泄露。

### 4.2 弊(限制"直接当检测图"的价值)
- **256×256 低分辨率**:PP-YOLOE train 输入 640(多尺度 320~768),上采样后模糊,定位学习收益有限。
- **单/双叶居中 + 均匀灰背景**:非自然多目标场景;"在 PV_CORE 上做检测"近似退化为 1~2 个大框;对已测瓶颈中的"密集多目标置信度校准"没有直接数据。
- **背景域差异**:灰实验室背景 vs PlantDoc 真实场景,可能引入背景捷径。
- **类不平衡更极端**(19.5×),直接混训需平衡。
- 无 bbox:框必须靠教师生成(伪框)或弱监督定位,引入噪声。

### 4.3 结论
> PV_CORE 的**精确类标签**是其最大、最独特的资产;它的最佳角色是**分类监督源**(锐化病害判别特征),而非**伪框检测源**(低分辨率、退化框、域差异)。伪框路线(S2)可行但必须:①类标签锁定为已知真值(消除伪标签类噪声);②只采高置信框;③把 256×256 上采样损失计入预期。**主路径应优先走 S1 分类监督。**

---

## 5. 三套候选创新方案

### 5.1 S1(CG-ACS 主推)— Classification-Grounded Auxiliary Classification Supervision(分类监督辅助检测)

**定位**:数据驱动的多任务/半监督结构,零推理开销。

**创新机制**:
- 检测网络(backbone+neck 共享)外挂一个**图像级分类辅助头**(aux head:neck 特征全局池化 → 小 MLP → 13 类 logits)。
- 双数据流同 batch 训练:
  - 流 A(检测监督):PlantDoc exiffix 916 图,bbox+class → 完整检测损失(VFL+IoU+DFL),**与 A0 逐字段一致**。
  - 流 B(分类监督):PV_CORE 19233 图,图像级 class → `λ_cls · CE(image_logits, label)`。
- 损失 = `L_det + λ_cls · L_cls`。
- **新颖点**:不是给检测头加 loss 权重(CGPM/class-weighted 已失败),而是**注入 21× 新判别样本**,让共享骨干学到"细菌性斑点 vs 早疫病 vs 健康"的可分特征;分类任务与检测任务共享判别能力,直接对准混淆瓶颈。
- 推理:aux head 仅训练用,推理网络 = 原 PP-YOLOE+-s 结构,零开销、FPS 不变。

**数据使用**:PV_CORE 全部 19233 张仅作图像级监督(不生成伪框);分类流按类平衡采样(缓解 19.5× 不平衡);检测:分类 batch 比例 R ∈ {1:1, 1:3, 1:5}。

**是否需要生成伪 bbox**:否(核心优势)。

**计算开销**:训练 +15~25%(分类流为 256×256,廉价);推理 0。

**实现难度**:中。新增分类 Dataset + aux head 模块 + 双流 sampler;改动隔离在新增模块与新增配置,`PPYOLOEHead` 主体不动(或薄继承)。

**解决的瓶颈**:分类混淆 P1(主)、弱类判别、precision。

**风险/缓解**:辅助任务与检测特征竞争 → λ_cls 灵敏度 {0.1,0.5,1.0};强类回退(D1 教训)→ 监测强类 AP 回退 >0.03 判 FAIL;域差异 → aux head 特征与检测 head 弱耦合(梯度截断选项)。

**答辩叙事**:数据重心型创新——"19,233 张精确标签分类图 + 916 张检测图,多任务判别增强,以最小改动(1 个 aux head + 1 项 CE)攻破小样本细粒度病害混淆"。

---

### 5.2 S2(CTPL 备用)— Class-anchored Teacher-Student Pseudo-Labeling(类锚定教师-学生伪标签)

**定位**:半监督检测(伪框/伪标签 + EMA teacher-student),最直接"用上 PV_CORE 的框"。

**创新机制**:
- **阶段0**:冻结 A0 为教师,对 19233 张 PV_CORE 推理生成框;框类 = 已知 PlantVillage 标签(不取教师 softmax 类),只取教师框坐标与 objectness。
  - **"类锚定"是核心新颖点**:标准 SSL 检测的伪标签类错误(在本数据集正是最难的问题——bact↔Septoria 等混淆)被直接消除,因为分类图自带精确类标签。
- **阶段1**:学生 = 在 916 检测图 + 伪标注 PV_CORE 上训练;未标注损失加权 λ_unsup;EMA 学生→教师持续迭代(PP-YOLOE 已内置 EMA)。
- **可选迭代**:训练若干 epoch 后以改进教师重生成伪框(round 2)。
- **256×256 处理**:教师以 256×256(或 pad 至 640)前向,框坐标回映射;仅保留 objectness ≥ τ_high 的框(τ∈{0.4,0.5,0.6});单/双叶图 → 每图 1~2 框,等价"叶片级"框级增广。

**数据使用**:PV_CORE 作伪标注检测数据(全部或置信子集);类标签固定为真值。

**是否需要生成伪 bbox**:是(教师生成,类锁定)。

**计算开销**:一次性 19233 次教师前向(分钟级)+ 训练 +30~50%。

**实现难度**:高(伪标签数据集管线、EMA 教师编排、256×256 前向与坐标映射、loss masking)。

**解决的瓶颈**:数据稀缺/泛化(弱类框样本扩充,mosaic 43→416 图、black rot 52→1232 图);对密集置信度瓶颈帮助有限。

**风险/缓解**:256×256 伪框质量、灰背景域差、强类干扰 → 高 τ、类锚定、λ_unsup 退火、严格监测;失败则如实记录(沿用纪律)。

---

### 5.3 S3(CRL 进阶)— Consistency-Regularized Joint SSL(一致性正则联合学习)

**定位**:全量半监督框架,天花板最高、控制最难;作为 S1→S2 之后的可选延伸,不建议首跑。

**创新机制**(三个正则的组合):
1. **图像级分类一致性(FixMatch 式)**:同一 PV_CORE 图的强弱两视角,图像级分类预测应一致;高置信伪类锚定已知标签,防漂移。
2. **框级教师-学生伪标签**(同 S2,仅高置信子集)。
3. **裁剪级一致性**:随机裁剪叶片区域,其分类与整图一致(显式利用"单叶居中"先验)。
- 损失 = `L_det + λ_cls·L_cls + λ_cons·L_cons + λ_unsup·L_unsup`。

**计算开销**:训练 +50~100%。**实现难度**:很高(三重正则的调度/掩码/消融)。
**解决的瓶颈**:全部(混淆 + 数据 + 校准),但对每一环的归因依赖精细消融。

---

## 6. 方案对比与 AIC2026 赛道契合度判断

| 维度 | S1 CG-ACS | S2 CTPL | S3 CRL |
|---|---|---|---|
| 直击主瓶颈(分类混淆) | ★★★ | ★★(框学习为主) | ★★★ |
| 用足 PV_CORE 核心资产(精确类标签) | ★★★ | ★★★ | ★★★ |
| 不依赖伪框质量(256×256) | ★★★(零伪框) | ★(核心依赖) | ★★ |
| 推理开销 | 0 | 0 | 0 |
| 训练开销 | +15~25% | +30~50% | +50~100% |
| 实现难度 | 中 | 高 | 很高 |
| 消融干净度 | ★★★(单变量 λ_cls/R) | ★★(多机制耦合) | ★(三重正则) |
| 失败风险 | 低-中 | 中-高 | 高 |
| 答辩叙事清晰度 | ★★★(数据重心,易懂可证) | ★★★(类锚定有记忆点) | ★★(机制多,归因难) |

### 6.1 推荐结论
**主推 S1(CG-ACS)**,理由:
1. **直击已测瓶颈**:数据证据表明瓶颈是"组内病害混淆 + 判别数据不足",S1 从判别特征层面注入 21× 新监督,是六次失败方向均未触碰过的**数据维度**变量。
2. **最大化 PV_CORE 独特价值**:精确类标签被 100% 利用;不因 256×256 低分辨率伪框而折损。
3. **风险最低**:不依赖伪框质量、零推理开销、单变量可消融、强类回退可监测(吸收 D1 教训)。
4. **与赛道评审契合**:数据集测评(mAP 提升概率最高)+ 路演答辩(机制简洁可证、消融完备)。
5. **项目历史背书**:唯一实证增益来自数据(A0 +0.0148);S1 是"数据增益"的自然延伸,而非又一次损失修补。

**备用 S2(CTPL)**:作为 S1 的对照与补充。若 S1 通过,S1+S2 组合(分类监督 + 类锚定伪框)是自然进阶——S1 先提升判别力 → 教师更准 → S2 伪框更干净。

**S3(CRL)**:仅在 S1/S2 验证了"半监督方向有效"后作为长期延伸,不进入首轮实验矩阵。

---

## 7. 实验矩阵(实施阶段执行;本阶段不训练)

> 全部实验使用 **A0(exiffix)为数据基础 + A0 配置逐字段不变**,唯一变量为标注列;落在新目录 `experiments/novelty_S1/`、`experiments/novelty_S2/` 等;VALID 用 VAL,TEST 封闭。

| 实验 | 方案 | 变量 | 与 A0 唯一差异 | 目的 |
|---|---|---|---|---|
| A0 | 对照 | — | — | 严格基线(VAL 0.4012),复用已有 checkpoint |
| N1-a | S1 | λ_cls=0.5, R=1:3 | +aux head +分类流 | 主实验:分类监督增益 |
| N1-b | S1 | λ_cls=0.1 / 1.0 | 仅 λ 变化 | λ 灵敏度 |
| N1-c | S1 | R=1:1 / 1:5 | 仅 R 变化 | 双流比例 |
| N1-d | S1 | 分类流类平衡采样 on/off | 仅采样 | 不平衡处理必要性 |
| N1-e(对照) | S1 | λ_cls=0(aux 头挂但 loss 置 0) | 仅"结构空跑" | 证明增益来自监督而非结构(单变量纯度) |
| N2-a | S2 | τ=0.5, λ_unsup 默认 | +伪标注流 | 备用:类锚定伪标签 |
| N2-b | S2 | τ=0.4 / 0.6 | 仅 τ | 伪框置信阈值 |
| N3-a | S1+S2 | 组合 | N1 最优 + N2 最优 | 组合路径(条件触发:N1 与 N2 至少其一通过) |

### 7.1 成功判据(预注册,沿用项目纪律)
- 主判据:VAL mAP@0.5:0.95 ≥ A0(0.4012) + **0.005** = **0.4062**。
- 辅助判据:①命名混淆对(bact↔Septoria、mosaic↔yellow、Scab↔rust)per-class AP 多数提升;②强类(Apple leaf/grape leaf/Apple Scab)无回退 >0.03;③TEST 不参与任何选择。
- 任一 FAIL 即如实记录,不调参不重试(D1 纪律)。

---

## 8. 实施步骤(仅分析完成后执行;每步需用户批准)

### 阶段 0 — 门槛核验(必做)
1. **TEST 交叉去重**:dHash 对 PV_CORE 全集 ↔ PlantDoc test 全集,hamming≤6 对数为 0 才允许使用 PV_CORE(需读取 TEST 文件头,仅核验不训练/不评估)。
2. 复核 PV_CORE md5(前阶段审计 PASS,再确认)。
3. 确认 PaddleDetection 训练管线可读 256×256 分类图(采样 100 张入 DataLoader smoke)。

### 阶段 1 — 数据管线
4. 新建 `dataset/PV_CORE_cls/`:类别 id 对齐 `label_list.txt` 的 csv 索引(不复制图片,引用 `PV_CORE_19233_audit/` 路径,只读)。
5. 分类流 Dataset + 类平衡 sampler + 双流 batch 采样器(检测:分类 = R)。

### 阶段 2 — 模块实现(S1 优先)
6. `ppdet/modeling/heads/aux_cls_head.py`(薄新增)+ 配置开关;`PPYOLOEHead` 主体不动。
7. 训练脚本:新增 `--novelty_s1` 开关;全部产物写 `experiments/novelty_S1/`,不触碰历史目录。
8. S2(如执行):伪标签生成脚本(教师 A0 前向 + 类锚定 + 坐标回映射)+ 伪标注 Dataset + EMA 编排;`experiments/novelty_S2/`。

### 阶段 3 — Smoke
9. 2~3 epoch smoke:loss 曲线正常、无 OOM、aux 损失下降、单变量可复现(seed 固定)。

### 阶段 4 — 正式消融
10. 按 §7 矩阵依次运行;每轮只改一个变量;VAL 评估;产出对比 CSV/图(沿用项目报告风格)。

### 阶段 5 — 判定与决策
11. 依 §7.1 判据逐项判定;形成 `experiments/novelty_*/NOVELTY_REPORT.md`;
12. TEST 最终评估仅在用户明确批准后执行(单次,封闭)。

---

## 9. 风险与结论

### 9.1 风险清单
| 风险 | 等级 | 缓解 |
|---|---|---|
| aux 分类任务与检测任务特征竞争,强类回退 | 中 | λ 灵敏度、强类回退监测、梯度截断选项 |
| PV_CORE 灰背景域差引入捷径 | 中 | aux 头弱耦合、类平衡采样、结果归因分析 |
| S2 256×256 伪框质量差 | 高 | 类锚定 + 高 τ + 只取高置信子集 + 坐标映射正确性验证 |
| PV_CORE 与 TEST 潜在重叠 | 低(待核) | 阶段0 强制 dHash 门槛 |
| 分类监督收益不足以跨越 0.005 | 中 | 诚实记录;S2/组合路径兜底;不调参 |

### 9.2 最终结论
1. 数据可行性:**通过**。类别 1:1、训练侧无泄露、PV_CORE 审计完整。
2. 创新方案:**主推 S1(CG-ACS 分类监督辅助检测)** —— 数据重心、直击分类混淆瓶颈、零推理开销、消融干净、答辩可证;**备用 S2(CTPL 类锚定教师-学生伪标签)**;**进阶 S3(CRL)**。
3. A0 严格对照,历史实验(processed_detection / baseline_v1 / direction2/3/4)全程不动;TEST 封闭。
4. 本报告为纯分析产物,未训练、未调参、未生成伪标签、未碰 TEST。

---

## 附录 A:关键数据复核脚本路径(只读)
- PV_CORE 完整性审计:`PV_CORE_REMOTE_AUDIT.md`(前阶段,全部 PASS)
- 类别映射/数量统计:本次 `python3` 内联脚本(读 `file_manifest.csv` + COCO JSON,只读)
- 图像分辨率:JPEG SOF 头扫描(780 样本全 256×256)
- 内容重叠:dHash(64bit,hamming≤6)PV_CORE ↔ PlantDoc train/val = 0 对

## 附录 B:引用
- AIC2026 农业赛道对应 CICAS"AI+绿色农牧场景"(病虫害识别预测): https://cicas.cn/Reg/Detail/245
