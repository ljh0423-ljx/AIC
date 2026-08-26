# PROJECT_DOCUMENT_TODO — 正式文档材料待办清单

> 生成时间：2026-08-18
> 本清单用于后续撰写**比赛申报书 / 技术文档 / PPT / 答辩稿**时，对照检查还缺哪些材料。
> 每项标注三档状态之一：
> - ✅ **已有证据**：原始报告中已存在、可直接引用（给出证据文件）
> - ⬜ **需要补充**：尚缺，但可从现有数据/代码/环境**合法生成**（需在明确授权下另行制作）
> - ⛔ **禁止猜测**：不得编造/虚构，没有证据就保持空缺

---

## 1. 数据集相关

| 材料 | 状态 | 说明 / 证据 |
|---|---|---|
| 数据集名称（PlantDoc） | ✅ 已有 | PROJECT_FINAL_STATUS.md §1 |
| 13 类完整英文名称 | ✅ 已有 | `dataset/processed_detection/label_list.txt`、PROJECT_FINAL_STATUS.md §4 |
| 数据集规模（3 作物/13 类/1144 图/3861 目标） | ✅ 已有 | DATASET_FINAL_REPORT.md |
| 官方 GitHub 地址 / 论文出处 / CC-BY-4.0 | ✅ 已有 | `D:\Fruit\PLANTDOC_DETECTION_AUDIT.md` |
| 官方下载 URL、license 原文、论文 DOI/链接 | ⬜ 需要补充 | 现有报告未记录 URL/license 原文 |
| 中文类名映射 | ⬜ 需要补充 | 报告多处注明「无官方中文类名映射」；填写 `CLASS_NAMES_CN` 后自动启用双语 |
| 训练/验证/测试划分数字 | ✅ 已有 | PROJECT_FINAL_STATUS.md §1（train 915/val 113/test 114） |
| 每类样本分布表/柱状图 | ⬜ 需要补充 | 每类 train 图数/目标数在 label_list.txt / PROJECT_FINAL_STATUS.md §4 有数据，图需另制 |
| 数据集样例图（含 bbox 可视化） | ⬜ 需要补充 | 有 `visual_check/` 与可视化脚本，需另行出图 |

---

## 2. 模型与训练

| 材料 | 状态 | 说明 / 证据 |
|---|---|---|
| 模型名称/结构（PP-YOLOE+-m） | ✅ 已有 | PROJECT_FINAL_STATUS.md §3.1 |
| checkpoint 路径与 md5 | ✅ 已有 | FREEZE_CHECK.json |
| 训练环境（AutoDL/4090/Paddle2.6.2） | ✅ 已有 | AUTODL_ENV_CHECK.md |
| 训练参数（epoch/bs/lr/EMA/预训练） | ✅ 已有 | PROJECT_FINAL_STATUS.md §3.2 / config |
| 训练过程记录（loss 曲线、mAP 随 epoch 变化） | ⬜ 需要补充 | 原始 train.log、vdlrecords 存在但未整理成图/表 |
| 消融实验对比表 | ✅ 已有 | PROJECT_EXPERIMENT_INDEX.md §3.6（各方向 VAL mAP 汇总） |
| 最终模型指标（VAL/TEST mAP 等） | ✅ 已有 | M_FINAL_TEST_REPORT.md |
| 13 类每类 AP 表 | ✅ 已有 | M_FINAL_TEST_REPORT.md §3.3 |

---

## 3. 推理与性能

| 材料 | 状态 | 说明 / 证据 |
|---|---|---|
| GPU 批量 FPS（12.84） | ✅ 已有 | GPU_BATCH_INFERENCE_REPORT.md |
| GPU Web 批量 FPS（12.00） | ✅ 已有 | GPU_WEB_VALIDATION_REPORT.md |
| CPU 批量 FPS（0.42） | ✅ 已有 | INFERENCE_VALIDATION_REPORT.md |
| CPU/GPU 对比（约 30×） | ✅ 已有 | GPU_BATCH_INFERENCE_REPORT.md §四 |
| 真实 FPS 口径说明 | ✅ 已有 | 多报告已注明口径（见索引 §3.5） |
| TensorRT 加速实测 | ⛔ 禁止猜测 | 暂无实测（BASELINE_SUMMARY 注明未启用） |
| RK3588 / RKNN 实测 FPS | ⛔ 禁止猜测 | 未部署/未转换，仅接口预留 + 预估（LOCAL_DEPLOYMENT_AUDIT.md） |
| 端到端耗时分解（预处理/前向/后处理） | ✅ 已有 | GPU_SINGLE_INFERENCE_REPORT.md §四 |

---

## 4. Web 系统

| 材料 | 状态 | 说明 / 证据 |
|---|---|---|
| 功能清单（14 项） | ✅ 已有 | GPU_WEB_VALIDATION_REPORT.md §十 |
| Backend 架构（DetectorBackend + 注册表） | ✅ 已有 | WEB_DEMO_FINAL_REPORT.md §6 |
| 15 张演示案例结果 | ✅ 已有 | DEMO_CASES_REPORT.md |
| 系统界面截图 | ⬜ 需要补充 | 报告为文字验收，无截图 |
| 技术架构图 / 流程图 | ⬜ 需要补充 | 需另行绘制 |
| 系统演示视频/GIF | ⬜ 需要补充 | 需另行录制 |

---

## 5. 项目价值 / 创新点 / 应用场景

| 材料 | 状态 | 说明 / 证据 |
|---|---|---|
| 创新方案（CG-ACS/CTPL/CRL 设计） | ✅ 已有（设计文档，但**均判定失败/未落地**） | NOVELTY_DESIGN_REPORT.md + 各方向最终报告（S1/B1-PVPRETRAIN FAIL） |
| **最终落地模型的创新点** | ⛔ 禁止猜测 | 现有证据表明最终模型为「官方 PP-YOLOE+ 结构 + EXIF 数据修复 + s→m 容量放大」，无自定义算法模块落地（PROJECT_FINAL_STATUS.md §2/§3.1）；如实陈述，不夸大 |
| 应用场景（番茄/苹果/葡萄病害检测） | ✅ 已有 | WEB_DEMO_FINAL_REPORT.md §2（3 作物场景） |
| 量化价值（省成本/提产量等） | ⛔ 禁止猜测 | 报告明确「未加入任何虚假准确率/节省成本指标」 |
| 与现有方法对比（同类文献对比） | ⬜ 需要补充 | 无对比实验记录 |

---

## 6. 需要补充的「禁止猜测」红线提醒

1. **RK3588 / TensorRT / NPU 任何 FPS 数字**：均无实测，只能写「接口已预留、尚未部署」，禁止写具体加速倍数。
2. **创新点**：最终模型是官方结构 + 数据修复 + 容量放大，**没有**已落地的自定义损失/模块（S1、B1-PVPRETRAIN、D1、D2、B1、A1/A2 全部失败）。申报时务必按证据如实描述，避免虚构「提出并验证了 XX 新算法」。
3. **数据集官方 URL / license 原文**：引用前需先补全，不要凭记忆填写链接。
4. **中文类名**：无官方映射，禁止自造中文类名硬写进文档（可注明「英文类名为官方标签」）。
5. **FPS 引用**：必须标注口径（冷启动/预热稳态、单图/批量、CPU/GPU、eval/推理流水线），不可混用 16.6 / 12.84 / 0.42 等不同口径数值。

---

## 7. 建议的后续产出（需在明确授权下另行制作，本阶段未做）

| 待产出物 | 用途 | 数据来源 |
|---|---|---|
| 系统界面截图（首页/结果页/演示模式/农业结果卡） | PPT/答辩稿 | 运行 Web 截图（需授权启动 Web） |
| 技术架构图（数据流：上传→DetectorBackend→后处理→可视化→统计） | 技术文档 | web/ + inference/ 代码结构 |
| 训练 mAP 曲线图 | 技术文档/答辩 | train.log / vdlrecords |
| 检测结果样例图（15 演示案例 + 可视化 bbox） | PPT | demo_outputs/ + 各 run visualized/ |
| 每类 AP 柱状图 | 技术文档 | M_FINAL_TEST_REPORT.md §3.3 数据 |

---

*本清单与 `PROJECT_EXPERIMENT_INDEX.md`、`PROJECT_FINAL_FACTS.md` 配套使用；状态栏标注以现有报告证据为准。*
