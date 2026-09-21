# 智农慧眼 真实农业 RAG 知识库 — 资料需求清单（13 类）

> 类别名称与顺序严格来自 `dataset/processed_detection_exiffix/annotations/val.json` 的 categories（id 升序，1–13）。
> 作物映射沿用 `web/render.py::_crop_by_class_id`：id 1–8 番茄 / 9–11 苹果 / 12–13 葡萄。
> 每类病害需覆盖 5 类字段：**症状 / 发生条件 / 传播途径 / 监测 / 综合防治**。

| 类别ID | 类别名（模型实际输出） | 作物 | 类型 | 病原（学名） | 资料需求字段 |
|---|---|---|---|---|---|
| 1 | Tomato Early blight leaf | 番茄 | 病害 | *Alternaria solani* | 症状/发生条件/传播/监测/防治 |
| 2 | Tomato Septoria leaf spot | 番茄 | 病害 | *Septoria lycopersici* | 症状/发生条件/传播/监测/防治 |
| 3 | Tomato leaf | 番茄 | 健康叶 | —（健康） | 无病害资料需求 |
| 4 | Tomato leaf bacterial spot | 番茄 | 病害 | *Xanthomonas* spp.（*euvesicatoria/perforans/vesicatoria/gardneri*） | 症状/发生条件/传播/监测/防治 |
| 5 | Tomato leaf late blight | 番茄 | 病害 | *Phytophthora infestans* | 症状/发生条件/传播/监测/防治 |
| 6 | Tomato leaf mosaic virus | 番茄 | 病害 | ToMV / TMV（Tobamovirus） | 症状/发生条件/传播/监测/防治 |
| 7 | Tomato leaf yellow virus | 番茄 | 病害 | TYLCV（Begomovirus，烟粉虱传播） | 症状/发生条件/传播/监测/防治 |
| 8 | Tomato mold leaf | 番茄 | 病害 | *Passalora fulva*（叶霉病） | 症状/发生条件/传播/监测/防治 |
| 9 | Apple Scab Leaf | 苹果 | 病害 | *Venturia inaequalis* | 症状/发生条件/传播/监测/防治 |
| 10 | Apple leaf | 苹果 | 健康叶 | —（健康） | 无病害资料需求 |
| 11 | Apple rust leaf | 苹果 | 病害 | *Gymnosporangium juniperi-virginianae*（苹果锈病） | 症状/发生条件/传播/监测/防治 |
| 12 | grape leaf | 葡萄 | 健康叶 | —（健康） | 无病害资料需求 |
| 13 | grape leaf black rot | 葡萄 | 病害 | *Guignardia bidwellii*（黑腐病） | 症状/发生条件/传播/监测/防治 |

## 与现有 Mock 字典的关系（重要）

`web/services/rag.py` 中的 `_KNOWLEDGE_BASE`（10 条）与上述 10 个病害类别**键名对齐**，但**仅为演示数据（mock）**，不冒充权威资料。正式建库时须以本清单 10 个病害类 + 权威来源为准，健康叶 3 类无病害知识。

## 说明

- 病原学名为本清单整理时按公开资料补注，用于后续检索对齐，不作为最终诊断依据；接入前须由植保专业人员复核。
- 作物匹配约束：苹果锈病=苹果/桧柏锈病（Gymnosporangium），葡萄黑腐病=葡萄黑腐病（Guignardia），不可用其他作物同名病害代替。
