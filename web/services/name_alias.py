# -*- coding: utf-8 -*-
"""
web/services/name_alias.py — 病害/作物名称别名规范化
====================================================
把用户输入的中文病名、常见中文别名、英文名（大小写/空格差异）统一映射到
val.json 的标准英文类别名（小写），供 RAG 检索过滤与缓存键使用。

- 规范键 = val.json categories 的英文名小写（与向量索引 chunk 的 disease_key 一致）。
- 仅收录「作物+病害」完整名称与已核对别名；不含作物的简称（如“早疫病”“锈病”）不收录，
  避免把不同作物或不同病害错误映射为同一类别。
- 无法确认的名称一律返回 unknown，由调用方返回 no_evidence，不猜测。
"""

# 标准英文类别名（val.json 小写）为唯一规范键
DISEASE_ALIASES: dict[str, str] = {
    # —— 番茄 ——
    "tomato early blight leaf": "tomato early blight leaf",
    "番茄早疫病": "tomato early blight leaf",
    "tomato septoria leaf spot": "tomato septoria leaf spot",
    "番茄斑枯病": "tomato septoria leaf spot",
    "番茄白星病": "tomato septoria leaf spot",
    "tomato leaf bacterial spot": "tomato leaf bacterial spot",
    "番茄细菌性斑点病": "tomato leaf bacterial spot",
    "番茄细菌性斑疹病": "tomato leaf bacterial spot",
    "tomato leaf late blight": "tomato leaf late blight",
    "番茄晚疫病": "tomato leaf late blight",
    "tomato leaf mosaic virus": "tomato leaf mosaic virus",
    "番茄花叶病毒病": "tomato leaf mosaic virus",
    "番茄花叶病": "tomato leaf mosaic virus",
    "tomato leaf yellow virus": "tomato leaf yellow virus",
    "番茄黄化曲叶病毒病": "tomato leaf yellow virus",
    "番茄黄化曲叶病": "tomato leaf yellow virus",
    "tylcv": "tomato leaf yellow virus",
    "tomato mold leaf": "tomato mold leaf",
    "番茄叶霉病": "tomato mold leaf",
    # —— 苹果 ——
    "apple scab leaf": "apple scab leaf",
    "苹果黑星病": "apple scab leaf",
    "苹果疮痂病": "apple scab leaf",
    "apple rust leaf": "apple rust leaf",
    "苹果锈病": "apple rust leaf",
    "苹果赤星病": "apple rust leaf",
    # —— 葡萄 ——
    "grape leaf black rot": "grape leaf black rot",
    "葡萄黑腐病": "grape leaf black rot",
}

HEALTHY_ALIASES: dict[str, str] = {
    "tomato leaf": "tomato leaf",
    "番茄健康叶": "tomato leaf",
    "番茄健康": "tomato leaf",
    "apple leaf": "apple leaf",
    "苹果健康叶": "apple leaf",
    "苹果健康": "apple leaf",
    "grape leaf": "grape leaf",
    "葡萄健康叶": "grape leaf",
    "葡萄健康": "grape leaf",
}

CROP_ALIASES: dict[str, str] = {
    "番茄": "番茄", "西红柿": "番茄", "tomato": "番茄",
    "苹果": "苹果", "apple": "苹果",
    "葡萄": "葡萄", "grape": "葡萄",
}


def _norm_text(s) -> str:
    """小写 + 去除首尾空白 + 折叠内部连续空白，统一大小写/空格差异。"""
    return " ".join((s or "").strip().lower().split())


def normalize_crop(crop_name) -> str | None:
    """作物名 → 标准作物（番茄/苹果/葡萄）；无法识别返回 None。"""
    return CROP_ALIASES.get(_norm_text(crop_name))


def normalize_disease(disease_name):
    """病害名 → (standard_name, kind, alias_matched)。

    kind: disease / healthy / empty / unknown
    standard_name: val.json 英文类别名（小写）；unknown 时为空串。
    alias_matched: 输入经规范化后与标准英文名不同（即经中文/别名映射命中）时为 True。
    """
    s = _norm_text(disease_name)
    if not s:
        return "", "empty", False
    if s in DISEASE_ALIASES:
        std = DISEASE_ALIASES[s]
        return std, "disease", (s != std)
    if s in HEALTHY_ALIASES:
        std = HEALTHY_ALIASES[s]
        return std, "healthy", (s != std)
    return "", "unknown", False


def crop_from_disease(standard_name: str) -> str | None:
    """从标准英文类别名推断作物（tomato/apple/grape → 番茄/苹果/葡萄）。"""
    n = (standard_name or "").lower()
    if n.startswith("tomato"):
        return "番茄"
    if n.startswith("apple"):
        return "苹果"
    if n.startswith("grape"):
        return "葡萄"
    return None


# 最长别名优先，避免「番茄花叶病」被「番茄花叶病毒病」误截断；英文名/别名统一小写匹配。
_ALL_ALIASES = sorted(set(list(DISEASE_ALIASES) + list(HEALTHY_ALIASES)), key=len, reverse=True)


def extract_disease_name(text: str):
    """从自由文本中提取已知病害/健康叶名称。

    返回 (standard_name, kind, alias_matched, matched_alias)
      kind: disease / healthy / unknown；unknown 时 standard_name 与 matched_alias 为空串。
    未命中任何已知名称时返回 unknown（不猜测）。
    """
    t = _norm_text(text)
    if not t:
        return "", "empty", False, ""
    for alias in _ALL_ALIASES:
        if alias in t:
            if alias in DISEASE_ALIASES:
                std = DISEASE_ALIASES[alias]
                return std, "disease", (alias != std), alias
            std = HEALTHY_ALIASES[alias]
            return std, "healthy", (alias != std), alias
    return "", "unknown", False, ""
