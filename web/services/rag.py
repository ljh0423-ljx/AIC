# -*- coding: utf-8 -*-
"""
web/services/rag.py — 农业知识增强 RAG 诊断模块（阶段3）
==========================================================
在 PP-YOLOE 检测 + 严重程度评估 + VLM 诊断基础上，检索农业知识库，
为诊断结果补充「病害知识 / 防治方法 / 推荐防治方案 / 注意事项」。

当前阶段（mock）：使用本地 Python 字典知识库（LocalKnowledgeBaseBackend），
不接入真实向量数据库。

未来接入（预留，保持 query_knowledge 签名不变）：
    - FAISS
    - Chroma
    - Milvus
    - Elasticsearch
只需实现 RAGBackend 子类并替换 _default_backend，Web 层与其余模块零改动。

本模块不依赖 Paddle、不依赖 inference 内部实现、不修改 PP-YOLOE 模型，
也不影响 severity.py / vlm.py 现有流程。
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path

from web.services.name_alias import crop_from_disease, normalize_crop, normalize_disease

logger = logging.getLogger("agri.web.services.rag")

_ROOT = Path(__file__).resolve().parent.parent.parent
_RAG_DB_DIR = os.environ.get("AGRI_RAG_DB_DIR") or str(_ROOT / "knowledge" / "vector_db")

# ---------------------------------------------------------------------------
# 检索缓存：容量上限 + TTL；缓存键包含 作物/病害/严重程度/Embedding模型/索引版本，
# 防止不同作物或索引版本间误用旧结果。仅缓存检索结果，不存 API Key 等敏感信息。
# ---------------------------------------------------------------------------
try:
    _RAG_CACHE_MAXSIZE = int(os.environ.get("AGRI_RAG_CACHE_MAXSIZE") or 128)
except ValueError:
    _RAG_CACHE_MAXSIZE = 128
try:
    _RAG_CACHE_TTL = float(os.environ.get("AGRI_RAG_CACHE_TTL") or 300.0)
except ValueError:
    _RAG_CACHE_TTL = 300.0


def _index_version() -> str:
    """读取知识库索引版本（built_at + total_chunks），用于缓存键，索引重建后自动失效。"""
    try:
        p = _ROOT / "knowledge" / "index_version.json"
        if p.is_file():
            d = json.loads(p.read_text(encoding="utf-8"))
            return f"{d.get('built_at', '')}:{d.get('total_chunks', '')}"
    except Exception:  # noqa: BLE001
        pass
    return ""


def _embed_model_name() -> str:
    try:
        from web.services.embedding import get_embedder

        e = get_embedder()
        return e.model if e.available else ""
    except Exception:  # noqa: BLE001
        return ""


def _cache_enabled() -> bool:
    """仅当真实向量检索可用（索引已建立）时启用缓存；索引缺失/回退 mock 时不缓存，
    避免在索引临时不可用时返回旧的「真实检索」结果。"""
    b = _default_backend
    if isinstance(b, VectorStoreRAGBackend):
        try:
            return b._available()
        except Exception:  # noqa: BLE001
            return False
    return False


class _RAGCache:
    """带 TTL 与容量上限的 LRU 缓存；记录命中/未命中与命中率。"""

    def __init__(self, maxsize: int, ttl: float) -> None:
        self._maxsize = maxsize
        self._ttl = ttl
        self._data: "OrderedDict[tuple, tuple]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key):
        now = time.time()
        if key in self._data:
            expire_ts, val = self._data[key]
            if now < expire_ts:
                self._data.move_to_end(key)
                self.hits += 1
                return val
            del self._data[key]
        self.misses += 1
        return None

    def put(self, key, val) -> None:
        now = time.time()
        self._data[key] = (now + self._ttl, val)
        self._data.move_to_end(key)
        while len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


_rag_cache = _RAGCache(_RAG_CACHE_MAXSIZE, _RAG_CACHE_TTL)


def _norm(s) -> str:
    return (s or "").strip().lower()


# ---------------------------------------------------------------------------
# 严重程度 → 防治提示
# ---------------------------------------------------------------------------
_SEVERITY_HINT = {
    "健康": "当前未检出明显病害，以预防性管理为主。",
    "轻度": "当前为轻度侵染，建议早期防治、控制扩散。",
    "中度": "当前为中度侵染，建议及时用药并配合农艺措施。",
    "重度": "当前为重度侵染，建议立即采取综合防治，必要时轮作与清园。",
}

# 健康叶类别（模型中的 3 个健康叶类，按小写英文名）
_HEALTHY_CLASSES = {"tomato leaf", "apple leaf", "grape leaf"}


# ---------------------------------------------------------------------------
# 本地知识库（mock）：key = 英文类别名（小写），value = 4 字段
# ---------------------------------------------------------------------------
_KNOWLEDGE_BASE: dict[str, dict] = {
    "tomato early blight leaf": {
        "disease_info": "由茄链格孢菌引起的真菌性病害，叶片、茎秆上形成同心轮纹状褐色病斑。",
        "control_method": "农业防治（轮作、清除病残体）结合化学防治（代森锰锌、苯醚甲环唑等）。",
        "treatment_plan": "发病初期摘除病叶，喷施保护性/治疗性杀菌剂，间隔 7-10 天，连续 2-3 次。",
        "precautions": "注意药剂轮换，避免连续使用单一药剂产生抗药性；施药避开高温时段。",
    },
    "tomato septoria leaf spot": {
        "disease_info": "由番茄壳针孢引起，叶片出现圆形灰白色病斑并散生小黑点。",
        "control_method": "清除病残体、加强通风，结合代森锰锌、百菌清等药剂防治。",
        "treatment_plan": "发病初期及时喷药，重点喷施中下部叶片，连续防治 2-3 次。",
        "precautions": "避免田间积水与过密种植，降低湿度以减少病害发生。",
    },
    "tomato leaf bacterial spot": {
        "disease_info": "由丁香假单胞菌番茄致病变种引起，叶片出现水渍状小斑点，后变褐坏死。",
        "control_method": "选用无病种子、避免喷灌，结合铜制剂或抗生素类药剂防治。",
        "treatment_plan": "发病初期喷施氢氧化铜等铜制剂，雨后及时补喷。",
        "precautions": "细菌性病害易随雨水传播，避免在潮湿天气进行农事操作。",
    },
    "tomato leaf late blight": {
        "disease_info": "由致病疫霉引起，属毁灭性病害，叶片出现水渍状暗绿色病斑，湿度大时生白色霉层。",
        "control_method": "以预防为主，结合甲霜灵·锰锌、烯酰吗啉等内吸性药剂。",
        "treatment_plan": "发现中心病株立即拔除并喷药保护，发病期缩短用药间隔至 5-7 天。",
        "precautions": "该病传播快、危害大，务必早发现早处理，防止全田蔓延。",
    },
    "tomato leaf mosaic virus": {
        "disease_info": "由烟草花叶病毒(TMV)等引起，叶片呈现花叶、斑驳、畸形，植株矮化。",
        "control_method": "以预防为主：选用抗病品种、防治蚜虫等传毒媒介、田间操作消毒。",
        "treatment_plan": "病毒病无特效药，重点做好传毒媒介防治与病株拔除，减少交叉感染。",
        "precautions": "农事操作前后对手和工具消毒，避免吸烟后接触植株。",
    },
    "tomato leaf yellow virus": {
        "disease_info": "由番茄黄化曲叶病毒(TYLCV)引起，经烟粉虱传播，叶片黄化、卷曲、植株矮缩。",
        "control_method": "重点防治烟粉虱媒介，选用抗 TYLCV 品种，使用防虫网隔离。",
        "treatment_plan": "苗期至结果期持续防治烟粉虱，发现病株及时拔除。",
        "precautions": "烟粉虱繁殖快、抗药性强，需多种药剂轮换并配合物理防治。",
    },
    "tomato mold leaf": {
        "disease_info": "由褐孢霉引起，叶片正面出现黄斑，背面生灰褐色霉层，多发生于高湿环境。",
        "control_method": "加强通风降湿，结合腐霉利、嘧菌酯等药剂防治。",
        "treatment_plan": "发病初期摘除老叶病叶，喷施治疗性杀菌剂，重点喷叶背。",
        "precautions": "控制棚内湿度是防治关键，避免大水漫灌。",
    },
    "apple scab leaf": {
        "disease_info": "由苹果黑星菌引起，叶片出现橄榄绿色至黑色霉状病斑，严重时落叶。",
        "control_method": "清除落叶、冬季清园，结合代森锰锌、戊唑醇等药剂。",
        "treatment_plan": "花前花后及雨季前各喷药 1 次，保护新叶与幼果。",
        "precautions": "落叶是主要初侵染源，务必彻底清除并销毁。",
    },
    "apple rust leaf": {
        "disease_info": "由山田胶锈菌引起，叶片出现橙黄色病斑并生毛状物，需转主寄主桧柏。",
        "control_method": "清除转主寄主（桧柏），结合三唑类药剂防治。",
        "treatment_plan": "花后展叶期喷施三唑酮、戊唑醇等，间隔 10-15 天。",
        "precautions": "果园周围避免栽植桧柏，可显著降低发病。",
    },
    "grape leaf black rot": {
        "disease_info": "由葡萄球座菌引起，叶片出现圆形褐色坏死斑，果实变黑僵化。",
        "control_method": "清除病残体、加强架面通风，结合代森锰锌、苯醚甲环唑等。",
        "treatment_plan": "开花前后及幼果期为防治关键期，及时喷药保护。",
        "precautions": "病果病叶应及时清除并深埋，减少越冬菌源。",
    },
}

_HEALTHY_ENTRY: dict = {
    "disease_info": "该样本未表现出明显病害特征，叶片外观相对健康。",
    "control_method": "以预防性管理为主，保持通风透光与合理水肥。",
    "treatment_plan": "无需用药，做好日常巡查与预防。",
    "precautions": "持续观察，发现异常及时送检确认。",
}

_CROP_CN = {"番茄": "番茄", "苹果": "苹果", "葡萄": "葡萄"}


def _generic_fallback(crop_name: str) -> dict:
    crop = _CROP_CN.get(crop_name, "")
    subject = f"{crop}作物" if crop else "该作物"
    return {
        "disease_info": f"暂未收录该病害的详细知识，建议结合{subject}田间实际进一步确认。",
        "control_method": "以农业防治为基础，结合对症药剂。",
        "treatment_plan": "咨询植保专业人员后制定防治方案。",
        "precautions": "用药前仔细阅读标签，遵守安全间隔期。",
    }


# ---------------------------------------------------------------------------
# RAG 后端抽象（未来可替换为 FAISS / Chroma / Milvus / Elasticsearch）
# ---------------------------------------------------------------------------
class RAGBackend(ABC):
    """农业知识检索后端抽象接口。"""

    backend_name: str = "base"
    status: str = "未连接"

    @abstractmethod
    def query(self, disease_name, crop_name, severity) -> dict:
        """按病害名 / 作物 / 严重程度检索知识，返回统一结果字典。

        返回：
            {
                "disease_info":   病害知识,
                "control_method": 防治方法,
                "treatment_plan": 推荐防治方案,
                "precautions":    注意事项,
            }
        """


class LocalKnowledgeBaseBackend(RAGBackend):
    """本地知识库后端（mock）：基于 Python 字典检索，不接入真实向量数据库。"""

    backend_name = "local_kb"
    status = "本地知识库（mock）"

    def query(self, disease_name, crop_name, severity) -> dict:
        return _local_kb_query(disease_name, crop_name, severity)


def _local_kb_query(disease_name, crop_name, severity) -> dict:
    key = _norm(disease_name)
    sev = severity if severity in _SEVERITY_HINT else "轻度"
    hint = _SEVERITY_HINT.get(sev, "")

    # 未检出 / 健康叶 → 健康条目
    if not key or key in _HEALTHY_CLASSES:
        entry = dict(_HEALTHY_ENTRY)
    else:
        entry = dict(_KNOWLEDGE_BASE.get(key) or _generic_fallback(crop_name))

    # 严重程度融入防治方案
    if hint:
        entry["treatment_plan"] = f"{hint} {entry.get('treatment_plan', '')}".strip()

    return entry


# ---------------------------------------------------------------------------
# 真实 RAG 后端：Chroma 向量库检索（按作物 + 病害类别隔离过滤，返回可追溯来源）
# ---------------------------------------------------------------------------
class VectorStoreRAGBackend(RAGBackend):
    """真实 RAG 后端：查询本地持久化向量库，按作物与病害类别过滤，返回来源可追溯。

    - 索引不可用 / 检索失败时，明确回退到本地演示字典（retrieval_status="mock_fallback"）。
    - 健康叶类 / 未检出时，返回无文档证据状态，不把「未检出」等同于「已确认健康」。
    """

    backend_name = "vector_store"
    status = "真实向量库检索"

    def __init__(self, db_dir: str | None = None) -> None:
        self.db_dir = db_dir or _RAG_DB_DIR
        self._collection = None
        self._load_error: str | None = None
        self._init_collection()

    def _init_collection(self) -> None:
        try:
            from web.services.vector_store import LocalVectorStore

            self._collection = LocalVectorStore(self.db_dir)
        except Exception as e:  # noqa: BLE001
            self._collection = None
            self._load_error = str(e)
            logger.warning("本地向量库加载失败：%s", e)

    def _available(self) -> bool:
        if self._collection is None:
            return False
        try:
            return self._collection.count() > 0
        except Exception:  # noqa: BLE001
            return False

    def query(self, disease_name, crop_name, severity) -> dict:
        key = _norm(disease_name)
        if not key:
            return self._no_detection_result()
        if key in _HEALTHY_CLASSES:
            return self._healthy_leaf_result(key)
        if not self._available():
            return self._mock_fallback(disease_name, crop_name, severity, "索引不可用或为空")

        try:
            from web.services.embedding import get_embedder

            embedder = get_embedder()
            if not embedder.available:
                return self._mock_fallback(disease_name, crop_name, severity, "Embedding 未配置 API Key")
            qvec = embedder.embed_query(f"{disease_name} {crop_name} {severity}")

            where: dict = {"disease_key": key}
            if crop_name in ("番茄", "苹果", "葡萄"):
                where["crop"] = crop_name

            res = self._collection.query(query_embedding=qvec, k=5, where=where)
            ids = res.get("ids", [[]])[0]
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            dists = res.get("distances", [[]])[0]
            if not docs:
                return self._no_evidence_result(disease_name, crop_name)
            return self._compose_result(ids, docs, metas, dists)
        except Exception as e:  # noqa: BLE001
            logger.error("RAG 真实检索失败，回退本地字典：%s", e)
            return self._mock_fallback(disease_name, crop_name, severity, f"检索失败：{e}")

    # ---- 各类状态结果 ----
    @staticmethod
    def _no_detection_result() -> dict:
        return {
            "disease_info": "本次检测未检出病害目标，不代表已确认健康；建议结合田间巡查与人工复核。",
            "control_method": "",
            "treatment_plan": "",
            "precautions": "未检出 ≠ 已确认健康，本结果不构成健康判定。",
            "sources": [],
            "retrieval_status": "no_detection",
        }

    @staticmethod
    def _healthy_leaf_result(key: str) -> dict:
        return {
            "disease_info": f"该类别为健康叶类（{key}），知识库无病害资料可检索。",
            "control_method": "以预防性管理为主，保持通风透光与合理水肥。",
            "treatment_plan": "无需用药，做好日常巡查。",
            "precautions": "健康叶判定仅依据当前检测结果，不等于已排除潜在病害。",
            "sources": [],
            "retrieval_status": "healthy_leaf",
        }

    @staticmethod
    def _no_evidence_result(disease_name, crop_name) -> dict:
        return {
            "disease_info": f"未检索到与「{disease_name}」（{crop_name}）相关的充分文档证据。",
            "control_method": "",
            "treatment_plan": "",
            "precautions": "知识库暂无该类别资料，请勿据此编造防治依据。",
            "sources": [],
            "retrieval_status": "no_evidence",
        }

    @staticmethod
    def _mock_fallback(disease_name, crop_name, severity, reason: str) -> dict:
        base = _local_kb_query(disease_name, crop_name, severity)
        base["sources"] = []
        base["retrieval_status"] = "mock_fallback"
        base["retrieval_note"] = f"未启用真实检索（{reason}），以下为本地演示字典内容，非权威资料。"
        return base

    @staticmethod
    def _compose_result(ids, docs, metas, dists) -> dict:
        sources = []
        for i in range(len(docs)):
            m = metas[i] or {}
            sources.append({
                "title": m.get("source_title", ""),
                "publisher": m.get("publisher", ""),
                "url": m.get("source_url", ""),
                "note_path": m.get("note_path", ""),
                "chunk_id": m.get("chunk_id", ids[i] if i < len(ids) else ""),
                "section": m.get("section", ""),
                "score": round(float(1.0 - dists[i]), 4) if i < len(dists) else None,
            })

        def section_text(section: str) -> str:
            for i in range(len(docs)):
                if (metas[i] or {}).get("section") == section:
                    return (docs[i] or "").strip()
            return ""

        parts = [p for p in [section_text("症状"), section_text("发生条件"), section_text("传播途径")] if p]
        disease_info = "\n".join(parts) if parts else "（未检索到症状描述）"
        control = section_text("综合防治") or ""
        monitor = section_text("监测") or ""

        disclaimer = (
            "以上为公开资料原文摘录（非模型总结），非确定的病原学诊断；"
            "防治信息含境外来源药剂名称，未经中国大陆登记与适用性核验，"
            "具体用药请遵当地植保部门指导与农药登记标签。"
        )
        return {
            "disease_info": disease_info,
            "control_method": control,
            "treatment_plan": control,
            "precautions": monitor,
            "sources": sources,
            "retrieval_status": "ok",
            "retrieval_note": disclaimer,
        }


# ---------------------------------------------------------------------------
# 默认后端工厂：优先真实 Chroma 检索；其内部在索引缺失时回退本地字典（明确标注）。
# ---------------------------------------------------------------------------
def _build_default_backend() -> RAGBackend:
    try:
        return VectorStoreRAGBackend()
    except Exception as e:  # noqa: BLE001
        logger.warning("VectorStoreRAGBackend 初始化失败，回退本地字典：%s", e)
        return LocalKnowledgeBaseBackend()


_default_backend: RAGBackend = _build_default_backend()


def _unknown_disease_result(original_disease, crop_name) -> dict:
    """名称无法确认时的结果：不猜测类别，明确返回无证据。"""
    return {
        "disease_info": f"无法确认病害名称「{original_disease}」，未执行检索（不猜测类别）。",
        "control_method": "",
        "treatment_plan": "",
        "precautions": "请提供标准英文类别名（如 Tomato leaf late blight）或完整中文病名（如「番茄晚疫病」）并明确作物。",
        "sources": [],
        "retrieval_status": "no_evidence",
        "retrieval_note": "名称无法确认，未猜测类别。",
        "cache_hit": False,
        "original_query": original_disease,
        "normalized_disease_name": "",
        "alias_matched": False,
        "normalized_crop": crop_name,
    }


def _attach_query_meta(result: dict, original_disease, std_disease, alias_matched, std_crop) -> dict:
    """附加查询级字段（不写入缓存，缓存仅存检索结果）。"""
    r = dict(result)
    r["original_query"] = original_disease
    r["normalized_disease_name"] = std_disease
    r["alias_matched"] = alias_matched
    r["normalized_crop"] = std_crop
    return r


def query_knowledge(disease_name, crop_name, severity) -> dict:
    """对外统一入口：检索农业知识并返回知识增强结果。

    参数：
        disease_name 病害类别名（英文标准名 / 中文病名 / 常见别名，均支持）
        crop_name    作物名（中文或英文，如 '番茄'/'tomato'）
        severity     严重程度（健康 / 轻度 / 中度 / 重度）

    返回：
        {
            "disease_info":   病害知识,
            "control_method": 防治方法,
            "treatment_plan": 推荐防治方案,
            "precautions":    注意事项,
            "sources":        来源列表（title/publisher/url/note_path/chunk_id/score，真实检索时非空）,
            "retrieval_status": ok / no_detection / healthy_leaf / no_evidence / mock_fallback,
            "retrieval_note": 检索状态或免责说明（可选）,
            "cache_hit":       是否命中缓存（True/False）,
            "retrieval_meta":  检索元信息（embed_model/index_version/latency_ms/cache_hit）,
            # 以下为向后兼容新增字段：
            "original_query":      用户原始病害名,
            "normalized_disease_name": 规范化后的标准英文类别名,
            "alias_matched":       是否经中文/别名映射命中,
            "normalized_crop":     规范化后的作物名,
        }
    """
    original_disease = disease_name
    std_disease, kind, alias_matched = normalize_disease(disease_name)

    # 名称无法确认 → 不猜测，返回无证据
    if kind == "unknown":
        return _unknown_disease_result(original_disease, crop_name)

    # 作物规范化：优先用户提供；否则从标准病害名推断；仍未知则保留原值
    std_crop = normalize_crop(crop_name) or crop_from_disease(std_disease) or crop_name

    sev_key = severity if severity in _SEVERITY_HINT else "轻度"
    cache_key = (std_crop, std_disease, sev_key, _embed_model_name(), _index_version())

    use_cache = _cache_enabled()
    if use_cache:
        cached = _rag_cache.get(cache_key)
        if cached is not None:
            t_hit = time.time()
            result = dict(cached)
            meta = dict(result.get("retrieval_meta") or {})
            meta["cache_hit"] = True
            meta["latency_ms"] = round((time.time() - t_hit) * 1000, 2)
            result["retrieval_meta"] = meta
            result["cache_hit"] = True
            return _attach_query_meta(result, original_disease, std_disease, alias_matched, std_crop)

    t0 = time.time()
    result = dict(_default_backend.query(std_disease, std_crop, severity))
    elapsed_ms = round((time.time() - t0) * 1000, 1)
    result.setdefault("sources", [])
    result.setdefault("retrieval_status", "mock")
    result["cache_hit"] = False
    result["retrieval_meta"] = {
        "embed_model": _embed_model_name(),
        "index_version": _index_version(),
        "latency_ms": elapsed_ms,
        "cache_hit": False,
    }
    # 仅缓存确定性结果；mock_fallback（索引缺失的临时状态）不缓存
    if use_cache and result.get("retrieval_status") != "mock_fallback":
        _rag_cache.put(cache_key, dict(result))
    return _attach_query_meta(result, original_disease, std_disease, alias_matched, std_crop)


def get_rag_status() -> str:
    """返回当前 RAG 后端状态（真实检索 / 索引未就绪 / 本地字典 mock）。"""
    b = _default_backend
    if isinstance(b, VectorStoreRAGBackend):
        if b._available():
            return b.status
        return "向量库检索未就绪（索引缺失，将回退 Mock）"
    return b.status


def get_rag_retrieval_info() -> dict:
    """返回 RAG 检索运行信息（供页面展示与诊断，不含密钥）。"""
    b = _default_backend
    info = {
        "backend": b.backend_name,
        "index_available": b._available() if isinstance(b, VectorStoreRAGBackend) else False,
        "embed_model": "",
        "cache_hits": _rag_cache.hits,
        "cache_misses": _rag_cache.misses,
        "cache_hit_rate": round(_rag_cache.hit_rate(), 4),
        "cache_maxsize": _RAG_CACHE_MAXSIZE,
        "cache_ttl": _RAG_CACHE_TTL,
        "index_version": _index_version(),
    }
    try:
        from web.services.embedding import get_embedder

        e = get_embedder()
        if e.available:
            info["embed_model"] = e.model
    except Exception:  # noqa: BLE001
        pass
    return info
