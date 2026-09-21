# -*- coding: utf-8 -*-
"""
web/services/embedding.py — 文本向量化（Embedding）模块
========================================================
供真实 RAG 建库（tools/build_rag_index.py）与检索（web/services/rag.py）共用。

当前后端（默认）：DashScope 文本 Embedding（OpenAI 兼容接口）
    - 模型：text-embedding-v3（默认，1024 维），可用 AGRI_RAG_EMBED_MODEL 覆盖
    - 密钥：复用 DASHSCOPE_API_KEY（或 AGRI_RAG_EMBED_API_KEY），不写入源码
    - 地址：AGRI_RAG_EMBED_BASE_URL（默认 DashScope 兼容域名）

可配置替代方案（本地轻量模型，当前环境不可用）：
    本机 torch=2.2.0+cpu，而最新 transformers 5.x 要求 torch>=2.5、sentence-transformers 6.x
    依赖 transformers 5.x；且 sentence-transformers 3.x 又要求 huggingface-hub<1.0，与 gradio 6
    冲突。因此本地 sentence-transformers 方案在本环境不可用，已回退为 DashScope API（见下）。
    若未来升级 torch>=2.5，可新增 LocalSentenceTransformerEmbedder 并切换
    AGRI_RAG_EMBED_BACKEND=local，签名保持 embed_texts / embed_query 不变。

本模块不依赖 Paddle、不依赖 inference 内部实现、不修改 PP-YOLOE 模型。
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("agri.web.services.embedding")

_DEFAULT_MODEL = "text-embedding-v3"
_DEFAULT_DIM = 1024
_DEFAULT_BATCH = 10
_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class DashScopeEmbedder:
    """DashScope 文本向量化后端（OpenAI 兼容接口）。"""

    backend_name = "dashscope"

    def __init__(self, api_key: str = "", model: str = "", base_url: str = "",
                 dim: int = 0, batch: int = 0) -> None:
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY") \
            or os.environ.get("AGRI_RAG_EMBED_API_KEY") or ""
        self.model = model or os.environ.get("AGRI_RAG_EMBED_MODEL") or _DEFAULT_MODEL
        self.base_url = base_url or os.environ.get("AGRI_RAG_EMBED_BASE_URL") or _DEFAULT_BASE_URL
        try:
            self.dim = int(dim or os.environ.get("AGRI_RAG_EMBED_DIM") or _DEFAULT_DIM)
        except ValueError:
            self.dim = _DEFAULT_DIM
        try:
            self.batch = int(batch or os.environ.get("AGRI_RAG_EMBED_BATCH") or _DEFAULT_BATCH)
        except ValueError:
            self.batch = _DEFAULT_BATCH
        try:
            self.max_retries = int(os.environ.get("AGRI_API_MAX_RETRIES") or 2)
        except ValueError:
            self.max_retries = 2
        self.available = bool(self.api_key)
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                                  timeout=60.0, max_retries=self.max_retries)
        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.available:
            raise RuntimeError("未配置 Embedding API Key（DASHSCOPE_API_KEY / AGRI_RAG_EMBED_API_KEY）")
        if not texts:
            return []
        client = self._ensure_client()
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch):
            batch = list(texts[i:i + self.batch])
            resp = client.embeddings.create(model=self.model, input=batch)
            out.extend([list(d.embedding) for d in resp.data])
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


_embedder: DashScopeEmbedder | None = None


def get_embedder() -> DashScopeEmbedder:
    """返回共享 Embedder 单例。"""
    global _embedder
    if _embedder is None:
        _embedder = DashScopeEmbedder()
        if _embedder.available:
            logger.info("RAG Embedding 已就绪：%s（model=%s, dim=%d）",
                        _embedder.backend_name, _embedder.model, _embedder.dim)
        else:
            logger.warning("RAG Embedding 未配置 API Key，真实检索不可用")
    return _embedder
