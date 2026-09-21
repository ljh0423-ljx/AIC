# -*- coding: utf-8 -*-
"""
web/services/vector_store.py — 本地持久化向量存储（纯 numpy 实现）
================================================================
背景：本机 Windows + Python 3.12 上，chromadb 1.5.9（Rust 后端）在写入向量时
Segmentation fault；chromadb 0.5.x 需从源码编译 chroma-hnswlib（无预编译 wheel，构建失败）。
故采用纯 numpy 本地向量存储作为「可配置替代方案」，API 对齐 chromadb 常用子集
（add / query / count / get_ids），便于未来 chromadb 稳定后无缝替换。

持久化格式（目录下）：
    embeddings.npy    float32 [N, dim]
    metadata.json     [{"id":..., "meta":{...}, "document":...}, ...]

检索：余弦相似度，支持 metadata 等值过滤（where），返回 ids/documents/metadatas/distances。
本模块不依赖 Paddle、不依赖 inference、无任何原生扩展，纯 Python+numpy，稳定可复现。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class LocalVectorStore:
    """本地持久化向量存储（numpy + JSON，余弦相似度检索）。"""

    def __init__(self, dir_path: str) -> None:
        self.dir = Path(dir_path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._emb_path = self.dir / "embeddings.npy"
        self._meta_path = self.dir / "metadata.json"
        self._ids: list[str] = []
        self._metas: list[dict] = []
        self._documents: list[str] = []
        self._embeddings: np.ndarray | None = None
        self._load()

    def _load(self) -> None:
        if self._emb_path.is_file() and self._meta_path.is_file():
            self._embeddings = np.load(self._emb_path).astype(np.float32)
            data = json.loads(self._meta_path.read_text(encoding="utf-8"))
            self._ids = [d["id"] for d in data]
            self._metas = [d["meta"] for d in data]
            self._documents = [d["document"] for d in data]
        else:
            self._embeddings = None
            self._ids, self._metas, self._documents = [], [], []

    def _save(self) -> None:
        np.save(self._emb_path, np.asarray(self._embeddings, dtype=np.float32))
        data = [{"id": i, "meta": m, "document": d}
                for i, m, d in zip(self._ids, self._metas, self._documents)]
        self._meta_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def count(self) -> int:
        return len(self._ids)

    def get_ids(self) -> set[str]:
        return set(self._ids)

    def add(self, ids: list[str], embeddings: list[list[float]],
            metadatas: list[dict], documents: list[str]) -> None:
        arr = np.asarray(embeddings, dtype=np.float32)
        if self._embeddings is None:
            self._embeddings = arr
        else:
            self._embeddings = np.vstack([self._embeddings, arr])
        self._ids.extend(ids)
        self._metas.extend(metadatas)
        self._documents.extend(documents)
        self._save()

    def _match(self, where: dict | None, meta: dict) -> bool:
        if not where:
            return True
        for k, v in where.items():
            if meta.get(k) != v:
                return False
        return True

    def query(self, query_embedding: list[float], k: int = 5, where: dict | None = None) -> dict:
        """余弦相似度 top-k 检索；返回与 chromadb 一致的嵌套结构。"""
        if self._embeddings is None or not self._ids:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        norms = np.linalg.norm(self._embeddings, axis=1)
        qnorm = float(np.linalg.norm(q)) or 1.0
        sims = (self._embeddings @ q) / (norms * qnorm + 1e-9)

        scored: list[int] = []
        for i in range(len(self._ids)):
            if self._match(where, self._metas[i]):
                scored.append(i)
        scored.sort(key=lambda i: -float(sims[i]))
        top = scored[:k]

        ids = [self._ids[i] for i in top]
        docs = [self._documents[i] for i in top]
        metas = [self._metas[i] for i in top]
        dists = [float(1.0 - sims[i]) for i in top]
        return {"ids": [ids], "documents": [docs], "metadatas": [metas], "distances": [dists]}
