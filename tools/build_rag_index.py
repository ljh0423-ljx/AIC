# -*- coding: utf-8 -*-
"""
tools/build_rag_index.py — 农业 RAG 知识库向量化建库工具
========================================================
功能：
  - 解析 knowledge/notes/*.md 病害笔记，按「症状/发生条件/传播途径/监测/综合防治」分块；
  - 从 knowledge/sources_manifest.json 读取来源元数据并绑定到每个 chunk；
  - 用 DashScope text-embedding-v3 向量化，写入本地持久化向量库（knowledge/vector_db/）；
  - 支持增量更新（按 chunk_id 去重）、重复文档检测、来源 metadata 绑定、索引版本记录。

说明：本机 Chroma 原生扩展在写入向量时崩溃（Segfault），故采用纯 numpy 本地向量库
（web/services/vector_store.py）作为可配置替代方案，检索接口对齐 Chroma 子集。

用法：
  python tools/build_rag_index.py                 # 增量构建
  python tools/build_rag_index.py --reset         # 重建（清空重导）
  python tools/build_rag_index.py --notes knowledge/notes --db knowledge/vector_db

环境变量（密钥不写入源码）：
  DASHSCOPE_API_KEY   Embedding 密钥（复用百炼）
  AGRI_RAG_EMBED_MODEL 默认 text-embedding-v3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag.build")

KNOWLEDGE_SECTIONS = ("症状", "发生条件", "传播途径", "监测", "综合防治")


# ---------------------------------------------------------------------------
# 笔记解析
# ---------------------------------------------------------------------------
def _strip_md(line: str) -> str:
    return re.sub(r"^\s*[-*]\s+", "", line).strip()


def parse_note(path: Path) -> dict:
    """解析单份笔记，返回 {title_cn, title_en, category_id, crop, pathogen, sections}。"""
    text = path.read_text(encoding="utf-8")
    title_cn = title_en = ""
    category_id = None
    crop = "未确定"
    pathogen = ""
    sections: dict[str, str] = {}
    current = ""

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("# "):
            t = line[2:].strip()
            if "/" in t:
                title_cn, title_en = (p.strip() for p in t.split("/", 1))
            else:
                title_cn = title_en = t
            continue
        if line.startswith("> "):
            continue
        m = re.search(r"类别ID[：:]\s*(\d+)", line)
        if m:
            category_id = int(m.group(1))
            mc = re.search(r"（(番茄|苹果|葡萄)·", line)
            if mc:
                crop = mc.group(1)
            continue
        m = re.search(r"病原[：:]\s*(.+)", line)
        if m:
            pathogen = _strip_md(m.group(1)).strip()
            continue
        if line.startswith("## "):
            sec = line[3:].strip()
            if "来源" in sec:
                current = ""
            else:
                current = sec
                sections.setdefault(sec, "")
            continue
        if current in KNOWLEDGE_SECTIONS:
            sections[current] += _strip_md(line) + "\n"

    return {
        "title_cn": title_cn,
        "title_en": title_en,
        "category_id": category_id,
        "crop": crop,
        "pathogen": pathogen,
        "sections": {k: v.strip() for k, v in sections.items() if v.strip()},
    }


def load_manifest_meta(manifest_path: Path) -> dict[str, list[dict]]:
    """读取 manifest，按 note 文件路径归组来源（标题/机构/URL）。"""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_note: dict[str, list[dict]] = {}
    for s in data.get("sources", []):
        lp = s.get("local_path", "")
        by_note.setdefault(lp, []).append({
            "url": s.get("url", ""),
            "title": s.get("title", ""),
            "publisher": s.get("publisher", ""),
        })
    return by_note


def build_chunks(notes_dir: Path, manifest_path: Path) -> list[dict]:
    """解析全部笔记并生成 chunk 列表（含 metadata）。"""
    by_note = load_manifest_meta(manifest_path)
    chunks: list[dict] = []
    for np_ in sorted(notes_dir.glob("*.md")):
        rel = np_.relative_to(ROOT).as_posix()
        note = parse_note(np_)
        sources = by_note.get(rel, [])
        primary = sources[0] if sources else {}
        all_urls = " | ".join(s["url"] for s in sources if s["url"])
        all_titles = " | ".join(s["title"] for s in sources if s["title"])
        all_pub = " | ".join(s["publisher"] for s in sources if s["publisher"])

        disease_key = (note["title_en"] or "").strip().lower()
        base_meta = {
            "category_id": note["category_id"] if note["category_id"] is not None else -1,
            "category_name": note["title_en"] or np_.stem,
            "crop": note["crop"],
            "disease_key": disease_key,
            "pathogen": note["pathogen"],
            "note_path": rel,
            "source_url": primary.get("url", ""),
            "source_title": all_titles,
            "publisher": all_pub,
            "source_urls": all_urls,
        }
        for section in KNOWLEDGE_SECTIONS:
            content = note["sections"].get(section)
            if not content:
                continue
            chunk_text = f"{note['title_cn']}｜{section}：{content}"
            cid = "rag_" + hashlib.sha1(f"{rel}::{section}".encode("utf-8")).hexdigest()[:20]
            meta = dict(base_meta)
            meta["section"] = section
            meta["chunk_id"] = cid
            chunks.append({"id": cid, "text": chunk_text, "meta": meta})
    return chunks


# ---------------------------------------------------------------------------
# 建库
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="农业 RAG 知识库向量化建库")
    ap.add_argument("--notes", default=str(ROOT / "knowledge" / "notes"))
    ap.add_argument("--manifest", default=str(ROOT / "knowledge" / "sources_manifest.json"))
    ap.add_argument("--db", default=str(ROOT / "knowledge" / "vector_db"))
    ap.add_argument("--reset", action="store_true", help="清空向量库后重建")
    args = ap.parse_args()

    notes_dir = Path(args.notes)
    manifest_path = Path(args.manifest)
    db_dir = Path(args.db)

    if not notes_dir.is_dir():
        logger.error("笔记目录不存在：%s", notes_dir)
        return 1
    if not manifest_path.is_file():
        logger.error("manifest 不存在：%s", manifest_path)
        return 1

    from web.services.embedding import get_embedder
    from web.services.vector_store import LocalVectorStore

    embedder = get_embedder()
    if not embedder.available:
        logger.error("Embedding 未配置 API Key，无法建库（DASHSCOPE_API_KEY / AGRI_RAG_EMBED_API_KEY）")
        return 1

    chunks = build_chunks(notes_dir, manifest_path)
    logger.info("解析得到 %d 个 chunk", len(chunks))

    if args.reset:
        import shutil
        if db_dir.is_dir():
            shutil.rmtree(db_dir)
        logger.info("已清空向量库目录")

    store = LocalVectorStore(str(db_dir))
    existing = store.get_ids()
    new_chunks = [c for c in chunks if c["id"] not in existing]
    logger.info("已有 %d 个，新增 %d 个", len(existing), len(new_chunks))

    if new_chunks:
        ids = [c["id"] for c in new_chunks]
        texts = [c["text"] for c in new_chunks]
        metas = [c["meta"] for c in new_chunks]
        embeddings = embedder.embed_texts(texts)
        store.add(ids=ids, embeddings=embeddings, metadatas=metas, documents=texts)
        logger.info("已写入 %d 个 chunk（embedding 维度 %d）", len(new_chunks), len(embeddings[0]))

    total = store.count()
    version = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_chunks": total,
        "new_chunks": len(new_chunks),
        "embed_model": embedder.model,
        "embed_backend": embedder.backend_name,
        "embed_dim": embedder.dim,
        "db_dir": db_dir.as_posix(),
        "store_backend": "numpy_local",
        "notes_dir": notes_dir.as_posix(),
    }
    ver_path = ROOT / "knowledge" / "index_version.json"
    ver_path.write_text(json.dumps(version, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("索引版本已写入 %s：总 chunk=%d，模型=%s", ver_path, total, embedder.model)
    print(json.dumps(version, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
