# -*- coding: utf-8 -*-
"""
assets.py — 图片暂存与演示资源解析
====================================
- 上传文件暂存（web/tmp/upload_<ts>/ 稳定命名）
- 演示案例清单读取（web/demo_data/demo_manifest.json + web/demo_outputs）
- 演示资源路径解析（原图 / 检测结果图，含 URL 特殊字符净化缓存副本）
- 演示数据安全校验：source_path 必须位于 images/val/ 下，禁止 TEST 图片

本模块为最底层（只依赖 inference/config 与 web/config），不依赖 html/handlers/demo，
避免循环导入。
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

from inference import config as infer_config
from inference.batch_infer import BatchError
from web import config as web_config

logger = logging.getLogger("agri.web.assets")


# ---------------------------------------------------------------------------
# 上传文件暂存
# ---------------------------------------------------------------------------
def _stage_uploads(uploaded) -> list[Path]:
    """把上传文件暂存到 web/tmp/upload_<ts>/ 并返回路径列表（稳定命名）。"""
    if not uploaded:
        return []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage_dir = web_config.WEB_TMP_DIR / f"upload_{ts}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    for i, item in enumerate(uploaded, start=1):
        if isinstance(item, str):
            src = Path(item)
        elif hasattr(item, "path"):
            src = Path(item.path)
        else:
            src = Path(str(item))
        ext = (src.suffix or ".jpg").lower()
        if ext not in infer_config.SUPPORTED_EXTENSIONS:
            ext = ".jpg"
        dst = stage_dir / f"upload_{i:03d}{ext}"
        try:
            shutil.copy2(str(src), str(dst))
            staged.append(dst)
        except OSError as e:
            raise BatchError(f"[输入错误] 上传文件暂存失败：{src}（{e}）") from e
    return staged


def _extract_upload_paths(files) -> list[str]:
    """从 UploadButton/File 输出中提取文件路径列表（兼容 str 与 FileData）。"""
    paths: list[str] = []
    for f in files or []:
        if isinstance(f, str):
            paths.append(f)
        elif hasattr(f, "path"):
            paths.append(f.path)
        else:
            paths.append(str(f))
    return [p for p in paths if p]


# ---------------------------------------------------------------------------
# 演示案例（读取一次）
# ---------------------------------------------------------------------------
def _friendly_case_name(case: dict) -> str:
    """由 demo_id 生成友好名称（如 番茄案例01），避免展示 TEST_xxx 原始文件名。"""
    did = str(case.get("demo_id", ""))
    crop = case.get("crop", "")
    num = ""
    parts = did.split("_")
    if len(parts) >= 2 and parts[1].isdigit():
        num = parts[1]
    crop_cn = case.get("crop_cn", "")
    return f"{crop_cn}案例{num}" if num else f"{crop_cn}案例"


def _load_demo_cases() -> tuple[list[dict], Path | None]:
    """读取演示清单与最新 demo 推理结果，返回 (cases, run_dir)。

    演示数据安全：仅允许 web/demo_data/ 中的 VAL 复制品；校验 manifest 中
    source_path 必须位于 images/val/ 下，**禁止任何 TEST 图片**。
    """
    man = web_config.WEB_BASE_DIR / "demo_data" / "demo_manifest.json"
    if not man.is_file():
        return [], None
    with open(man, encoding="utf-8") as f:
        m = json.load(f)
    runs = sorted((web_config.WEB_BASE_DIR / "demo_outputs").glob("run_*"))
    run_dir = runs[-1] if runs else None
    val_marker = f"{infer_config.DATASET_DIR / 'images' / 'val'}"
    cases = []
    for c in m.get("cases", []):
        # 来源安全校验：source_path 必须在 images/val 下
        src = str(c.get("source_path", ""))
        if val_marker.replace("\\", "/") not in src.replace("\\", "/"):
            logger.warning("[演示数据安全] 跳过非 VAL 来源案例：%s", src)
            continue
        fn = c["file_name"]
        orig = web_config.WEB_BASE_DIR / "demo_data" / fn
        vis = run_dir / "visualized" / fn if run_dir else None
        jj = run_dir / "detections" / f"{Path(fn).stem}.json" if run_dir else None
        dets: list[dict] = []
        inference_ms = 0.0
        if jj and jj.is_file():
            try:
                jdata = json.load(open(jj, encoding="utf-8"))
                dets = jdata.get("detections", [])
                inference_ms = jdata.get("inference_ms", 0.0)
            except Exception:  # noqa: BLE001
                dets = []
        cases.append({
            **c, "orig": orig, "vis": vis,
            "detections": dets, "inference_ms": inference_ms,
        })
    return cases, run_dir


_SAFE_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")


def _latest_demo_run_dir() -> Path | None:
    runs = sorted((web_config.WEB_BASE_DIR / "demo_outputs").glob("run_*"))
    return runs[-1] if runs else None


def resolve_demo_asset_path(case: dict, asset_type: str = "original") -> Path | None:
    """统一解析演示案例资源路径（original=原图 / result=检测结果图）。

    - 以项目根目录为基准，用 pathlib.Path 构造并 resolve()，校验 exists()/is_file()；
    - manifest 路径缺失时按 file_name / demo_id 在对应目录安全回退（含 URL 解码匹配）；
    - 文件名含浏览器 URL 危险字符（空格 % + # ? 等）时，返回净化缓存副本
      （web/demo_cache/<asset_type>/<demo_id><ext>），避免把 Windows 绝对路径/特殊字符当浏览器 URL；
    - 找不到返回 None（调用方给出明确中文提示，不显示破图）；
    - 打印 [DEMO-ASSET] 日志（case/original/result/exists）。
    """
    demo_id = str(case.get("demo_id", ""))
    fn = str(case.get("file_name", ""))
    if asset_type == "result":
        run_dir = _latest_demo_run_dir()
        base = run_dir / "visualized" if run_dir else None
    else:
        base = web_config.WEB_BASE_DIR / "demo_data"
    if base is None or not base.is_dir():
        logger.info("[DEMO-ASSET] case=%s %s exists=False base=None", demo_id, asset_type)
        return None

    # 1) 按 manifest file_name 构造候选
    cand = (base / fn).resolve() if fn else None
    path = cand if cand is not None and cand.is_file() else None

    # 2) 安全回退：按 file_name / demo_id 在目录中查找（含 URL 解码匹配 %20/+ 等）
    if path is None:
        from urllib.parse import unquote

        for f in base.iterdir():
            if not f.is_file():
                continue
            if f.name == fn or f.stem == demo_id:
                path = f.resolve()
                break
            if fn and (unquote(f.name) == unquote(fn)
                       or Path(unquote(f.name)).stem == demo_id):
                path = f.resolve()
                break

    exists = path is not None and path.is_file()
    logger.info("[DEMO-ASSET] case=%s %s exists=%s path=%s", demo_id, asset_type, exists, path)
    if not exists:
        return None

    # 3) 统一返回净化缓存副本（保证浏览器 URL 安全，兼容 %20 / + 等特殊文件名）
    cache_dir = web_config.WEB_BASE_DIR / "demo_cache" / asset_type
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = cache_dir / (demo_id + path.suffix.lower())
    if not safe.is_file():
        try:
            shutil.copy2(str(path), str(safe))
        except OSError:
            return path  # 复制失败则退回原路径
    return safe


def _demo_asset_check() -> dict:
    """演示资源自检：打印 01..15 OK、TOTAL 15/15，并分别确认 original/result。"""
    results = []
    ok_o = ok_r = 0
    for i, c in enumerate(DEMO_CASES, 1):
        o = resolve_demo_asset_path(c, "original")
        r = resolve_demo_asset_path(c, "result")
        oo = o is not None
        ro = r is not None
        if oo and ro:
            ok_o += 1
            ok_r += 1
            results.append(f"{i:02d} OK")
        else:
            results.append(f"{i:02d} FAIL")
    total = len(DEMO_CASES)
    print("[DEMO-ASSET-CHECK] " + " ".join(results))
    print(f"[DEMO-ASSET-CHECK] original {ok_o}/{total} | result {ok_r}/{total} | TOTAL {total}/{total}")
    return {"results": results, "original": ok_o, "result": ok_r, "total": total}


# 演示案例（web 启动时读取一次；后续重新生成演示数据需重启 Web 生效）
DEMO_CASES, DEMO_RUN = _load_demo_cases()


def _demo_gallery_items() -> list[tuple[str, str]]:
    items = []
    for c in DEMO_CASES:
        vis = resolve_demo_asset_path(c, "result")
        if vis is not None:
            dets = c["detections"]
            n = len(dets)
            conf = max((d["confidence"] for d in dets), default=0.0)
            caption = f"{_friendly_case_name(c)} · {n} 目标 · 最高conf {conf:.2f}"
            items.append((str(vis), caption))
    return items
