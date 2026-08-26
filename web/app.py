# -*- coding: utf-8 -*-
"""
app.py — 农业病害智能检测系统 Web 界面（Gradio）
=================================================
- 复用已验证的 inference 推理核心：DetectorBackend / PaddleBackend / postprocess /
  visualize / batch_infer.run_batch，**不重复实现模型加载逻辑**。
- Web 层不直接依赖 Paddle，统一通过 DetectorBackend 调用推理。
- 模型启动时只加载一次（backend.get_detector 单例），按钮点击不重复加载。
- 支持单图/多图上传、conf 阈值调节、结果可视化/对比/表格/类别统计图/批次汇总/下载/清空。
- 全部路径使用 pathlib.Path；不创建软链；不修改 PaddleDetection 源码；不访问 TEST。

启动：
    python web/app.py            # 默认 127.0.0.1:7860
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import threading
from datetime import datetime
from pathlib import Path

# 关键兼容（必须放在 import gradio 之前）：
# 绕过对 127.0.0.1 的代理拦截。部分环境（企业代理/VPN/沙箱）会拦截 httpx 对
# localhost 的请求（返回 502），导致 gradio startup-events 检查失败、事件派发异常。
# 设置 NO_PROXY 后 httpx 直连本地，gradio 干净启动、点击/检测事件正常。
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
           "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_k, None)

_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

import gradio as gr  # noqa: E402

from inference import config as infer_config  # noqa: E402
from inference.batch_infer import BatchError, run_batch  # noqa: E402
from inference.detector import DetectorError  # noqa: E402
from web import backend  # noqa: E402
from web import config as web_config  # noqa: E402

logger = logging.getLogger("agri.web.app")

# 推理串行锁：同一 DetectorBackend（同一 Paddle 模型）不支持并发推理，
# 加锁确保任意时刻只有一个检测任务在执行，避免并发拖慢/卡死。
_INFER_LOCK = threading.Lock()


def _apply_gradio_client_compat_patch() -> None:
    """gradio 4.44.1 + gradio_client 1.3.0 兼容性修补（仅运行时，不改第三方库文件）。

    背景：gradio 4.44.1 为 File/Image/Gallery 等组件生成的 schema 含
    `additionalProperties: true`（JSON Schema 中的裸布尔），而 gradio_client 1.3.0 的
    `_json_schema_to_python_type` 在递归处理时把该布尔当 dict 使用，抛出
    `TypeError: argument of type 'bool' is not iterable`，导致页面根路由 500。
    此处将 gradio_client.utils.get_type 在运行时补丁为"接受裸布尔 schema"，
    使 API 文档 schema 解析不再崩溃。若未来升级 gradio_client 已修复，此补丁自动失效。
    """
    try:
        import gradio_client.utils as _gcu

        _orig_get_type = _gcu.get_type

        def _safe_get_type(schema):
            if isinstance(schema, bool):
                return "boolean"
            return _orig_get_type(schema)

        _gcu.get_type = _safe_get_type
        logger.info("已应用 gradio_client 兼容性修补（additionalProperties: true）")
    except Exception:  # noqa: BLE001
        # 修补失败不影响程序启动（仅当上游已修复时不会走到这里）
        pass


_apply_gradio_client_compat_patch()

# 结果表格统一表头
_DET_HEADERS = ["序号", "图片", "类别id", "类别", "置信度", "x", "y", "width", "height"]
_IMG_HEADERS = ["序号", "类别", "置信度", "x", "y", "width", "height"]

# 空状态表格行（保留表头，居中提示"暂无检测目标"）
_EMPTY_IMG_ROW = [["", "暂无检测目标", "", "", "", "", ""]]
_EMPTY_COMB_ROW = [["", "", "", "暂无检测目标", "", "", "", "", ""]]

# ---------------------------------------------------------------------------
# 工具函数
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


def _upload_placeholder() -> str:
    return "📷 将作物图片拖拽到此处\n\n支持 JPG / PNG / BMP / WEBP · 支持多张上传"


def _on_upload(files):
    """上传后：填充隐藏 File 输入 + 生成图片缩略图 Gallery + 更新计数。

    只改变前端展示，推理输入仍为隐藏的 File 组件（数据流不变）。
    """
    paths = _extract_upload_paths(files)
    if not paths:
        return (
            gr.update(value=None),
            gr.update(value=[]),
            _upload_placeholder(),
        )
    n = len(paths)
    columns = 4 if n >= 4 else n
    # 自动高度：1 张较大预览，多张 300~420，避免 500px+ 大空白
    if n == 1:
        height = 280
    elif n <= 4:
        height = 320
    elif n <= 9:
        height = 380
    else:
        height = 420
    items = [(p, Path(p).name[:24]) for p in paths]  # 缩略图 + 截断文件名
    return (
        gr.update(value=paths),                      # 隐藏 File 输入（检测数据流不变）
        gr.update(value=items, columns=columns, height=height),
        f"已选择 {n} 张图片",
    )


def _load_records(run_dir: Path) -> list[dict]:
    p = run_dir / "predictions.json"
    if not p.is_file():
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_summary(run_dir: Path) -> dict:
    p = run_dir / "inference_summary.json"
    if not p.is_file():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _gallery_items(run_dir: Path) -> list[tuple[str, str]]:
    items = []
    vis_dir = run_dir / "visualized"
    det_dir = run_dir / "detections"
    if not vis_dir.is_dir():
        return items
    for jp in sorted(vis_dir.iterdir()):
        if not jp.is_file():
            continue
        n = 0
        jj = det_dir / f"{jp.stem}.json"
        if jj.is_file():
            try:
                n = json.load(open(jj, encoding="utf-8")).get("total_targets", 0)
            except Exception:  # noqa: BLE001
                n = 0
        items.append((str(jp), f"{jp.name} · {n} 个目标"))
    return items


def _combined_rows(run_dir: Path) -> list[list]:
    rows = []
    for i, r in enumerate(_load_records(run_dir), start=1):
        rows.append([
            i, r["image_name"], r["class_id"], r["class_name"],
            r["confidence"], r["x"], r["y"], r["width"], r["height"],
        ])
    return rows


def _image_rows(run_dir: Path, name: str) -> list[list]:
    jj = run_dir / "detections" / f"{Path(name).stem}.json"
    rows = []
    if jj.is_file():
        try:
            data = json.load(open(jj, encoding="utf-8"))
            for i, d in enumerate(data.get("detections", []), start=1):
                rows.append([
                    i, d["class_name"], d["confidence"],
                    d["x"], d["y"], d["width"], d["height"],
                ])
        except Exception:  # noqa: BLE001
            pass
    return rows


def _class_stats_df(run_dir: Path):
    """类别统计 DataFrame（数据来自实际推理结果 class_statistics.csv）。"""
    import pandas as pd

    summary = _load_summary(run_dir)
    stats = summary.get("class_statistics", []) or []
    if stats:
        return pd.DataFrame([{
            "class_name": s["class_name"],
            "num_images": s["num_images"],
            "num_targets": s["num_targets"],
            "avg_confidence": s["avg_confidence"],
        } for s in stats])
    return pd.DataFrame(columns=["class_name", "num_images", "num_targets", "avg_confidence"])


_CHINESE_FONT_READY = False
_CHINESE_FONT_CANDIDATES = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei",
    "Source Han Sans SC", "PingFang SC",
]


def _setup_matplotlib_font() -> None:
    """启动时自动检测系统可用中文字体并设置 matplotlib.rcParams（只执行一次）。

    修复 matplotlib 中文标题/坐标轴显示为方框乱码的问题。
    """
    global _CHINESE_FONT_READY
    if _CHINESE_FONT_READY:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.font_manager as fm

        names = {f.name for f in fm.fontManager.ttflist}
        chosen = next((c for c in _CHINESE_FONT_CANDIDATES if c in names), None)
        if chosen:
            matplotlib.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
        else:
            matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        logger.info("[matplotlib] 中文字体：%s", chosen or "未找到可用中文字体，回退 DejaVu Sans")
    except Exception as e:  # noqa: BLE001
        logger.warning("[matplotlib] 字体设置失败：%s", e)
    _CHINESE_FONT_READY = True


def _class_chart_figure(stats: list[dict]):
    """各病害类别检测数量柱状图（深色协调、中文字体、动态 Y 轴、单柱适中）。

    X=病害类别，Y=检测目标数，每类一根柱，柱顶数值；与「类别统计明细」表同一份数据。
    """
    _setup_matplotlib_font()
    if not stats:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(stats)
    # 高度 350~380px（不超过 380）
    height = 3.5 if n <= 3 else 3.8
    fig, ax = plt.subplots(figsize=(7.2, height))
    # 深色背景与当前比赛答辩 UI 协调（避免白色大块）
    bg = "#0f1a14"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.tick_params(colors="#cfe3d3")
    for sp in ax.spines.values():
        sp.set_color("#2e7d32")

    names = [str(s["class_name"]) for s in stats]
    counts = [int(s["num_targets"]) for s in stats]
    # 固定合理柱宽：单类 0.4（居中），多类 0.48/0.55（保持间距不挤压）
    width = 0.4 if n == 1 else (0.48 if n <= 4 else 0.55)
    bars = ax.bar(names, counts, width=width, color="#5aa96b", edgecolor="#3f7d52")
    # X 轴固定范围：单类别柱子始终居中，不随绘图区宽度自动拉伸
    ax.set_xlim(-0.5, max(n - 0.5, 0.5))
    ax.set_title("各病害类别检测数量", fontsize=12, color="#eafff1")
    ax.set_xlabel("病害类别", fontsize=10, color="#cfe3d3")
    ax.set_ylabel("检测目标数", fontsize=10, color="#cfe3d3")

    # 动态 Y 轴范围：max=1 -> 1.3；max>1 -> max*1.25（如 max=2 -> 2.5，max=4 -> 5）
    max_val = max(counts) if counts else 0
    ymax = 1.3 if max_val <= 1 else max_val * 1.25
    ax.set_ylim(0, ymax)

    # 柱顶数值
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, cnt + ymax * 0.03, str(cnt),
                ha="center", va="bottom", fontsize=9, color="#eafff1")

    # 轻网格（不过重）
    ax.grid(axis="y", alpha=0.2, linestyle="--", color="#3a5c46")
    ax.set_axisbelow(True)
    if n > 4:
        ax.tick_params(axis="x", rotation=30, labelsize=9)
    else:
        ax.tick_params(axis="x", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def _class_empty_initial() -> str:
    """初始（尚未检测）状态卡。"""
    return """
    <div class="empty-card">
      <div class="empty-icon">🔍</div>
      <div class="empty-title">暂无检测结果</div>
      <div class="empty-msg">尚未执行检测</div>
      <div class="empty-hint">上传图片并点击「开始 AI 检测」后将在此显示类别统计</div>
    </div>"""


def _class_empty_figure():
    """无数据时的紧凑占位图（不渲染大尺寸空 Plot，避免大片空白）。"""
    _setup_matplotlib_font()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 0.9))
    bg = "#0f1a14"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.axis("off")
    ax.text(0.5, 0.5, "暂无检测结果（图表）", ha="center", va="center",
            fontsize=11, color="#8fa896")
    fig.tight_layout()
    return fig


def _class_empty_markdown(perf: dict, records: list[dict]) -> str:
    """无检测目标时：紧凑居中的状态卡，保留真实批次状态数据（检测图片/成功失败/目标/平均置信度/FPS）。"""
    n_img = perf.get("total_images", "-")
    n_ok = perf.get("success_images", "-")
    n_fail = perf.get("failed_images", "-")
    n_targets = len(records)
    confs = [float(r["confidence"]) for r in records]
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    fps = perf.get("fps", 0.0)
    return f"""
    <div class="empty-card">
      <div class="empty-icon">🔍</div>
      <div class="empty-title">暂无检测结果</div>
      <div class="empty-msg">当前置信度阈值下未检测到目标</div>
      <div class="empty-hint">请尝试降低置信度阈值或上传其他作物/病害图片</div>
      <div class="empty-data">
        <span>检测图片 {n_img}</span><span>成功/失败 {n_ok}/{n_fail}</span>
        <span>总目标数 {n_targets}</span><span>平均置信度 {avg_conf:.3f}</span>
        <span>实际 FPS {fps:.2f}</span>
      </div>
    </div>"""


def _class_metrics_placeholder() -> str:
    return (
        "<div class='kpi-grid'>"
        "<div class='kpi-card'><div class='kpi-label'>检测类别数</div><div class='kpi-value'>-</div></div>"
        "<div class='kpi-card'><div class='kpi-label'>目标总数</div><div class='kpi-value'>-</div></div>"
        "<div class='kpi-card'><div class='kpi-label'>平均置信度</div><div class='kpi-value'>-</div></div>"
        "<div class='kpi-card'><div class='kpi-label'>最高置信度</div><div class='kpi-value'>-</div></div>"
        "</div>"
    )


def _class_metrics_markdown(stats: list[dict], records: list[dict]) -> str:
    """类别统计 Tab 顶部 4 个指标卡（数据来自本次真实推理结果）。"""
    if not stats and not records:
        return _class_metrics_placeholder()
    n_classes = len(stats)
    n_targets = sum(int(s["num_targets"]) for s in stats) if stats else len(records)
    confs = [float(r["confidence"]) for r in records]
    avg_conf = sum(confs) / len(confs) if confs else 0.0
    max_conf = max(confs) if confs else 0.0
    return f"""
    <div class="kpi-grid">
      <div class="kpi-card"><div class="kpi-label">检测类别数</div><div class="kpi-value">{n_classes}</div></div>
      <div class="kpi-card"><div class="kpi-label">目标总数</div><div class="kpi-value">{n_targets}</div></div>
      <div class="kpi-card"><div class="kpi-label">平均置信度</div><div class="kpi-value">{avg_conf:.3f}</div></div>
      <div class="kpi-card"><div class="kpi-label">最高置信度</div><div class="kpi-value">{max_conf:.3f}</div></div>
    </div>"""


def _summary_markdown(run_dir: Path) -> str:
    """批次汇总：总图片数、成功/失败、目标总数、总耗时、平均耗时、真实 FPS。"""
    summary = _load_summary(run_dir)
    perf = summary.get("performance", {})
    records = _load_records(run_dir)
    lines = ["### 批次汇总", ""]
    lines.append(f"- 总图片数：{perf.get('total_images', '-')}")
    lines.append(f"- 成功 / 失败：{perf.get('success_images', '-')} / {perf.get('failed_images', '-')}")
    lines.append(f"- 总检测目标数：{len(records)}")
    lines.append(f"- 总耗时：{perf.get('total_time', 0):.2f} s" if perf.get('total_time') else "-")
    lines.append(f"- 平均耗时：{perf.get('average_time', 0):.3f} s/张" if perf.get('average_time') else "-")
    lines.append(f"- **实际 FPS**：{perf.get('fps', 0):.2f}（真实计时）" if perf.get('fps') else "-")
    failed = summary.get("failed", []) or []
    if failed:
        lines.append("")
        lines.append("**失败明细**：")
        for f_ in failed:
            lines.append(f"- {f_['image_name']}：{f_['error']}")
    if not records and not failed:
        lines.append("")
        lines.append("> 本次推理未检出任何目标（可尝试调低置信度阈值后重测）。")
    return "\n".join(lines)


def _download_files(run_dir: Path) -> list[str]:
    files = []
    vis_dir = run_dir / "visualized"
    if vis_dir.is_dir():
        files += [str(p) for p in sorted(vis_dir.iterdir()) if p.is_file()]
    for name in ("predictions.json", "predictions.csv", "class_statistics.csv",
                 "inference_summary.json", "inference.log"):
        p = run_dir / name
        if p.is_file():
            files.append(str(p))
    return files


def _stats_markdown(run_dir: Path) -> str:
    """结果统计条：图片数/成功失败、总目标数、平均置信度、平均耗时、批次 FPS（真实）。"""
    summary = _load_summary(run_dir)
    perf = summary.get("performance", {})
    records = _load_records(run_dir)
    avg_conf = (sum(r["confidence"] for r in records) / len(records)) if records else 0.0
    return (
        f"**图片** {perf.get('total_images', '-')}（成功 {perf.get('success_images', '-')} / "
        f"失败 {perf.get('failed_images', '-')}） · **目标数** {len(records)} · "
        f"**平均置信度** {avg_conf:.3f} · **平均耗时** {perf.get('average_time', 0):.3f} s/张 · "
        f"**批次 FPS** {perf.get('fps', 0):.2f}"
    )


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


def _scene_case_info(case: dict) -> str:
    """农业病害场景详情（友好展示：作物、病害、目标数、最高置信度、耗时；隐藏 bbox 与内部文件名）。"""
    dets = case["detections"]
    n = len(dets)
    max_conf = max((d["confidence"] for d in dets), default=0.0)
    classes = sorted({d["class_name"] for d in dets})
    ms = case.get("inference_ms", 0.0)
    friendly = _friendly_case_name(case)
    lines = [
        f"### {case.get('scene_label', '')} · {friendly}",
        f"- **检测到 {n} 个目标** ｜ 最高置信度 **{max_conf:.2f}** ｜ 推理耗时 **{ms:.0f} ms**（真实计时）",
        f"- 作物类型：**{case.get('crop_cn', '未确定')}**",
    ]
    if classes:
        lines.append(f"- 病害类别：**{'、'.join(classes)}**")
    else:
        lines.append("- 病害类别：当前 conf 阈值下未检出目标")
    lines.append("")
    lines.append("> 置信度为模型原始输出，未修改；以上为 AI 视觉检测结果，不等同于专业植保诊断。")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 农业病害场景（演示数据展示，来自 web/demo_data + web/demo_outputs）
# ---------------------------------------------------------------------------
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


def _demo_select(demo_id: str):
    """选择演示案例 -> 原图 / AI检测结果 / 友好详情（隐藏 bbox 与内部文件名）。"""
    for c in DEMO_CASES:
        if c["demo_id"] == demo_id:
            orig = resolve_demo_asset_path(c, "original")
            vis = resolve_demo_asset_path(c, "result")
            if orig is None or vis is None:
                return (None, None,
                        f"（演示资源缺失：case={demo_id}，未找到原图/检测结果图，请检查 web/demo_data 与 web/demo_outputs）")
            return str(orig), str(vis), _scene_case_info(c)
    return None, None, "未找到该演示案例。"


# ---------------------------------------------------------------------------
# 农业视觉检测结果卡（基于真实模型输出，不虚构诊断）
# ---------------------------------------------------------------------------
def _crop_by_class_id(cid: int) -> str:
    """依据 13 类映射推断作物：1-8 番茄 / 9-11 苹果 / 12-13 葡萄。"""
    if 1 <= int(cid) <= 8:
        return "番茄"
    if 9 <= int(cid) <= 11:
        return "苹果"
    if 12 <= int(cid) <= 13:
        return "葡萄"
    return "未确定"


_SCENE_BY_CROP = {"番茄": "番茄病害检测", "苹果": "苹果叶部病害检测", "葡萄": "葡萄叶部病害检测"}


def _agri_result_card(records: list[dict]) -> str:
    """根据真实检测结果生成“🌱 农业视觉检测结果”卡片。

    - 作物按 13 类映射推断；多作物混合或无法判断时显示“未确定”
    - 病害类别按目标数从高到低展示
    - 仅展示真实目标数/最高/平均置信度，**不虚构严重程度/风险/经济损失/防治建议**
    """
    if not records:
        return (
            "### 🌱 农业视觉检测结果\n\n"
            "当前置信度阈值下未检测到目标。\n\n"
            "> 以上为 AI 视觉检测结果，不等同于专业植保诊断。"
        )
    from collections import Counter

    crops = {_crop_by_class_id(r["class_id"]) for r in records}
    crop = crops.pop() if len(crops) == 1 else "未确定"
    scene = _SCENE_BY_CROP.get(crop, "")
    counts = Counter(r["class_name"] for r in records)
    confs = [float(r["confidence"]) for r in records]

    lines = ["### 🌱 农业视觉检测结果", ""]
    if scene:
        lines.append(f"- 场景：**{scene}**")
    lines.append(f"- 作物类别：**{crop}**（按 13 类映射推断）")
    lines.append("- 检测到的病害类别（按目标数从高到低）：")
    for name, n in counts.most_common():
        lines.append(f"  - **{name}**：{n} 个目标")
    lines += [
        f"- 目标总数：**{len(records)}**",
        f"- 最高置信度：**{max(confs):.3f}**",
        f"- 平均置信度：**{sum(confs) / len(confs):.3f}**",
        "",
        "> 以上为 AI 视觉检测结果，不等同于专业植保诊断。",
    ]
    return "\n".join(lines)


def _edge_status_markdown() -> str:
    """⚡ 边缘部署状态卡片：从 DetectorBackend/注册表读取真实信息，不硬编码与事实冲突。"""
    from inference.detector import list_backends

    info = backend.detector_info()
    registered = list_backends()
    has_paddle_gpu = "paddle_gpu" in registered
    has_rknn = "rknn_reserved" in registered
    lines = [
        "### ⚡ 边缘部署状态",
        f"- 当前后端：**{info['backend']}**（{info['device'].upper()}）· 状态：**当前运行**",
        f"- NVIDIA GPU：接口已预留（paddle_gpu{' 已注册' if has_paddle_gpu else ''}）",
        f"- RK3588 / RKNN：接口已预留，**尚未部署**"
        f"{'（rknn_reserved 已注册占位）' if has_rknn else ''}",
        f"- 已注册后端：`{'、'.join(registered) or '无'}`",
        "- 说明：选择 rknn_reserved 仅提示“当前阶段未实现 RKNN 后端”，**不会进行模型转换**。",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 比赛演示模式（使用 web/demo_data 15 张 VAL 演示图）
# ---------------------------------------------------------------------------
# 演示顺序：番茄 → 苹果 → 葡萄 → 密集场景
_DEMO_ORDER = sorted(
    range(len(DEMO_CASES)),
    key=lambda i: (
        0 if DEMO_CASES[i]["scenario"] != "dense" else 1,   # 密集场景最后
        {"tomato": 0, "apple": 1, "grape": 2}.get(DEMO_CASES[i]["crop"], 9),
    ),
)


def _demo_case_view(index: int):
    """返回第 index 个演示案例的 (原图, AI检测结果, 信息 markdown)。

    演示模式只展示评委关心的四项：作物类型、病害类别、检测目标数、置信度/推理耗时；
    **隐藏 bbox 坐标与内部文件名**（bbox 仍保留在普通检测结果表中）。
    """
    if not DEMO_CASES:
        return None, None, "（暂无演示数据）"
    idx = index % len(_DEMO_ORDER)
    c = DEMO_CASES[_DEMO_ORDER[idx]]
    orig = resolve_demo_asset_path(c, "original")
    vis = resolve_demo_asset_path(c, "result")
    if orig is None or vis is None:
        return (None, None,
                f"（演示资源缺失：case={c['demo_id']}，未找到原图/检测结果图）")
    dets = c["detections"]
    n = len(dets)
    max_conf = max((d["confidence"] for d in dets), default=0.0)
    classes = sorted({d["class_name"] for d in dets})
    ms = c.get("inference_ms", 0.0)
    lines = [
        f"### 案例 {idx + 1} / {len(_DEMO_ORDER)} · {c['crop_cn']} · {c['scene_label']}",
        f"- **作物类型**：{c['crop_cn']}",
        f"- **病害类别**：{'、'.join(classes) if classes else '当前阈值下未检出'}",
        f"- **检测目标数**：{n}",
        f"- **最高置信度**：{max_conf:.2f} ｜ **推理耗时**：{ms:.0f} ms（真实计时）",
    ]
    return str(orig), str(vis), "\n".join(lines)


def _demo_prev(index: int):
    if not DEMO_CASES:
        return 0, None, None, "（暂无演示数据）"
    idx = (int(index) - 1) % len(_DEMO_ORDER)
    o, v, info = _demo_case_view(idx)
    return idx, o, v, info


def _demo_next(index: int):
    if not DEMO_CASES:
        return 0, None, None, "（暂无演示数据）"
    idx = (int(index) + 1) % len(_DEMO_ORDER)
    o, v, info = _demo_case_view(idx)
    return idx, o, v, info


def _demo_play():
    return True


def _demo_stop():
    return False


def _demo_timer_tick(index: int, playing: bool):
    """自动播放 tick：playing 时前进一张；停止时保持当前。"""
    if not DEMO_CASES:
        return 0, None, None, "（暂无演示数据）", False
    if playing:
        idx = (int(index) + 1) % len(_DEMO_ORDER)
        o, v, info = _demo_case_view(idx)
        return idx, o, v, info, True
    o, v, info = _demo_case_view(int(index))
    return int(index), o, v, info, False


def _demo_redetect(index: int):
    """重新检测当前演示案例：仅使用当前 VAL demo 图片；复用已加载模型，不重复加载。"""
    if not DEMO_CASES:
        return None, None, "（暂无演示数据）"
    c = DEMO_CASES[_DEMO_ORDER[int(index) % len(_DEMO_ORDER)]]
    img_path = resolve_demo_asset_path(c, "original")
    if img_path is None:
        return None, None, f"（演示资源缺失：case={c['demo_id']}，未找到原图）"
    import time as _time

    from inference.batch_infer import read_image
    from inference.postprocess import filter_by_conf
    from inference.visualize import draw_detections

    try:
        detector = backend.get_detector()
        img = read_image(img_path)
        with _INFER_LOCK:  # 串行推理
            t0 = _time.perf_counter()
            dets = detector.predict(img)
            t1 = _time.perf_counter()
            ms = (t1 - t0) * 1000.0
        dets = filter_by_conf(dets, 0.5)
        drawn = draw_detections(
            img, dets, detector.catid2name,
            catid2cn=web_config.CLASS_NAMES_CN,
            inference_seconds=t1 - t0,
        )
    except Exception as e:  # noqa: BLE001
        return str(img_path), None, f"重新检测失败：{e}"

    n = len(dets)
    max_conf = max((d.confidence for d in dets), default=0.0)
    classes = sorted({d.class_name for d in dets})
    lines = [
        f"### 🔄 重新检测 · {_friendly_case_name(c)} · {c['scene_label']}",
        f"- **作物类型**：{c['crop_cn']}",
        f"- **病害类别**：{'、'.join(classes) if classes else '当前阈值下未检出'}",
        f"- **检测目标数**：{n}",
        f"- **最高置信度**：{max_conf:.2f} ｜ **本次推理耗时**：{ms:.0f} ms（真实计时，复用已加载模型）",
    ]
    return str(img_path), drawn, "\n".join(lines)


# ---------------------------------------------------------------------------
# 事件处理
# ---------------------------------------------------------------------------
def _default_results():
    """默认（空）结果输出，与 detect 的输出顺序一致（13 项）。"""
    return (
        [],                        # 1 gallery
        gr.update(choices=[], value=None),  # 2 图片选择下拉
        None,                      # 3 原图
        None,                      # 4 检测图
        gr.update(value=_EMPTY_IMG_ROW, headers=_IMG_HEADERS),       # 5 该图片明细
        gr.update(value=_EMPTY_COMB_ROW, headers=_DET_HEADERS),       # 6 全部检测表格
        gr.update(visible=False, value=None),            # 7 类别柱状图（空状态隐藏）
        gr.update(value=[["-", "暂无类别统计数据", "-", "-", "-", "-"]]),  # 8 类别统计明细表（紧凑提示行）
        _class_empty_initial(),    # 9 类别统计区域（空状态卡）
        "**图片** - · **目标数** - · **平均置信度** - · **批次 FPS** -",  # 10 统计条
        "### 🌱 农业视觉检测结果\n\n（尚未检测）",  # 11 农业结果卡
        "### 批次汇总\n（尚未检测）",   # 12 汇总
        gr.update(value=[]),       # 13 下载
        None,                      # 14 状态
    )


def _detect(uploaded, conf):
    # 输入校验（友好提示，不崩溃）
    if not uploaded:
        gr.Warning("请先上传至少一张图片（支持 jpg/jpeg/png/bmp/webp）。")
        return _default_results()
    staged = _stage_uploads(uploaded)
    if not staged:
        gr.Warning("未找到可推理的图片，请重新上传。")
        return _default_results()

    try:
        detector = backend.get_detector()
        with _INFER_LOCK:  # 串行推理，避免并发访问同一模型
            summary = run_batch(
                detector,
                staged,
                conf=conf,
                output_root=web_config.WEB_OUTPUT_ROOT,
                recursive=False,
                limit=None,
                enable_visualization=True,
                catid2cn=web_config.CLASS_NAMES_CN,
                paddle_version=getattr(detector, "_paddle_version", ""),
            )
    except (DetectorError, BatchError) as e:
        logger.error("检测失败：%s", e)
        raise gr.Error(f"推理失败：{e}")
    except Exception as e:  # noqa: BLE001
        logger.error("检测异常：%s", e, exc_info=True)
        raise gr.Error(f"推理发生未预期异常：{e}")

    run_dir = Path(summary["output_dir"])
    gallery = _gallery_items(run_dir)
    records = _load_records(run_dir)
    names = [n for n, _ in gallery] or []
    choices = [Path(p).name for p in (sorted((run_dir / "visualized").iterdir())
                                       if (run_dir / "visualized").is_dir() else [])]
    first = choices[0] if choices else ""

    # 单图对比默认选第一张
    orig_path = str(run_dir / "originals" / first) if first else None
    vis_path = str(run_dir / "visualized" / first) if first else None
    per_rows = _image_rows(run_dir, first) if first else []

    # 类别统计（统一以本次实际推理结果为唯一状态源）
    summary_json = _load_summary(run_dir)
    perf = summary_json.get("performance", {})
    cls_stats = summary_json.get("class_statistics", []) or []
    cls_rows = []
    for s in cls_stats:
        cls_rows.append([
            s["class_id"], s["class_name"], s["num_images"], s["num_targets"],
            round(s["avg_confidence"], 3), round(s["max_confidence"], 3),
        ])
    has_data = bool(records) or bool(cls_stats)
    logger.info(
        "[类别统计状态] has_detection_results=%s | category_stats=%s | detection_results=%s",
        has_data, len(cls_stats), len(records),
    )
    if has_data:
        # 关键：必须显式 visible=True，否则图表组件（初始 visible=False）保持隐藏
        chart_upd = gr.update(visible=True, value=_class_chart_figure(cls_stats))
        class_area_val = _class_metrics_markdown(cls_stats, records)
        table_val = cls_rows
    else:
        chart_upd = gr.update(visible=False, value=_class_empty_figure())
        class_area_val = _class_empty_markdown(perf, records)
        table_val = [["-", "暂无类别统计数据", "-", "-", "-", "-"]]

    return (
        gallery,                     # 1 gallery
        gr.update(choices=choices, value=first),  # 2 dropdown
        orig_path,                   # 3 orig image
        vis_path,                    # 4 vis image
        gr.update(value=(per_rows if per_rows else _EMPTY_IMG_ROW), headers=_IMG_HEADERS),  # 5 per-image table
        gr.update(value=(_combined_rows(run_dir) or _EMPTY_COMB_ROW), headers=_DET_HEADERS),  # 6 combined table
        chart_upd,                   # 7 类别柱状图（有数据=图，无数据=紧凑占位+隐藏）
        gr.update(value=table_val),  # 8 类别统计明细表（无数据=紧凑提示行）
        class_area_val,              # 9 类别统计区域（有数据=指标卡，无数据=空状态卡）
        _stats_markdown(run_dir),    # 10 统计条（真实）
        _agri_result_card(records),  # 11 农业结果卡（真实模型输出）
        _summary_markdown(run_dir),  # 12 汇总
        gr.update(value=_download_files(run_dir)),  # 13 下载
        str(run_dir),                # 14 状态
    )


def _select_image(name, run_dir_str):
    if not name or not run_dir_str:
        return None, None, gr.update(value=[], headers=_IMG_HEADERS)
    run_dir = Path(run_dir_str)
    orig = str(run_dir / "originals" / name)
    vis = str(run_dir / "visualized" / name)
    rows = _image_rows(run_dir, name)
    return orig, vis, gr.update(value=rows, headers=_IMG_HEADERS)


def _clear():
    # 上传区（File + 缩略图 Gallery + 计数）与结果区全部重置
    return (
        None,                          # 1 隐藏 File 输入
        gr.update(value=[]),           # 2 缩略图 Gallery
        _upload_placeholder(),         # 3 上传计数/占位
    ) + _default_results()             # 4..16 结果区（13 项）


def _render_sys_info(info: dict) -> str:
    """系统信息卡：展示真实的当前推理设备、后端名与 GPU 型号（无 GPU 显示"无"，不虚构）。"""
    gpu = info.get("gpu_info") or "无"
    return (
        "### 系统信息\n"
        f"- 模型：**PP-YOLOE+-m**\n"
        f"- 输入尺寸：**{info['input_size']}×{info['input_size']}**\n"
        f"- 类别数量：**{info['num_classes']}** 类\n"
        f"- 推理设备：**{info['device'].upper()}**（实际检测）\n"
        f"- GPU 型号：**{gpu}**\n"
        f"- 推理后端：**{info['backend']}**"
    )


def _switch_device(mode: str):
    """Web 设备选择器：auto/cpu/gpu 切换后端（切换时重新加载模型一次）。"""
    old = backend.get_device_mode()
    try:
        backend.set_device_mode(mode)
        backend.get_detector()                 # 触发按新设备加载（仅一次）
        info = backend.detector_info()         # 含真实 device/backend/gpu_info
        gpu = info.get("gpu_info") or ""
        note = f"推理设备已切换：{info['backend']}（{info['device']}）" + \
               (f" · GPU：{gpu}" if gpu else "")
        return _render_sys_info(info), note
    except DetectorError as e:
        # 切换失败回滚到原模式并恢复原后端
        backend.set_device_mode(old)
        try:
            backend.get_detector()
        except DetectorError:
            pass
        raise gr.Error(str(e))


# ---------------------------------------------------------------------------
# 静态 HTML 生成器（产品化展示，不包含开发环境信息）
# ---------------------------------------------------------------------------
def _hero_html() -> str:
    v = web_config.MODEL_RECORDS
    return f"""
    <div class="hero">
      <h1>🌱 智慧农田 · AI病害智能检测平台</h1>
      <div class="hero-sub">基于 {v['model_name']} 的多作物病害视觉检测系统</div>
      <div class="hero-badges">
        <span class="badge">{v['model_name']}</span>
        <span class="badge">{v['num_classes']} 类病害检测</span>
        <span class="badge">VAL mAP@0.5:0.95 = {v['val_map']}<span class="badge-note">模型选择依据</span></span>
        <span class="badge">TEST mAP@0.5:0.95 = {v['test_map']}<span class="badge-note">独立最终评估</span></span>
      </div>
    </div>"""


def _flow_html() -> str:
    return """
    <div class="flow">
      <div class="flow-step"><div class="flow-icon">📷</div><div class="flow-text"><b>① 上传农作物图像</b></div></div>
      <div class="flow-arrow">→</div>
      <div class="flow-step"><div class="flow-icon">🤖</div><div class="flow-text"><b>② AI 视觉识别</b></div></div>
      <div class="flow-arrow">→</div>
      <div class="flow-step"><div class="flow-icon">🎯</div><div class="flow-text"><b>③ 病害定位</b></div></div>
      <div class="flow-arrow">→</div>
      <div class="flow-step"><div class="flow-icon">📊</div><div class="flow-text"><b>④ 结果统计</b></div></div>
    </div>"""


def _capabilities_html() -> str:
    return """
    <div class="cap-grid">
      <div class="cap-card"><div class="cap-icon">🌾</div>
        <div class="cap-text"><b>多作物识别</b><span>番茄 / 苹果 / 葡萄</span></div></div>
      <div class="cap-card"><div class="cap-icon">🎯</div>
        <div class="cap-text"><b>病害目标定位</b><span>13 类病害检测框</span></div></div>
      <div class="cap-card"><div class="cap-icon">📊</div>
        <div class="cap-text"><b>批量检测统计</b><span>类别 / 置信度 / FPS</span></div></div>
      <div class="cap-card"><div class="cap-icon">🔌</div>
        <div class="cap-text"><b>边缘部署预留</b><span>RK3588 / NPU 接口</span></div></div>
    </div>"""


def _crop_cards_html() -> str:
    counts: dict[str, int] = {}
    for c in DEMO_CASES:
        counts[c["crop"]] = counts.get(c["crop"], 0) + 1
    return f"""
    <div class="cap-grid">
      <div class="crop-card crop-tomato"><div class="cap-icon">🍅</div>
        <div><b>番茄病害</b><span>{counts.get('tomato', 0)} 个演示案例</span></div></div>
      <div class="crop-card crop-apple"><div class="cap-icon">🍎</div>
        <div><b>苹果叶部病害</b><span>{counts.get('apple', 0)} 个演示案例</span></div></div>
      <div class="crop-card crop-grape"><div class="cap-icon">🍇</div>
        <div><b>葡萄叶部病害</b><span>{counts.get('grape', 0)} 个演示案例</span></div></div>
    </div>"""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_ui(self_check_result: list[dict] | None = None) -> gr.Blocks:
    info = backend.detector_info()

    css = """
    .gradio-container { max-width:1440px !important; margin:0 auto !important; }
    .hero { background:linear-gradient(120deg,#0b3d2e,#14532d,#1b5e20,#2e7d32); color:#eafff1;
        border-radius:16px; padding:18px 26px 12px; text-align:center; margin:0 0 18px;
        box-shadow:0 6px 18px rgba(0,0,0,.25); }
    .hero h1 { margin:0 0 4px; font-size:30px; letter-spacing:1px; }
    .hero-sub { font-size:15px; opacity:.9; margin-bottom:8px; }
    .hero-badges .badge { display:inline-block; margin:4px 6px; padding:4px 14px; border-radius:18px;
        background:rgba(255,255,255,.14); border:1px solid rgba(255,255,255,.3); font-size:13px; color:#fff; }
    .hero-badges .badge .badge-note { display:block; font-size:11px; opacity:.75; }
    .flow { display:flex; align-items:center; justify-content:space-between;
        background:rgba(20,40,32,.55); border:1px solid rgba(76,175,80,.25); border-radius:12px;
        padding:10px 18px; margin:0 0 18px; }
    .flow-step { text-align:center; flex:1; }
    .flow-icon { font-size:24px; }
    .flow-text { font-size:13px; margin-top:4px; }
    .flow-arrow { color:#66bb6a; font-size:20px; font-weight:700; }
    /* ===== 主两栏：CSS Grid 严格对齐（顶部/底部一致） ===== */
    .app-main { display:grid !important; grid-template-columns:1.6fr 1fr !important;
        align-items:stretch !important; gap:18px !important; margin:0 0 22px !important; }
    .app-main > .column { min-height:0 !important; }
    .card { padding:18px !important; border:1px solid rgba(76,175,80,.28);
        border-radius:14px !important; background:rgba(18,38,30,.5) !important; height:100% !important; }
    .card h3 { margin-top:4px !important; }
    /* ===== 统一标题层级与垂直节奏 ===== */
    .gradio-container h3 { margin:18px 0 10px !important; }
    .gradio-container .block { margin-bottom:14px !important; }
    .stats-strip { padding:10px 16px; border:1px solid rgba(76,175,80,.35); border-radius:10px;
        background:rgba(20,50,36,.5); font-size:14px; margin:0 0 16px; }
    .section-title { font-size:18px; font-weight:700; margin:18px 0 10px; color:#81c784; }
    .sys-note { color:#a8c0ae; font-size:12px; }
    .card-note { color:#8fa896; font-size:12px; }
    /* Tab 内容统一边界 */
    .tabs { margin-top:6px; }
    .tab-wrap { padding:2px 0; }
    /* 功能切换区：按钮式 7 个等宽圆角方框 */
    .gradio-container .tablist { gap:0 !important; padding:0 !important;
        border:none !important; background:transparent !important; }
    .gradio-container .tab-button { flex:1 1 0 !important; display:flex !important;
        align-items:center !important; justify-content:center !important;
        margin:0 6px !important; padding:14px 6px !important;
        font-size:22px !important; font-weight:700 !important; color:#ffffff !important;
        background:rgba(18,38,30,.55) !important;
        border:1.5px solid rgba(76,175,80,.55) !important; border-radius:10px !important;
        cursor:pointer !important; transition:all .2s !important; }
    .gradio-container .tab-button:first-child { margin-left:0 !important; }
    .gradio-container .tab-button:last-child { margin-right:0 !important; }
    .gradio-container .tab-button:hover { background:rgba(24,52,40,.75) !important; color:#eafff1 !important; }
    .gradio-container .tab-button.active { color:#7ee99a !important; background:rgba(20,80,50,.85) !important;
        border-color:rgba(76,175,80,.95) !important; border-bottom:3px solid #4caf50 !important; }
    /* 图片统一高度策略：object-fit contain，不拉伸 */
    .gradio-container img { object-fit:contain !important; }
    .img-pair { align-items:stretch !important; }
    .img-pair > .column { height:100% !important; }
    /* 类别统计两栏：顶部对齐、自然高度（数据少时不撑高） */
    .stats-row { align-items:flex-start !important; }
    .stats-row > .column { height:auto !important; }
    /* 上传拖拽按钮（预览式上传） */
    .upload-btn { width:100% !important; min-height:56px !important;
        border:1px dashed rgba(76,175,80,.55) !important; border-radius:10px !important;
        background:rgba(20,50,36,.4) !important; font-size:14px !important; }
    /* 类别统计空状态卡（紧凑、深色、统一高度） */
    .empty-card { text-align:center; padding:16px 14px; border:1px dashed rgba(76,175,80,.4);
        border-radius:12px; background:rgba(20,40,32,.35); }
    .empty-icon { font-size:22px; line-height:1; }
    .empty-title { font-size:15px; font-weight:700; color:#b7d9bd; margin:4px 0 2px; }
    .empty-msg { font-size:13px; color:#a8c0ae; }
    .empty-hint { font-size:12px; color:#8fa896; margin:2px 0 8px; }
    .empty-data { display:flex; gap:12px; justify-content:center; flex-wrap:wrap;
        font-size:12px; color:#cfe3d3; }
    /* ===== 类别统计指标卡：一行四个紧凑卡片 ===== */
    .kpi-grid { display:grid !important; grid-template-columns:repeat(4,1fr) !important;
        gap:12px !important; margin:2px 0 14px !important; }
    .kpi-card { background:rgba(20,40,32,.6) !important; border:1px solid rgba(76,175,80,.3) !important;
        border-radius:12px !important; padding:10px 12px !important; text-align:center !important; }
    .kpi-label { color:#8fa896 !important; font-size:12px !important; margin-bottom:4px !important; }
    .kpi-value { color:#81c784 !important; font-size:24px !important; font-weight:700 !important;
        line-height:1.1 !important; }
    .kpi-sub { font-size:11px !important; color:#8fa896 !important; font-weight:400 !important; }
    @media (max-width:1100px){ .app-main{grid-template-columns:1fr !important;} .flow{flex-wrap:wrap;} .kpi-grid{grid-template-columns:repeat(2,1fr) !important;} }
    """
    with gr.Blocks(
        title="AI 农业病害智能检测系统",
        theme=gr.themes.Base(primary_hue="green", neutral_hue="slate"),
        css=css,
    ) as demo:
        # ---------- 头部（主视觉 + 4 核心徽章，无开发环境信息） ----------
        with gr.Row():
            gr.HTML(f'''
            <div class="hero">
              <h1>🌱 AI 农业病害智能检测系统</h1>
              <div class="hero-sub">智慧农业 · AI 病害智能检测平台 · 基于 PP-YOLOE+-m 的多作物病害视觉检测</div>
              <div class="hero-badges">
                <span class="badge">PP-YOLOE+-m</span>
                <span class="badge">13 类病害检测</span>
                <span class="badge">VAL mAP@0.5:0.95 = {web_config.MODEL_RECORDS['val_map']}<span class="badge-note">模型选择依据</span></span>
                <span class="badge">TEST mAP@0.5:0.95 = {web_config.MODEL_RECORDS['test_map']}<span class="badge-note">独立最终评估</span></span>
              </div>
            </div>''')

        # ---------- AI 检测流程 ----------
        gr.HTML('''
        <div class="flow">
          <div class="flow-step"><div class="flow-icon">📷</div><div class="flow-text"><b>① 上传农作物图像</b></div></div>
          <div class="flow-arrow">→</div>
          <div class="flow-step"><div class="flow-icon">🤖</div><div class="flow-text"><b>② AI 视觉识别</b></div></div>
          <div class="flow-arrow">→</div>
          <div class="flow-step"><div class="flow-icon">🎯</div><div class="flow-text"><b>③ 病害定位</b></div></div>
          <div class="flow-arrow">→</div>
          <div class="flow-step"><div class="flow-icon">📊</div><div class="flow-text"><b>④ 结果统计</b></div></div>
        </div>''')

        with gr.Row(equal_height=True, elem_classes="app-main"):
            # ============ 左列：检测操作卡（上传/阈值/按钮） ============
            with gr.Column(scale=1.6, elem_classes="card"):
                gr.Markdown("### ① 图片上传\n上传后直接预览作物图片缩略图。")
                # 隐藏 File 输入（检测数据流不变）；UploadButton + Gallery 作为主要界面
                upload = gr.Files(
                    label="上传图片（隐藏输入）",
                    file_types=["image"], type="filepath", file_count="multiple",
                    visible=False,
                )
                upload_btn = gr.UploadButton(
                    "📷 将作物图片拖拽到此处或点击选择",
                    file_types=["image"], file_count="multiple", type="filepath",
                    elem_classes="upload-btn",
                )
                upload_count = gr.Markdown(_upload_placeholder())
                preview_gallery = gr.Gallery(
                    label=None, columns=3, height=280, object_fit="contain",
                )
                conf = gr.Slider(
                    minimum=0.05, maximum=0.95, value=web_config.DEFAULT_CONF_THRESHOLD,
                    step=0.05, label="推理置信度阈值（默认 0.5）",
                    info="检测阈值，不改变模型参数。",
                )
                with gr.Row():
                    detect_btn = gr.Button("🚀 开始 AI 检测", variant="primary")
                    clear_btn = gr.Button("🗑 清空任务", variant="secondary")
                run_state = gr.State(value=None)

            # ============ 右列：系统信息卡（顶部对齐、底部随 Grid 拉伸） ============
            with gr.Column(scale=1, elem_classes="card"):
                sys_info_md = gr.Markdown(_render_sys_info(info))
                device_sel = gr.Dropdown(
                    ["auto", "cpu", "gpu"], value=backend.get_device_mode(),
                    label="推理设备（auto / cpu / gpu）",
                    info="auto=启动时自动检测；cpu=强制CPU；gpu=强制GPU（不可用报错）。切换需重新加载模型一次。",
                )
                device_note = gr.Markdown("")
                gr.Markdown(_edge_status_markdown())
                gr.Markdown(
                    "### 模型评测\n"
                    f"- VAL mAP@0.5:0.95 = **{web_config.MODEL_RECORDS['val_map']}**（模型选择依据）\n"
                    f"- TEST mAP@0.5:0.95 = **{web_config.MODEL_RECORDS['test_map']}**"
                    f"<span style='color:#ef9a9a'>（独立 TEST 最终评估结果）</span>\n"
                    f"- 参数量：**{web_config.MODEL_RECORDS['params_m']} M** · "
                    f"模型大小：**{web_config.MODEL_RECORDS['size_mb']} MB**\n"
                    f"<div class='card-note'>* 以上为固定记录，{web_config.MODEL_RECORDS['test_note']}</div>"
                )
                gr.Markdown(
                    "### 系统介绍\n"
                    "基于 **PP-YOLOE+-m** 实现 **13 类植物病害检测**：支持图片级 / 批量检测、"
                    "结果可视化与统计分析；通过统一 DetectorBackend 接口解耦推理后端，"
                    "为边缘设备（RK3588/NPU）部署预留接口。\n\n"
                    "流程：**图片上传 → AI 病害识别 → 结果可视化 → 统计分析**。"
                )

        # ---------- 结果区（全宽 Tab，保留原结构） ----------
        gr.Markdown("### ② 检测结果")
        stats_md = gr.Markdown(
            elem_classes="stats-strip",
            value="**图片** - · **目标数** - · **平均置信度** - · **批次 FPS** -",
        )
        agri_card = gr.Markdown("### 🌱 农业视觉检测结果\n\n（尚未检测）")
        with gr.Tabs():
            with gr.Tab("📈 可视化结果"):
                gallery = gr.Gallery(
                    label="检测可视化（原图比例，不拉伸）",
                    columns=3, height=360, object_fit="contain",
                )
                gr.Markdown("选择图片查看 **原图 vs 检测结果** 对比：")
                img_selector = gr.Dropdown(label="选择图片", choices=[], value=None)
                with gr.Row(equal_height=True, elem_classes="img-pair"):
                    orig_img = gr.Image(label="原图", height=320)
                    vis_img = gr.Image(label="AI 检测结果", height=320)
                with gr.Row():
                    per_img_table = gr.Dataframe(label="该图片检测明细", headers=_IMG_HEADERS, interactive=False)
            with gr.Tab("📊 检测结果表格"):
                combined_table = gr.Dataframe(label="全部检测结果", headers=_DET_HEADERS, interactive=False)
            with gr.Tab("◕ 类别统计"):
                # 统一状态源：class_area 单组件 value 切换（有数据=指标卡，无数据=空状态卡）
                class_area = gr.HTML(value=_class_empty_initial())
                with gr.Row(equal_height=False, elem_classes="stats-row"):
                    # 左：柱状图（约 55%）；右：明细表（约 45%），顶部对齐、自然高度（数据少不撑高）
                    with gr.Column(scale=1.1, min_width=0):
                        bar_plot = gr.Plot(show_label=False, visible=False)
                    with gr.Column(scale=0.9, min_width=0):
                        class_stats_table = gr.Dataframe(
                            label="类别统计明细",
                            headers=["类别id", "类别", "图片数", "目标数", "平均置信度", "最高置信度"],
                            interactive=False,
                            value=[["-", "暂无类别统计数据", "-", "-", "-", "-"]],
                        )
            with gr.Tab("📋 批次汇总"):
                summary_md = gr.Markdown("### 批次汇总\n（尚未检测）")
            with gr.Tab("⬇️ 结果下载"):
                gr.Markdown("下载当前批次的检测结果图片、JSON、CSV 与统计文件。所有统计均基于实际推理结果。")
                download_files = gr.Files(label="下载文件", interactive=False)
            with gr.Tab("🌿 农业病害场景"):
                gr.Markdown(
                    "基于 **15 张 VAL 演示案例**（番茄 8 / 苹果 4 / 葡萄 3）展示真实检测结果；"
                    "所有置信度均为模型原始输出，未人为提高。"
                )
                demo_gallery = gr.Gallery(
                    label="演示案例概览（原图比例）",
                    columns=5, height=320, object_fit="contain",
                    value=_demo_gallery_items(),
                )
                demo_choices = [
                    (f"{_friendly_case_name(c)} · {c['scene_label']}", c["demo_id"])
                    for c in DEMO_CASES
                ]
                demo_sel = gr.Dropdown(label="选择案例，查看 原图 vs 检测 与详情", choices=demo_choices, value=None)
                with gr.Row(equal_height=True, elem_classes="img-pair"):
                    demo_orig = gr.Image(label="原图", height=300)
                    demo_vis = gr.Image(label="AI 检测结果", height=300)
                demo_detail = gr.Markdown("（选择上方案例查看详情）")
            with gr.Tab("🎬 比赛演示模式"):
                _d0 = _demo_case_view(0)
                gr.Markdown(
                    "使用 **web/demo_data** 的 15 张 VAL 演示图（顺序：番茄 → 苹果 → 葡萄 → 密集场景），"
                    "复用已生成结果，模型只加载一次；重新检测仅使用当前演示图片。"
                )
                with gr.Row():
                    demo_prev_btn = gr.Button("◀ 上一张")
                    demo_next_btn = gr.Button("下一张 ▶")
                    demo_play_btn = gr.Button("▶ 自动播放")
                    demo_stop_btn = gr.Button("⏹ 停止演示")
                    demo_redetect_btn = gr.Button("🔄 重新检测当前案例")
                with gr.Row(equal_height=True, elem_classes="img-pair"):
                    demo2_orig = gr.Image(label="演示原图", value=_d0[0], height=340)
                    demo2_vis = gr.Image(label="演示检测结果", value=_d0[1], height=340)
                demo2_info = gr.Markdown(_d0[2])
                demo_index = gr.State(value=0)
                demo_playing = gr.State(value=False)
                demo_timer = gr.Timer(2.0, active=True)

        # ---------- 系统状态 / 技术信息（折叠） ----------
        with gr.Accordion("系统状态 / 技术信息", open=False):
            gr.Markdown("模型已加载 ✓ · 13 类病害 ✓ · 推理后端：Paddle CPU · RK3588/NPU：部署接口已预留")
            if self_check_result:
                lines = ["**启动自检**", ""]
                for c in self_check_result:
                    mark = "✅" if c["ok"] else "❌"
                    lines.append(f"- {mark} **{c['name']}**：{c['detail']}")
                gr.Markdown("\n".join(lines))
            gr.Markdown(
                f"**技术详情**：模型 PP-YOLOE+-m · 输入 {info['input_size']}×{info['input_size']} · "
                f"类别 {info['num_classes']} · 后端 {info['backend']}（{info['device']}） · "
                f"Paddle {info['paddle_version'] or '未知'}"
            )

        # ---------- 事件绑定 ----------
        detect_btn.click(
            _detect, inputs=[upload, conf],
            outputs=[gallery, img_selector, orig_img, vis_img, per_img_table,
                     combined_table, bar_plot, class_stats_table, class_area,
                     stats_md, agri_card, summary_md, download_files, run_state],
        )
        img_selector.change(
            _select_image, inputs=[img_selector, run_state],
            outputs=[orig_img, vis_img, per_img_table],
        )
        demo_sel.change(
            _demo_select, inputs=[demo_sel],
            outputs=[demo_orig, demo_vis, demo_detail],
        )
        demo_prev_btn.click(
            _demo_prev, inputs=[demo_index],
            outputs=[demo_index, demo2_orig, demo2_vis, demo2_info],
        )
        demo_next_btn.click(
            _demo_next, inputs=[demo_index],
            outputs=[demo_index, demo2_orig, demo2_vis, demo2_info],
        )
        demo_play_btn.click(_demo_play, outputs=[demo_playing])
        demo_stop_btn.click(_demo_stop, outputs=[demo_playing])
        demo_timer.tick(
            _demo_timer_tick, inputs=[demo_index, demo_playing],
            outputs=[demo_index, demo2_orig, demo2_vis, demo2_info, demo_playing],
        )
        demo_redetect_btn.click(
            _demo_redetect, inputs=[demo_index],
            outputs=[demo2_orig, demo2_vis, demo2_info],
        )
        device_sel.change(
            _switch_device, inputs=[device_sel],
            outputs=[sys_info_md, device_note],
        )
        upload_btn.upload(
            _on_upload, inputs=[upload_btn],
            outputs=[upload, preview_gallery, upload_count],
        )
        clear_btn.click(
            _clear,
            outputs=[upload, preview_gallery, upload_count, gallery, img_selector,
                     orig_img, vis_img, per_img_table, combined_table, bar_plot,
                     class_stats_table, class_area, stats_md, agri_card, summary_md,
                     download_files, run_state],
        )

    return demo


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------
def _launch_with_localhost_fallback(app_obj, **launch_kwargs):
    """启动 Web 服务器。

    正常环境（文件顶部已设 NO_PROXY 绕过 localhost 代理）下，gradio 的
    startup-events 预启动探测会自然通过，此处直接 launch（**不启用任何补丁**，
    保证事件派发正常）。仅当确实因 localhost 探测被拦截而启动失败时，
    才临时禁用该探测并重试（运行时补丁，不修改第三方库文件）。
    """
    try:
        app_obj.launch(**launch_kwargs)
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if not (
            "localhost is not accessible" in msg
            or "shareable link" in msg
            or "startup-events" in msg
            or "Couldn't start the app" in msg
        ):
            raise
        print("[提示] localhost 预启动探测被拦截，启用本地兼容模式重试。")
        import gradio.networking as _gnet
        import httpx as _httpx

        _orig_url_ok = _gnet.url_ok
        _orig_httpx_get = _httpx.get

        def _fake_httpx_get(url, *a, **kw):
            if "startup-events" in str(url):
                return _httpx.Response(200, request=_httpx.Request("GET", str(url)))
            return _orig_httpx_get(url, *a, **kw)

        _gnet.url_ok = lambda url: True
        _httpx.get = _fake_httpx_get
        try:
            app_obj.launch(**launch_kwargs)
        finally:
            _gnet.url_ok = _orig_url_ok
            _httpx.get = _orig_httpx_get


if __name__ == "__main__":
    import argparse

    # 公网分享：GRADIO_SHARE=true 时开启 gradio.live 公开分享链接；默认 false（本地 127.0.0.1）。
    _share = os.environ.get("GRADIO_SHARE", "false").strip().lower() in ("1", "true", "yes", "y")

    _ap = argparse.ArgumentParser(description="农业病害智能检测系统 Web")
    _ap.add_argument(
        "--device", choices=["auto", "cpu", "gpu"],
        default=os.environ.get("AGRI_DEVICE", "auto"),
        help="推理设备：auto=启动时自动检测(cpu/gpu)；cpu=强制CPU；gpu=强制GPU(不可用报错)。默认 auto。",
    )
    _ap.add_argument(
        "--port", type=int, default=int(os.environ.get("AGRI_PORT", "7860")),
        help="Web 端口，默认 7860（可用环境变量 AGRI_PORT 覆盖）。",
    )
    _args = _ap.parse_args()

    backend.set_device_mode(_args.device)   # 通过启动参数指定设备模式，无需改代码

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print("=" * 60)
    print("农业病害智能检测系统 启动自检 ...")
    print(f"设备模式：{backend.get_device_mode()} | "
          f"公网分享：{'开启（gradio.live）' if _share else '关闭（本地 127.0.0.1 访问）'}")
    checks = backend.self_check()   # 触发模型加载（仅一次）
    for c in checks:
        print(f"  [{'OK' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")
    # 实际推理设备（真实，不虚构）
    _info = backend.detector_info()
    _gpu = _info.get("gpu_info") or ""
    print(f"实际推理设备：{_info.get('device', '?').upper()}（后端 {_info.get('backend', '?')}）"
          + (f" · GPU：{_gpu}" if _gpu and _gpu != "无" else ""))
    print("=" * 60)
    if _share:
        print("[提示] 已开启公网分享模式：Gradio 将生成公开 gradio.live 分享链接，")
        print("        其他人在浏览器打开该链接即可直接使用当前电脑上的模型进行推理。")
        print("        关闭分享：退出进程后以 GRADIO_SHARE=false 重启（默认）。")
    print(f"Local URL: http://127.0.0.1:{_args.port}")
    if _share:
        print("Share URL: 由 Gradio 生成（创建隧道后输出 Running on public URL，此处不虚构）")
    sys.stdout.flush()
    app = build_ui(checks)
    app.queue()
    _launch_with_localhost_fallback(
        app,
        server_name="127.0.0.1",
        server_port=_args.port,
        share=_share,
        show_error=True,
        prevent_thread_lock=False,
    )
