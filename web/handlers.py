# -*- coding: utf-8 -*-
"""
handlers.py — 事件处理与结果回读渲染
======================================
- 检测（_detect）/ 选图对比（_select_image）/ 清空（_clear）/ 设备切换（_switch_device）
- 结果回读：从 run_dir 读取 predictions.json / inference_summary.json 等产物
- 结果派生的 markdown（批次汇总 / 统计条）

推理统一通过 backend.get_detector() 单例调用，模型只加载一次；
同一 DetectorBackend 不支持并发推理，用 _INFER_LOCK 串行（demo 重检测共用此锁）。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import gradio as gr

from inference.batch_infer import BatchError, run_batch
from inference.detector import DetectorError
from web import backend
from web import config as web_config
from web.assets import _extract_upload_paths, _stage_uploads
from web.charts import _class_chart_figure, _class_empty_figure
from web.render import (
    _agent_card,
    _agri_result_card,
    _class_empty_initial,
    _class_empty_markdown,
    _class_metrics_markdown,
    _crop_by_class_id,
    _DET_HEADERS,
    _EMPTY_COMB_ROW,
    _EMPTY_IMG_ROW,
    _IMG_HEADERS,
    _rag_card,
    _render_sys_info,
    _robot_card,
    _upload_placeholder,
    _vlm_card,
)
from web.services import agent
from web.services import camera
from web.services import flowchart
from web.services import rag
from web.services import report
from web.services import robot
from web.services import speech
from web.services import vlm
from web.services.name_alias import crop_from_disease, extract_disease_name, normalize_crop
from web.services.severity import calculate_severity

logger = logging.getLogger("agri.web.handlers")

# 推理串行锁：同一 DetectorBackend（同一 Paddle 模型）不支持并发推理，
# 加锁确保任意时刻只有一个检测任务在执行，避免并发拖慢/卡死。
_INFER_LOCK = threading.Lock()

# 会话级检测上下文已改为 Gradio gr.State（见 ui.py 的 session_state），
# 不再用模块级全局保存图片/诊断结果，避免不同浏览器会话串用上下文。


# ---------------------------------------------------------------------------
# 结果回读
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 结果派生 markdown（依赖回读，故留在 handlers 而非 html，避免循环导入）
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 病害严重程度评估（阶段1 规则评估，不修改模型/推理）
# ---------------------------------------------------------------------------
def _compute_batch_severity(run_dir: Path, records: list[dict]) -> list[dict]:
    """按图片分组计算病害严重程度评估（阶段1 规则评估，不修改模型/推理）。

    从 originals/ 读取每张原图尺寸，从 records 按 image_name 分组后调用
    web.services.severity.calculate_severity；读取尺寸失败则跳过该图（不影响检测流程）。
    """
    if not records:
        return []
    from PIL import Image

    by_image: dict[str, list[dict]] = {}
    for r in records:
        by_image.setdefault(r.get("image_name", ""), []).append(r)

    assessments: list[dict] = []
    for name, dets in by_image.items():
        if not name:
            continue
        orig = run_dir / "originals" / name
        try:
            with Image.open(orig) as im:
                w, h = im.size
        except Exception:  # noqa: BLE001
            continue
        a = dict(calculate_severity(dets, w, h))
        a["image_name"] = name
        assessments.append(a)
    return assessments


def _compute_vlm_diagnosis(run_dir: Path, records: list[dict], assessments: list[dict], first: str) -> dict:
    """调用 VLM 后端生成视觉语言诊断（阶段2，不影响检测/严重程度流程）。

    关键：只把「当前展示图片（first）」这一张图的检测结果交给 VLM，避免把整批
    记录误当作某一张图片的视觉依据；严重程度也取对应图片的评估结果。
    """
    image_path = str(run_dir / "originals" / first) if first else None
    img_records = [r for r in records if r.get("image_name") == first] if first else []
    severity = next((a for a in assessments if a.get("image_name") == first), {})
    return vlm.analyze_image(image_path, img_records, severity)


def _compute_rag_knowledge(records: list[dict], assessments: list[dict]) -> dict:
    """调用 RAG 后端检索农业知识（阶段3 mock，不影响检测/严重程度/VLM 流程）。"""
    severity = assessments[0] if assessments else {}
    disease_name = severity.get("class_name", "")
    crop_name = "未确定"
    if records:
        crop_name = _crop_by_class_id(records[0].get("class_id", 0))
    sev = severity.get("severity", "轻度")
    return rag.query_knowledge(disease_name, crop_name, sev)


# ---------------------------------------------------------------------------
# 事件处理
# ---------------------------------------------------------------------------
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
        _vlm_card(None),           # 12 AI 视觉诊断（VLM 阶段2）
        _rag_card(None),           # 13 农业知识辅助诊断（RAG 阶段3）
        _agent_card(None),         # 14 智能决策建议（Agent 阶段4）
        "### 批次汇总\n（尚未检测）",   # 15 汇总
        gr.update(value=[]),       # 16 下载
        None,                      # 17 状态
    )


def _detect(uploaded, conf, session_state=None):
    # 输入校验（友好提示，不崩溃）
    if not uploaded:
        gr.Warning("请先上传至少一张图片（支持 jpg/jpeg/png/bmp/webp）。")
        return _default_results() + (session_state, "（尚未生成报告）", None, None, None, None)
    staged = _stage_uploads(uploaded)
    if not staged:
        gr.Warning("未找到可推理的图片，请重新上传。")
        return _default_results() + (session_state, "（尚未生成报告）", None, None, None, None)

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

    # 病害严重程度评估（阶段1 规则评估，附加展示，不改变既有检测流程）
    assessments = _compute_batch_severity(run_dir, records)
    # VLM 视觉语言诊断（阶段2 mock，附加展示，不改变既有检测流程）
    vlm_analysis = _compute_vlm_diagnosis(run_dir, records, assessments, first)
    vlm_status = vlm.get_vlm_status()
    # RAG 农业知识增强（阶段3 mock，附加展示，不改变既有检测流程）
    rag_knowledge = _compute_rag_knowledge(records, assessments)
    rag_status = rag.get_rag_status()
    # 农业智能决策 Agent（阶段4 mock，附加展示，不改变既有检测流程）
    agent_decision = agent.make_decision(
        records, assessments[0] if assessments else {}, vlm_analysis, rag_knowledge,
        vlm_status=vlm_status, rag_status=rag_status,
    )
    agent_status = agent.get_agent_status()
    new_session_state = {
        "image_name": first,
        "orig_path": orig_path,
        "vis_path": vis_path,
        "detection_result": records,
        "severity_result": assessments[0] if assessments else {},
        "vlm_result": vlm_analysis,
        "rag_result": rag_knowledge,
        "agent_decision": agent_decision,
        "timestamp": time.time(),
        "voice_done": False,
        "report_done": False,
    }

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
        _agri_result_card(records, assessments),  # 11 农业结果卡（真实模型输出 + 严重程度评估）
        _vlm_card(vlm_analysis, vlm_status),      # 12 AI 视觉诊断（VLM 阶段2）
        _rag_card(rag_knowledge, rag_status),     # 13 农业知识辅助诊断（RAG 阶段3）
        _agent_card(agent_decision, agent_status),  # 14 智能决策建议（Agent 阶段4）
        _summary_markdown(run_dir),  # 15 汇总
        gr.update(value=_download_files(run_dir)),  # 16 下载
        str(run_dir),                # 17 状态
        new_session_state,           # 18 会话上下文（gr.State）
        "（尚未生成报告）",            # 19 报告预览重置
        None,                        # 20 报告 PDF 重置
        None,                        # 21 流程图预览重置
        None,                        # 22 流程图 PNG 重置
        None,                        # 23 流程图 PDF 重置
    )


def _camera_detect(camera_frame, conf, session_state=None):
    """摄像头拍照检测：复用 _detect 流程（PP-YOLOE → severity → VLM → RAG → Agent）。"""
    frame_path = camera.save_frame(camera_frame)
    if frame_path is None:
        gr.Warning("请先打开摄像头并拍照，再点击检测。")
        return _default_results() + (None, "（尚未生成报告）", None, None, None, None)
    return _detect([str(frame_path)], conf, session_state)


def _select_image(name, run_dir_str):
    if not name or not run_dir_str:
        return None, None, gr.update(value=[], headers=_IMG_HEADERS)
    run_dir = Path(run_dir_str)
    orig = str(run_dir / "originals" / name)
    vis = str(run_dir / "visualized" / name)
    rows = _image_rows(run_dir, name)
    return orig, vis, gr.update(value=rows, headers=_IMG_HEADERS)


def _clear(session_state=None):
    # 上传区（File + 缩略图 Gallery + 计数）与结果区全部重置；仅清空当前会话上下文
    return (
        None,                          # 1 隐藏 File 输入
        gr.update(value=[]),           # 2 缩略图 Gallery
        _upload_placeholder(),         # 3 上传计数/占位
    ) + _default_results() + (None, "（尚未生成报告）", None, None, None, None)   # 结果区 + 会话上下文 + 报告/流程图重置


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


def _agent_status_note() -> str:
    """Agent 运行状态提示（供语音/文本回答，确保 TTS 播报准确展示真实 Agent 或 Mock 回退）。"""
    st = (agent.get_agent_status() or "").lower()
    if "mock" in st or "规则决策" in st:
        return "（提示：当前为 Mock 规则决策，不等同于专业植保决策）"
    return "（由真实 LLM 智能体综合分析，仅供人工复核参考）"


def _format_agent_answer(decision: dict | None) -> str:
    """把 Agent 决策组织成自然语言回答（供 TTS 合成与展示）。"""
    if not decision:
        return "请先上传图片并完成一次检测，再进行语音咨询。"
    return (
        f"根据智能诊断，{decision.get('task', '')}，"
        f"优先级为{decision.get('priority', '')}，建议{decision.get('action', '')}。"
        f"目标区域：{decision.get('target', '')}。"
        f"决策依据：{decision.get('reason', '')}"
        f"{_agent_status_note()}"
    )


def _voice_answer(asr_text: str, decision: dict | None) -> str:
    """根据用户语音问题 + Agent 决策生成针对性回答（语音驱动 Agent 闭环，mock 意图理解）。"""
    if not decision:
        return "请先上传图片并完成一次检测，再进行语音咨询。"
    text = asr_text or ""
    if "严重" in text:
        focus = "您关心的是病害严重程度"
    elif "怎么处理" in text or "防治" in text or "处理" in text:
        focus = "您关心的是处理建议"
    elif "什么病" in text or "病害" in text or "是什么" in text:
        focus = "您关心的是病害类型"
    else:
        focus = ""

    base = (
        f"根据智能诊断，{decision.get('task', '')}，"
        f"优先级为{decision.get('priority', '')}，建议{decision.get('action', '')}。"
        f"目标区域：{decision.get('target', '')}。"
        f"决策依据：{decision.get('reason', '')}"
        f"{_agent_status_note()}"
    )
    return f"{focus}。{base}" if focus else base


# 判定「振幅接近 0（麦克风未采集到声音）」的阈值
_MIN_AUDIO_AMPLITUDE = 1e-3


def _debug_audio(audio, context: str) -> bool:
    """打印音频调试信息（是否为空/类型/shape/采样率/长度/最大振幅），返回音频是否有效。"""
    import numpy as np

    if audio is None:
        logger.warning("[语音调试][%s] audio 为空(None) —— 麦克风未采集到数据", context)
        return False

    logger.info("[语音调试][%s] audio 类型 = %s", context, type(audio).__name__)

    if isinstance(audio, (tuple, list)) and len(audio) == 2:
        sr, samples = audio
        arr = np.asarray(samples)
        logger.info(
            "[语音调试][%s] numpy 元组 | 采样率=%s | shape=%s | len=%s | dtype=%s",
            context, sr, arr.shape, arr.size, arr.dtype,
        )
        if arr.size == 0:
            logger.warning("[语音调试][%s] 样本长度为 0 —— 无音频数据", context)
            return False
        max_amp = float(np.abs(arr).max())
        logger.info("[语音调试][%s] 最大振幅 max(abs(samples)) = %.6f", context, max_amp)
        if max_amp < _MIN_AUDIO_AMPLITUDE:
            logger.warning(
                "[语音调试][%s] 振幅接近 0（%.6f < %.0e）—— 麦克风没有采集声音",
                context, max_amp, _MIN_AUDIO_AMPLITUDE,
            )
            return False
        return True

    if isinstance(audio, str):
        logger.info("[语音调试][%s] 文件路径=%s | 存在=%s", context, audio, Path(audio).exists())
        return bool(audio) and Path(audio).exists()

    logger.warning("[语音调试][%s] audio 类型异常：%s", context, type(audio).__name__)
    return False


def _voice_empty_result(reason: str):
    """返回语音交互「空」结果（7 项），reason 写入麦克风状态。"""
    return (
        f"麦克风状态：{reason}",
        "识别文本：无",
        "AI回答：未收到有效语音，请先录音再咨询。",
        f"ASR状态：{speech.get_asr_status()}",
        f"TTS状态：{speech.get_tts_status()}",
        "语音输出状态：未生成音频",
        None,
    )


def _on_audio_record_stop(audio):
    """录音结束：打印调试信息，并把本次录音保存到会话级 recorded_audio（供「开始语音咨询」按钮读取）。"""
    _debug_audio(audio, "录音结束")
    return audio


def _on_audio_record_start():
    """开始新录音：清空上一次缓存的 recorded_audio，避免本次录音失败后误用旧音频。"""
    logger.info("[语音调试][录音开始] 清空 recorded_audio 缓存")
    return None


def _on_audio_upload(audio):
    """上传音频：保存到会话级 recorded_audio，与麦克风录音走同一缓存，保证上传音频也能咨询。"""
    _debug_audio(audio, "上传")
    return audio


def _on_audio_clear():
    """音频组件被清空/移除：同步清空 recorded_audio 缓存。"""
    logger.info("[语音调试][音频清空] 清空 recorded_audio 缓存")
    return None


_IMAGE_QUESTION_KEYWORDS = ("这张图片", "这张图", "刚才检测", "刚才的", "这张叶子", "图里", "图片", "叶子", "叶片")


def _sources_display(sources) -> str:
    seen = set()
    parts = []
    for s in (sources or [])[:3]:
        url = s.get("url", "")
        pub = s.get("publisher", "") or s.get("title", "") or s.get("note_path", "")
        if url and url not in seen:
            seen.add(url)
            parts.append(f"[{pub}]({url})")
    return " · ".join(parts) if parts else ""


def _sources_tts(sources) -> str:
    seen = set()
    pubs = []
    for s in (sources or [])[:3]:
        pub = s.get("publisher", "") or s.get("title", "")
        if pub and pub not in seen:
            seen.add(pub)
            pubs.append(pub)
    return ("；来源：" + "、".join(pubs)) if pubs else ""


def _voice_qa_answer(text: str, session_state=None) -> dict:
    """语音问答：解析用户问题 → 分支处理（病害知识 / 当前图片 / 需补充信息）。
    返回 {"tts": 播报文本（不含 URL）, "display": 展示文本（markdown 链接）, "sources": 来源列表}。"""
    text = (text or "").strip()
    if not text:
        return {"tts": "未检测到语音内容，请重新录音再咨询。",
                "display": "未检测到语音内容，请重新录音再咨询。", "sources": []}

    std_name, kind, alias_matched, matched = extract_disease_name(text)
    is_image_q = any(k in text for k in _IMAGE_QUESTION_KEYWORDS)

    # 1) 明确病害知识问题（优先级最高）→ 真实 RAG + Agent
    if kind == "disease":
        crop = normalize_crop("") or crop_from_disease(std_name) or "番茄"
        disease_arg = matched or std_name
        rag_result = rag.query_knowledge(disease_arg, crop, "中度")
        status = rag_result.get("retrieval_status")
        if status == "ok":
            qa = agent.answer_question(text, rag_result)
            answer = qa.get("answer") or ""
            sources = qa.get("evidence_sources") or rag_result.get("sources") or []
            src_display = _sources_display(sources)
            src_tts = _sources_tts(sources)
            note = _agent_status_note()
            display = answer + (f"\n来源：{src_display}" if src_display else "") + note
            tts = answer + src_tts + note
            return {"tts": tts, "display": display, "sources": sources}
        if status == "no_evidence":
            return {"tts": "抱歉，知识库暂未检索到该病害的充分资料，无法给出可靠回答。",
                    "display": "抱歉，知识库暂未检索到该病害的充分资料，无法给出可靠回答。", "sources": []}
        return {"tts": f"当前检索不可用（{status}），请稍后再试。",
                "display": f"当前检索不可用（{status}），请稍后再试。", "sources": []}

    if kind == "healthy":
        msg = "该类别为健康叶，知识库无病害资料；健康叶判定不等于已排除潜在病害。"
        return {"tts": msg, "display": msg, "sources": []}

    # 2) 当前图片问题 → 仅复用本次会话严格对应的结果（gr.State）
    if is_image_q:
        ctx = session_state
        if ctx and ctx.get("agent_decision"):
            msg = _format_agent_answer(ctx["agent_decision"])
            return {"tts": msg, "display": msg,
                    "sources": (ctx.get("rag_result") or {}).get("sources", [])}
        msg = "请先上传或拍摄一张作物图片并完成检测，我再针对当前图片回答。"
        return {"tts": msg, "display": msg, "sources": []}

    # 3) 未指定病害、也非图片问题
    msg = "请说明具体病害（如「番茄晚疫病怎么防治」）或先上传图片完成检测。"
    return {"tts": msg, "display": msg, "sources": []}


def _voice_stream_interact(audio_stream, session_state=None):
    """实时麦克风语音交互：麦克风输入 → 真实 ASR → 问题解析 → RAG/Agent → TTS。

    ASR 无结果/失败时如实报告，不返回固定 mock 文本冒充识别结果。
    session_state 为本浏览器会话的 gr.State（会话级隔离）；识别成功时标记 voice_done。
    """
    # 调试 + 有效性检查：为空或振幅接近 0 时直接返回，不进入 ASR
    if not _debug_audio(audio_stream, "按钮点击"):
        return _voice_empty_result("麦克风未采集到有效语音") + (session_state,)

    logger.info("[语音调试] audio 正常，进入 ASR 流程")
    asr_result = speech.microphone_to_text(audio_stream)
    text = (asr_result.get("text") or "").strip()

    if not text:
        return _voice_empty_result("未检测到语音（ASR 无结果）") + (session_state,)

    qa = _voice_qa_answer(text, session_state)
    tts_result = speech.text_to_audio(qa["tts"])
    audio_path = tts_result.get("audio_path", "")

    new_ss = dict(session_state) if isinstance(session_state, dict) else {}
    new_ss["voice_done"] = True

    mic_status = "麦克风状态：识别完成"
    text_line = f"识别文本：{text}"
    answer_line = f"AI回答：{qa['display']}"
    asr_line = f"ASR状态：{speech.get_asr_status()}"
    tts_line = f"TTS状态：{speech.get_tts_status()}"
    output_status = (
        f"语音输出状态：已合成音频（{Path(audio_path).suffix or '?'}）：{Path(audio_path).name}"
        if audio_path else "语音输出状态：TTS 未生成音频"
    )
    return mic_status, text_line, answer_line, asr_line, tts_line, output_status, audio_path, new_ss


def _robot_execute(session_state=None):
    """机械臂执行：复用当前会话 Agent 决策生成执行指令（阶段6.1 mock）。"""
    decision = (session_state or {}).get("agent_decision") if isinstance(session_state, dict) else None
    if decision:
        task = decision.get("task", "")
        action = decision.get("action", "")
        target = decision.get("target", "")
    else:
        task = action = target = ""
    result = robot.execute_task(task, action, target)
    return _robot_card(result, robot.get_robot_status(), task)


def _generate_report(session_state=None):
    """生成 AI 诊断验收报告：复用当前会话已有结果，不调用付费 API。
    返回 (预览 markdown, PDF 路径, 更新后的 session_state)。无有效会话上下文时提示先检测。"""
    if not isinstance(session_state, dict) or not session_state.get("agent_decision"):
        return "### 📄 AI 诊断验收报告\n\n（请先上传图片并完成一次检测）", None, session_state
    preview, pdf_path = report.generate_report(session_state)
    new_ss = dict(session_state)
    new_ss["report_done"] = True
    return preview, pdf_path, new_ss


def _generate_flowchart(session_state=None):
    """生成诊断流程图：复用当前会话结果，不调用付费 API。
    返回 (PNG 预览, PNG 下载, PDF 下载)。无有效会话时返回空。"""
    if not isinstance(session_state, dict) or not session_state.get("agent_decision"):
        return None, None, None
    png, pdf = flowchart.generate_flowchart(session_state)
    return png, png, pdf
