# -*- coding: utf-8 -*-
"""
web/services/report.py — AI 诊断验收报告生成
============================================
复用当前会话已有检测/VLM/RAG/Agent 结果生成验收报告（Markdown 预览 + PDF），
不重新执行检测、不调用任何付费 API。PDF 使用 reportlab + SimHei 中文字体。

报告为「系统运行验收」记录，明确不等同于病害专业确诊；未检出 ≠ 确认健康。
PDF 存储于 web/tmp/reports/（会话隔离的临时目录，UUID 命名，带 TTL 清理）。
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

_REPORT_DIR = Path(__file__).resolve().parent.parent / "tmp" / "reports"
_FONT_PATH = "C:/Windows/Fonts/simhei.ttf"
_REPORT_TTL_HOURS = 24.0


def _crop_of(class_name: str) -> str:
    n = (class_name or "").lower()
    if n.startswith("tomato"):
        return "番茄"
    if n.startswith("apple"):
        return "苹果"
    if n.startswith("grape"):
        return "葡萄"
    return "未确定"


def cleanup_old_reports() -> None:
    """删除超过 TTL 的报告 PDF，避免临时文件无限增长。"""
    try:
        if not _REPORT_DIR.is_dir():
            return
        cutoff = time.time() - _REPORT_TTL_HOURS * 3600
        for p in _REPORT_DIR.glob("*.pdf"):
            if p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


def _build_content(session_state: dict) -> dict:
    sev = session_state.get("severity_result") or {}
    vlm_r = session_state.get("vlm_result") or {}
    rag_r = session_state.get("rag_result") or {}
    agent_d = session_state.get("agent_decision") or {}

    disease = sev.get("class_name") or ""
    crop = _crop_of(disease)
    no_detection = (not disease) or (sev.get("severity") == "健康")

    return {
        "report_id": f"AGRI-{time.strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}",
        "gen_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "crop": crop,
        "disease": "" if no_detection else disease,
        "no_detection": no_detection,
        "confidence": float(sev.get("confidence", 0.0)),
        "ratio": float(sev.get("infection_ratio", 0.0)),
        "severity": sev.get("severity", ""),
        "bbox": sev.get("bbox", ""),
        "visual": vlm_r.get("visual_evidence", "") or vlm_r.get("disease_description", ""),
        "sources": rag_r.get("sources", []) or [],
        "rag_status": rag_r.get("retrieval_status", "未知"),
        "rag_cache": bool(rag_r.get("cache_hit", False)),
        "vlm_status": vlm_r.get("status", "未连接"),
        "vlm_model": vlm_r.get("model", ""),
        "agent_status": agent_d.get("agent_status", ""),
        "agent_model": agent_d.get("model_name", ""),
        "action": agent_d.get("action", ""),
        "reason": agent_d.get("reason", ""),
        "orig_path": session_state.get("orig_path"),
        "vis_path": session_state.get("vis_path"),
        "image_name": session_state.get("image_name", ""),
    }


def build_preview(c: dict) -> str:
    disease_line = "未检出病害目标（不等于确认健康）" if c["no_detection"] else f"{c['disease']}（{c['crop']}）"
    lines = [
        "### 📄 AI 诊断验收报告",
        "",
        f"- **编号**：{c['report_id']}",
        f"- **生成时间**：{c['gen_time']}",
        f"- **作物**：{c['crop']}",
        f"- **病害类别**：{disease_line}",
        f"- **检测置信度**：{c['confidence']:.3f}",
        f"- **检测框面积代理指标**：{c['ratio']*100:.1f}%（严重程度 {c['severity']}，检测框 {c['bbox']}）",
        f"- **视觉证据**：{c['visual'] or '（无）'}",
    ]
    if c["sources"]:
        lines.append("- **知识来源**：")
        for s in c["sources"][:5]:
            pub = s.get("publisher", "") or s.get("title", "")
            url = s.get("url", "")
            lines.append(f"  - [{pub}]({url})")
    else:
        lines.append("- **知识来源**：（无）")
    lines += [
        f"- **人工复核建议**：{c['action'] or '（无）'}",
        f"- **系统运行状态**：VLM={c['vlm_status']}（{c['vlm_model']}）· RAG={c['rag_status']}（{'缓存复用' if c['rag_cache'] else '实际检索'}）· Agent={c['agent_status']}（{c['agent_model']}）",
        "",
        "> 本报告为**系统运行验收**记录，复用当前会话已有检测结果，**不等同于病害专业确诊**；未检出病害目标 ≠ 已确认健康；检测框面积占比为规则代理指标，不认定真实感染面积。",
    ]
    return "\n".join(lines)


def _build_pdf(c: dict) -> str:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    if "SimHei" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("SimHei", _FONT_PATH))

    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = _REPORT_DIR / f"{c['report_id']}.pdf"

    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                            leftMargin=18*mm, rightMargin=18*mm, topMargin=15*mm, bottomMargin=15*mm)
    title = ParagraphStyle("title", fontName="SimHei", fontSize=16, leading=22, spaceAfter=8)
    h = ParagraphStyle("h", fontName="SimHei", fontSize=12, leading=16, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("body", fontName="SimHei", fontSize=10, leading=16, spaceAfter=4)
    small = ParagraphStyle("small", fontName="SimHei", fontSize=8.5, leading=12, textColor=colors.HexColor("#666666"))

    disease_line = "未检出病害目标（不等于确认健康）" if c["no_detection"] else f"{c['disease']}（{c['crop']}）"

    story = [
        Paragraph("智农慧眼 · AI 诊断验收报告", title),
        Paragraph(f"编号：{c['report_id']}　生成时间：{c['gen_time']}", small),
        Spacer(1, 6),
    ]

    rows = [
        ["作物", c["crop"], "病害类别", disease_line],
        ["检测置信度", f"{c['confidence']:.3f}", "严重程度", c["severity"] or "—"],
        ["检测框面积代理指标", f"{c['ratio']*100:.1f}%", "检测框", c["bbox"] or "—"],
    ]
    tbl = Table(rows, colWidths=[32*mm, 58*mm, 32*mm, 58*mm])
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "SimHei"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f2f2f2")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#f2f2f2")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 10))

    # 视觉证据 / 人工复核建议 / 决策依据（长文本自动换页）
    story.append(Paragraph("一、视觉证据", h))
    story.append(Paragraph(c["visual"] or "（无）", body))
    story.append(Paragraph("二、人工复核建议", h))
    story.append(Paragraph(c["action"] or "（无）", body))
    story.append(Paragraph("三、决策依据", h))
    story.append(Paragraph(c["reason"] or "（无）", body))

    # 图片嵌入
    story.append(Paragraph("四、图像记录", h))
    img_count = 0
    for label, path in (("原图", c["orig_path"]), ("检测结果图", c["vis_path"])):
        if path and Path(path).is_file():
            try:
                img = Image(str(path), width=150*mm, height=110*mm, kind="proportional")
                story.append(Paragraph(label, small))
                story.append(img)
                story.append(Spacer(1, 4))
                img_count += 1
            except Exception:  # noqa: BLE001
                pass
    if img_count == 0:
        story.append(Paragraph("（无图像记录）", body))

    # 知识来源（可点击链接）
    story.append(Paragraph("五、知识来源", h))
    if c["sources"]:
        for s in c["sources"][:6]:
            pub = s.get("publisher", "") or s.get("title", "")
            url = s.get("url", "")
            if url:
                story.append(Paragraph(f'· {pub}：<link href="{url}">{url}</link>', body))
            else:
                story.append(Paragraph(f"· {pub}", body))
    else:
        story.append(Paragraph("（无来源）", body))

    # 系统运行状态
    story.append(Paragraph("六、系统运行状态", h))
    story.append(Paragraph(
        f"VLM：{c['vlm_status']}（{c['vlm_model']}）<br/>"
        f"RAG：{c['rag_status']}（{'缓存复用' if c['rag_cache'] else '实际检索'}）<br/>"
        f"Agent：{c['agent_status']}（{c['agent_model']}）", body))

    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "声明：本报告为系统运行验收记录，复用当前会话已有检测结果，不等同于病害专业确诊；"
        "未检出病害目标 ≠ 已确认健康；检测框面积占比为规则代理指标，不认定真实感染面积。", small))

    doc.build(story)
    return str(pdf_path)


def generate_report(session_state: dict) -> tuple[str, str]:
    """生成验收报告，返回 (markdown 预览, PDF 路径)。复用会话上下文，不调用任何 API。"""
    cleanup_old_reports()
    c = _build_content(session_state)
    preview = build_preview(c)
    pdf_path = _build_pdf(c)
    return preview, pdf_path
