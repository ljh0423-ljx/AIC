# -*- coding: utf-8 -*-
"""
render.py — 静态 HTML / CSS 与 markdown 卡片生成
================================================
- 页面 CSS、Hero/流程图等静态 HTML
- 检测结果相关的 markdown 卡片（农业结果卡、类别统计指标卡、空状态卡、系统信息卡等）
- 检测结果表格统一表头 / 空状态行等展示常量

本模块只做「数据 -> 展示文本」的纯渲染，不读文件、不触发推理；
依赖 assets（DEMO_CASES / _friendly_case_name）与 backend（系统信息），不依赖 handlers，
避免循环导入。
"""

from __future__ import annotations

import base64
from pathlib import Path

from web import backend
from web import config as web_config
from web.assets import DEMO_CASES, _friendly_case_name

# ---------------------------------------------------------------------------
# 结果表格统一表头与空状态行
# ---------------------------------------------------------------------------
_DET_HEADERS = ["序号", "图片", "类别id", "类别", "置信度", "x", "y", "width", "height"]
_IMG_HEADERS = ["序号", "类别", "置信度", "x", "y", "width", "height"]

# 空状态表格行（保留表头，居中提示"暂无检测目标"）
_EMPTY_IMG_ROW = [["", "暂无检测目标", "", "", "", "", ""]]
_EMPTY_COMB_ROW = [["", "", "", "暂无检测目标", "", "", "", "", ""]]


def _upload_placeholder() -> str:
    return "📷 将作物图片拖拽到此处\n\n支持 JPG / PNG / BMP / WEBP · 支持多张上传"


# ---------------------------------------------------------------------------
# 页面静态 HTML（live，被 build_ui 直接使用）
# ---------------------------------------------------------------------------
def hero_html() -> str:
    v = web_config.MODEL_RECORDS
    return f"""
    <div class="hero">
      <h1>🌱 AI 农业病害智能检测系统</h1>
      <div class="hero-sub">智慧农业 · AI 病害智能检测平台 · 基于 PP-YOLOE+-m 的多作物病害视觉检测</div>
      <div class="hero-badges">
        <span class="badge">PP-YOLOE+-m</span>
        <span class="badge">13 类病害检测</span>
        <span class="badge">VAL mAP@0.5:0.95 = {v['val_map']}<span class="badge-note">模型选择依据</span></span>
        <span class="badge">TEST mAP@0.5:0.95 = {v['test_map']}<span class="badge-note">独立最终评估</span></span>
      </div>
    </div>"""


def flow_html() -> str:
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


# ---------------------------------------------------------------------------
# 历史遗留的静态 HTML 生成器（当前未被 build_ui 调用，仅为与原 app.py 完全一致而保留，
# 无任何行为影响）
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
# 类别统计空状态 / 指标卡
# ---------------------------------------------------------------------------
def _class_empty_initial() -> str:
    """初始（尚未检测）状态卡。"""
    return """
    <div class="empty-card">
      <div class="empty-icon">🔍</div>
      <div class="empty-title">暂无检测结果</div>
      <div class="empty-msg">尚未执行检测</div>
      <div class="empty-hint">上传图片并点击「开始 AI 检测」后将在此显示类别统计</div>
    </div>"""


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


def _severity_section(assessments: list[dict]) -> str:
    """病害严重程度评估区（基于检测框面积占比的规则评估，阶段1）。"""
    lines = ["### 🌡️ 病害严重程度评估（规则评估）", ""]
    for i, a in enumerate(assessments, start=1):
        name = a.get("image_name") or f"图片{i}"
        cls = a.get("class_name") or "未检出"
        conf = float(a.get("confidence", 0.0))
        ratio = float(a.get("infection_ratio", 0.0))
        sev = a.get("severity", "轻度")
        rec = a.get("recommendation", "")
        lines.append(f"**{name}**")
        lines.append(f"- 病害类别：{cls}")
        lines.append(f"- 置信度：{conf:.3f}")
        lines.append(f"- 病害区域占比（检测框代理）：{ratio * 100:.1f}%")
        lines.append(f"- 严重程度：**{sev}**")
        lines.append(f"- 建议：{rec}")
        lines.append("")
    lines.append(
        "> 严重程度为基于病害区域占比（检测框代理）的规则评估（阶段1），不等同于专业植保诊断；"
        "后续可替换为病斑分割 / 深度学习分类模型。"
    )
    return "\n".join(lines)


def _vlm_card(analysis: dict | None, status: str = "") -> str:
    """AI 视觉诊断卡（VLM 阶段2：真实 Qwen-VL 或 mock 回退）。"""
    if not analysis:
        return "### 🤖 AI 视觉诊断\n\n（尚未检测）"
    model = analysis.get("model") or ""
    lines = [
        "### 🤖 AI 视觉诊断",
        "",
        f"- 病害描述：{analysis.get('disease_description', '')}",
        f"- 视觉依据：{analysis.get('visual_evidence', '')}",
        f"- 风险分析：{analysis.get('severity_analysis', '')}",
        f"- 处理建议：{analysis.get('recommendation', '')}",
    ]
    if model:
        lines.append(f"- 视觉模型：**{model}**")
    lines += [
        f"- 状态：{status or 'VLM未连接/mock模式'}",
        "",
        "> 视觉模型输出为基于图像的可视特征分析，非确定的病原学诊断；"
        "严重程度为规则评估，处理建议未经植保知识库或专业人员验证。",
    ]
    return "\n".join(lines)


def _rag_card(knowledge: dict | None, status: str = "") -> str:
    """农业知识辅助诊断卡（RAG 阶段3：真实向量检索 / mock 回退）。"""
    if not knowledge:
        return "### 🌾 农业知识辅助诊断\n\n（尚未检测）"
    lines = [
        "### 🌾 农业知识辅助诊断",
        "",
        f"- 病害知识：{knowledge.get('disease_info', '')}",
        f"- 防治方法：{knowledge.get('control_method', '')}",
        f"- 推荐防治方案：{knowledge.get('treatment_plan', '')}",
        f"- 注意事项：{knowledge.get('precautions', '')}",
    ]
    rs = knowledge.get("retrieval_status", "mock")
    note = knowledge.get("retrieval_note", "")
    if rs == "ok":
        cache_hit = knowledge.get("cache_hit", False)
        cache_label = "（缓存复用）" if cache_hit else "（实际向量检索）"
        lines.append(f"- 检索状态：**真实向量检索**{cache_label}（{status}）")
        sources = knowledge.get("sources", [])
        if sources:
            lines.append("- 参考来源：")
            seen = set()
            for s in sources:
                url = s.get("url", "")
                pub = s.get("publisher", "") or s.get("title", "")
                if url and url not in seen:
                    seen.add(url)
                    lines.append(f"  - [{pub}]({url})")
        if note:
            lines.append("")
            lines.append(f"> {note}")
    else:
        lines.append(f"- 状态：{status or '本地知识库（mock）'}")
        if note:
            lines.append("")
            lines.append(f"> {note}")
        if rs in ("no_detection", "healthy_leaf", "no_evidence"):
            lines.append("> 未检索到充分文档证据；不得据此编造防治依据。")
    lines.append("")
    lines.append("> 知识检索结果不等同于专业植保诊断；防治信息未经中国大陆适用性核验。")
    return "\n".join(lines)


def _agent_card(decision: dict | None, status: str = "") -> str:
    """智能决策建议卡（Agent 阶段4：真实 LLM 智能体 / mock 规则决策）。"""
    if not decision:
        return "### 🧭 智能决策建议\n\n（尚未检测）"
    model = decision.get("model_name") or ""
    lines = [
        "### 🧭 智能决策建议",
        "",
        f"- 当前任务：{decision.get('task', '')}",
        f"- 优先级：**{decision.get('priority', '')}**",
        f"- 推荐动作：{decision.get('action', '')}",
        f"- 目标区域：{decision.get('target', '')}",
        f"- 决策依据：{decision.get('reason', '')}",
    ]
    if model:
        lines.append(f"- 智能体模型：**{model}**")
    tool_calls = decision.get("tool_calls") or []
    if tool_calls:
        names = "、".join(sorted({t.get("name", "") for t in tool_calls if t.get("name")}))
        lines.append(f"- 工具调用：{names}（{len(tool_calls)} 次）")
    lines.append(f"- 状态：{status or decision.get('agent_status') or 'mock模式（规则决策）'}")
    lines.append("")
    if decision.get("backend_name") == "llm" or "真实" in (status or ""):
        lines.append("> 决策由真实 LLM 智能体基于检测/VLM/RAG 综合生成，仅供人工复核参考；"
                     "严重程度为规则代理指标，不含具体施药指令。")
    else:
        lines.append("> 决策由规则 Agent（当前 mock 模式）基于多源结果生成，不等同于专业植保决策。")
    return "\n".join(lines)


def _robot_card(execution: dict | None, status: str = "", task: str = "") -> str:
    """SO-101 机械臂执行卡（阶段6.1，当前 mock）。"""
    if not execution:
        return "### 🤖 SO-101 机械臂执行\n\n（尚未执行）"
    lines = [
        "### 🤖 SO-101 机械臂执行",
        "",
        f"- 设备名称：{execution.get('robot', '')}",
        f"- 当前状态：{status or 'mock模式（SO-101未连接）'}",
        f"- 当前任务：{task or '无'}",
        f"- 执行动作：{execution.get('action', '')}",
        f"- 目标区域：{execution.get('target', '')}",
        f"- 执行结果：{execution.get('status', '')}：{execution.get('message', '')}",
        "",
        "> 机械臂执行当前为 mock 模式（SO-101 未连接），未驱动真实硬件；后续可接入 LeRobot / SO-101 follower API。",
    ]
    return "\n".join(lines)


def _agri_result_card(records: list[dict], assessments: list[dict] | None = None) -> str:
    """根据真实检测结果生成“🌱 农业视觉检测结果”卡片。

    - 作物按 13 类映射推断；多作物混合或无法判断时显示“未确定”
    - 病害类别按目标数从高到低展示
    - 仅展示真实目标数/最高/平均置信度，**不虚构严重程度/风险/经济损失/防治建议**
    - assessments 非空时，追加「病害严重程度评估」区（阶段1 规则评估）
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
    result = "\n".join(lines)
    if assessments:
        result += "\n\n" + _severity_section(assessments)
    return result


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


# ---------------------------------------------------------------------------
# 页面 CSS（深色农业主题，与比赛答辩 UI 协调）
# ---------------------------------------------------------------------------
def _build_background_css() -> str:
    """生成最外层背景 CSS：深绿生物科技叶脉图 + 深绿半透明遮罩。

    - 图片以 base64 data URI 内联，规避 Gradio 无静态目录、相对路径失效的问题；
    - background-size: cover + background-attachment: fixed 铺满视口、滚动固定、不拉伸变形；
    - 图片加载失败（文件缺失/读取异常）时回退到纯深色背景 #071b12。
    """
    bg_color = "#071b12"
    bg_image = "none"
    try:
        img = Path(__file__).resolve().parent.parent / "深绿生物科技叶脉背景.png"
        b64 = base64.b64encode(img.read_bytes()).decode("ascii")
        bg_image = f'url("data:image/png;base64,{b64}")'
    except Exception:  # noqa: BLE001
        bg_image = "none"
    return f"""
/* ===== 最外层背景：深绿生物科技叶脉（铺满视口、不拉伸、失败回退纯深色） ===== */
body {{
    background-color: {bg_color} !important;
    background-image: linear-gradient(rgba(8, 25, 18, .50), rgba(8, 25, 18, .50)), {bg_image} !important;
    background-position: center center !important;
    background-size: cover !important;
    background-repeat: no-repeat !important;
    background-attachment: fixed !important;
}}
gradio-app {{ background: transparent !important; }}
"""


CSS = _build_background_css() + """
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
/* ===== 大标题横幅独立样式（隔离全局主题变量，保持绿色渐变与文字颜色不变） ===== */
.hero { background:linear-gradient(120deg,#0b3d2e,#14532d,#1b5e20,#2e7d32) !important; color:#eafff1 !important; }
.hero h1 { color:#eafff1 !important; }
.hero-sub { color:#eafff1 !important; }
.hero-badges .badge { color:#fff !important; background:rgba(255,255,255,.14) !important; border:1px solid rgba(255,255,255,.3) !important; }
"""
