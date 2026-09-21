# -*- coding: utf-8 -*-
"""
web/services/agent.py — 农业智能决策 Agent 模块（阶段4）
==========================================================
在 PP-YOLOE 检测 + 严重程度评估 + VLM 诊断 + RAG 知识增强基础上，
输出可执行的农业决策建议（任务 / 优先级 / 推荐动作 / 目标区域 / 决策依据）。

当前阶段（mock）：不接入真实 LLM，使用规则决策（MockAgentBackend）。

决策规则（按 severity）：
    重度 → 优先级「高」，动作「立即处理」
    中度 → 优先级「中」，动作「持续监测并准备防治」
    轻度 → 优先级「低」，动作「加强观察」
    健康 → 优先级「无」，动作「无需处理」

未来接入（预留，保持 make_decision 签名不变）：
    - DeepSeek
    - GPT API
    - Qwen-Agent
    - LangGraph
只需实现 AgentBackend 子类并替换 _default_backend，Web 层与其余模块零改动。

本模块不依赖 Paddle、不依赖 inference 内部实现、不修改 PP-YOLOE 模型，
也不影响 severity.py / vlm.py / rag.py 现有流程。
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger("agri.web.services.agent")

_DEFAULT_MODEL = "qwen-plus"
_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DEFAULT_TIMEOUT = 90.0
_MAX_TOOL_ROUNDS = 3


# 决策规则表（severity -> priority / action）
_DECISION_RULES = {
    "重度": {"priority": "高", "action": "立即处理"},
    "中度": {"priority": "中", "action": "持续监测并准备防治"},
    "轻度": {"priority": "低", "action": "加强观察"},
    "健康": {"priority": "无", "action": "无需处理"},
}


def _crop_of(class_name) -> str:
    n = (class_name or "").strip().lower()
    if n.startswith("tomato"):
        return "番茄"
    if n.startswith("apple"):
        return "苹果"
    if n.startswith("grape"):
        return "葡萄"
    return "该作物"


class AgentBackend(ABC):
    """农业智能决策后端抽象接口：DeepSeek / GPT / Qwen-Agent / LangGraph 等统一入口。"""

    backend_name: str = "base"
    status: str = "未连接"
    model_name: str = ""

    @abstractmethod
    def decide(self, detection_result, severity_result, vlm_result, rag_result) -> dict:
        """综合多源结果输出决策建议。

        返回：
            {
                "task":     当前任务,
                "priority": 优先级,
                "action":   推荐动作,
                "target":   目标区域,
                "reason":   决策依据,
            }
        """

    @abstractmethod
    def answer_question(self, question, rag_result) -> dict:
        """基于 RAG 检索证据回答用户知识问题（中文）。

        返回：
            {
                "answer":          中文回答文本,
                "model_name":      实际模型名（mock 为空）,
                "agent_status":    Agent 运行状态,
                "backend_name":    "llm" / "mock",
                "evidence_sources": 引用的来源列表,
            }
        """


class MockAgentBackend(AgentBackend):
    """mock 后端：基于规则表决策，不接入真实 LLM。"""

    backend_name = "mock"
    status = "mock模式（规则决策）"

    def decide(self, detection_result, severity_result, vlm_result, rag_result,
               vlm_status: str = "", rag_status: str = "") -> dict:
        return _mock_decision(detection_result, severity_result, vlm_result, rag_result,
                              vlm_status, rag_status)

    def answer_question(self, question, rag_result) -> dict:
        rag_r = rag_result if isinstance(rag_result, dict) else {}
        answer = (
            f"（Mock 本地演示，非真实知识问答）"
            f"{rag_r.get('disease_info', '')}；{rag_r.get('control_method', '')}；"
            f"{rag_r.get('precautions', '')}"
        )
        return {
            "answer": answer,
            "model_name": "",
            "agent_status": self.status,
            "backend_name": self.backend_name,
            "evidence_sources": [],
        }


def _is_mock_status(status: str) -> bool:
    """按状态串判断对应后端是否为 mock（未接入真实 VLM / 文档检索）。

    状态串为空（未透传）时按 mock 处理，宁可多标注 mock，也不误报为真实模型。
    """
    s = (status or "").strip().lower()
    if not s:
        return True
    return "mock" in s or "未连接" in s or "本地知识库" in s


def _vlm_reason(vlm_status: str) -> str:
    """决策依据中的 VLM 表述：mock 时明确为模拟解释，真实时才是视觉模型确认。"""
    if _is_mock_status(vlm_status):
        return "VLM 视觉诊断（Mock：仅依据已有检测结果生成的模拟解释，非真实视觉模型确认）"
    return "经 VLM 视觉诊断确认"


def _rag_reason(rag_status: str, control_method: str) -> str:
    """决策依据中的 RAG 表述：区分本地 Mock 字典与真实有来源文档检索。"""
    method = str(control_method).rstrip("。")
    if _is_mock_status(rag_status):
        return f"本地 Mock 字典（非文档检索）防治依据：{method}"
    return f"知识库文档检索防治依据：{method}"


def _mock_decision(detection_result, severity_result, vlm_result, rag_result,
                   vlm_status: str = "", rag_status: str = "") -> dict:
    """mock 实现：按 severity 规则表决策，并结合检测/VLM/RAG 结果生成依据。

    决策规则（优先级/动作/任务）保持不变；VLM/RAG 的依据文案按各自真实运行状态
    （mock / 真实）诚实标注，避免把 mock 结果表述为真实视觉模型确认或文档检索。
    """
    sev = severity_result if isinstance(severity_result, dict) else {}
    vlm_r = vlm_result if isinstance(vlm_result, dict) else {}
    rag_r = rag_result if isinstance(rag_result, dict) else {}

    severity = sev.get("severity", "轻度")
    class_name = sev.get("class_name", "")
    bbox = sev.get("bbox", "")
    ratio = float(sev.get("infection_ratio", 0.0))
    crop = _crop_of(class_name)
    n_dets = len(list(detection_result or []))

    rule = _DECISION_RULES.get(severity, _DECISION_RULES["轻度"])
    priority = rule["priority"]
    action = rule["action"]

    if severity == "健康":
        task = f"{crop}例行健康巡查" if crop != "该作物" else "作物例行健康巡查"
        target = "整片田块（常规巡查）"
    else:
        task = f"{crop}·{class_name} 的防控处置"
        target = f"病害侵染区域（检测框 {bbox}）" if bbox else "病害侵染区域"

    reason_parts = [f"严重程度判定为「{severity}」，病害区域占比（检测框代理）{ratio * 100:.1f}%"]
    if n_dets:
        reason_parts.append(f"共检测到 {n_dets} 个病害目标")
    if vlm_r:
        reason_parts.append(_vlm_reason(vlm_status))
    if rag_r.get("control_method"):
        reason_parts.append(_rag_reason(rag_status, rag_r["control_method"]))
    reason = "；".join(reason_parts) + "。"

    return {
        "task": task,
        "priority": priority,
        "action": action,
        "target": target,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# 真实 LLM 智能体后端：受控工具调用（function-calling）
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "你是农业视觉诊断辅助智能体，负责综合多源结果生成信息性诊断解释与人工复核建议。\n"
    "必须严格遵守：\n"
    "1. 明确区分四类信息并标注来源：PP-YOLOE 检测模型推断、VLM 可见特征、RAG 文档证据、你自己的综合分析。\n"
    "2. 所有资料性主张只能引用实际检索到的来源，不得编造文献、页码、农药登记号或防治方法。\n"
    "3. 严重程度（severity）仅为检测框面积占比的规则代理指标，不得据此认定真实感染面积，也不得据此触发施药。\n"
    "4. 健康叶类别或未检出病害，不得表述为「已确认健康」。\n"
    "5. 不得生成具体药剂、剂量、施药间隔或自动施药指令（现有 RAG 资料为境外公开来源，未经中国大陆登记与安全间隔期核验）。\n"
    "6. 只输出信息性诊断解释与人工复核建议，不向机械臂下发物理动作，不增加自动喷药/采样/自主执行。\n"
    "7. 需要证据时必须通过工具调用获取（query_knowledge / get_detection_result / get_vlm_analysis），不得臆造。\n"
    "最终以 JSON 输出，字段：task（信息性任务描述）、priority（高/中/低/无，并注明是基于规则代理的粗略估计）、"
    "action（信息性人工复核建议，不得含具体药剂/剂量/自动施药）、target（检测框对应区域描述）、"
    "reason（综合分析与依据，明确区分检测推断/VLM可见/RAG证据/LLM综合）。只输出 JSON。"
)

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_knowledge",
            "description": "检索农业知识库，获取指定病害的症状、发生条件、传播途径、监测与综合防治资料（含可追溯来源）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "disease_name": {"type": "string", "description": "病害英文类别名"},
                    "crop_name": {"type": "string", "description": "作物名：番茄/苹果/葡萄"},
                    "severity": {"type": "string", "description": "严重程度：健康/轻度/中度/重度"},
                },
                "required": ["disease_name", "crop_name", "severity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_detection_result",
            "description": "读取当前图片的 PP-YOLOE 检测结果（类别、置信度、检测框）。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_vlm_analysis",
            "description": "获取当前图片的 VLM 视觉诊断分析（可见特征描述，不重复调用 VLM）。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _summarize_detection(detection_result) -> list[dict]:
    out = []
    for d in list(detection_result or []):
        def g(k, dv=""):
            if isinstance(d, dict):
                return d.get(k, dv)
            return getattr(d, k, dv)
        out.append({
            "class_name": str(g("class_name", "")),
            "confidence": round(float(g("confidence", 0.0)), 3),
            "bbox": f"x={g('x', 0):.0f},y={g('y', 0):.0f},w={g('width', 0):.0f},h={g('height', 0):.0f}",
        })
    return out


def _summarize_vlm(vlm_result) -> dict:
    v = vlm_result if isinstance(vlm_result, dict) else {}
    return {
        "disease_description": v.get("disease_description", ""),
        "visual_evidence": v.get("visual_evidence", ""),
        "severity_analysis": v.get("severity_analysis", ""),
        "recommendation": v.get("recommendation", ""),
        "model": v.get("model", ""),
    }


def _build_context(detection_result, severity_result, vlm_result, rag_result, vlm_status, rag_status) -> str:
    sev = severity_result if isinstance(severity_result, dict) else {}
    lines = [
        "【PP-YOLOE 检测模型推断（非诊断结论）】",
        json.dumps(_summarize_detection(detection_result), ensure_ascii=False),
        f"【规则严重程度评估（检测框面积占比代理指标，非像素级感染面积）】severity={sev.get('severity', '未知')}，"
        f"infection_ratio={float(sev.get('infection_ratio', 0)):.4f}，class_name={sev.get('class_name', '')}，"
        f"bbox={sev.get('bbox', '')}",
        f"【VLM 可见特征分析（{vlm_status or '未知状态'}）】",
        json.dumps(_summarize_vlm(vlm_result), ensure_ascii=False),
        f"【已检索到的 RAG 资料（{rag_status or '未知状态'}；来源见 sources）】",
    ]
    rag_r = rag_result if isinstance(rag_result, dict) else {}
    lines.append(json.dumps({
        "disease_info": rag_r.get("disease_info", ""),
        "control_method": rag_r.get("control_method", ""),
        "precautions": rag_r.get("precautions", ""),
        "retrieval_status": rag_r.get("retrieval_status", ""),
        "sources": rag_r.get("sources", []),
    }, ensure_ascii=False))
    return "\n".join(lines)


def _parse_json(content: str) -> dict:
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:  # noqa: BLE001
        pass
    return {"reason": text}


def _collect_evidence(rag_result, tool_results: list[dict]) -> list[dict]:
    """汇总实际可追溯来源：预检索的 rag_result.sources + 工具调用 query_knowledge 返回的 sources。"""
    evidence: list[dict] = []
    seen = set()

    def add(src):
        url = (src or {}).get("url", "")
        if url and url not in seen:
            seen.add(url)
            evidence.append(src)

    rag_r = rag_result if isinstance(rag_result, dict) else {}
    for s in rag_r.get("sources", []) or []:
        add(s)
    for tr in tool_results:
        for s in (tr.get("sources") or []):
            add(s)
    return evidence


class LLMAgentBackend(AgentBackend):
    """真实 LLM 智能体后端：支持工具调用的语言模型（默认 qwen-plus）。

    - 通过 function-calling 真实发起工具调用（tool_calls 记录实际调用名与参数），
      与「直接拼接已生成结果」严格区分。
    - get_vlm_analysis 只返回已生成的 VLM 结果，绝不重复调用 VLM（避免额外费用）。
    - 严格校验工具名与参数；最多 _MAX_TOOL_ROUNDS 轮；设置超时。
    - 仅生成信息性诊断解释与人工复核建议。
    """

    backend_name = "llm"

    def __init__(self, api_key: str = "", model: str = "", base_url: str = "", timeout=None) -> None:
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY") \
            or os.environ.get("AGRI_AGENT_API_KEY") or ""
        self.model_name = model or os.environ.get("AGRI_AGENT_MODEL") or _DEFAULT_MODEL
        self.base_url = base_url or os.environ.get("AGRI_AGENT_BASE_URL") or _DEFAULT_BASE_URL
        try:
            self.timeout = float(timeout or os.environ.get("AGRI_AGENT_TIMEOUT") or _DEFAULT_TIMEOUT)
        except ValueError:
            self.timeout = _DEFAULT_TIMEOUT
        try:
            self.max_retries = int(os.environ.get("AGRI_API_MAX_RETRIES") or 2)
        except ValueError:
            self.max_retries = 2
        self.available = bool(self.api_key)
        self.status = f"真实 LLM 智能体（{self.model_name}）" if self.available \
            else "真实 LLM 未配置（缺 API Key，已回退 Mock）"
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                                  timeout=self.timeout, max_retries=self.max_retries)
        return self._client

    def _execute_tool(self, name: str, args: dict, detection_result, vlm_result) -> dict:
        """受控工具执行：校验工具名与参数后执行。"""
        if name == "query_knowledge":
            dn = str(args.get("disease_name", "")).strip()
            crop = str(args.get("crop_name", "")).strip()
            sev = str(args.get("severity", "")).strip()
            if sev not in ("健康", "轻度", "中度", "重度"):
                sev = "轻度"
            if crop not in ("番茄", "苹果", "葡萄"):
                guess = _crop_of(dn)
                crop = guess if guess != "该作物" else "番茄"
            if not dn:
                return {"error": "query_knowledge 缺少 disease_name 参数"}
            from web.services import rag

            r = rag.query_knowledge(dn, crop, sev)
            return {
                "retrieval_status": r.get("retrieval_status", ""),
                "disease_info": r.get("disease_info", ""),
                "control_method": r.get("control_method", ""),
                "precautions": r.get("precautions", ""),
                "sources": r.get("sources", []),
            }
        if name == "get_detection_result":
            return {"detection_summary": _summarize_detection(detection_result)}
        if name == "get_vlm_analysis":
            return {"vlm_analysis": _summarize_vlm(vlm_result)}
        return {"error": f"未知工具：{name}"}

    def decide(self, detection_result, severity_result, vlm_result, rag_result,
               vlm_status: str = "", rag_status: str = "") -> dict:
        if not self.available:
            raise RuntimeError("未配置 Agent API Key（DASHSCOPE_API_KEY / AGRI_AGENT_API_KEY）")
        client = self._ensure_client()
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_context(detection_result, severity_result, vlm_result, rag_result, vlm_status, rag_status)},
        ]
        tool_calls_log: list[dict] = []
        tool_results: list[dict] = []
        final_content = ""
        llm_rounds = 0

        for _ in range(_MAX_TOOL_ROUNDS):
            llm_rounds += 1
            resp = client.chat.completions.create(
                model=self.model_name, messages=messages, tools=_TOOLS,
                tool_choice="auto", temperature=0.2, max_tokens=1200,
            )
            msg = resp.choices[0].message
            if msg.tool_calls:
                messages.append({
                    "role": "assistant", "content": msg.content or "",
                    "tool_calls": [{
                        "id": tc.id, "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    } for tc in msg.tool_calls],
                })
                for tc in msg.tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                        if not isinstance(args, dict):
                            args = {}
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls_log.append({"name": name, "arguments": args})
                    result = self._execute_tool(name, args, detection_result, vlm_result)
                    tool_results.append(result)
                    messages.append({
                        "role": "tool", "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                continue
            final_content = msg.content or ""
            break

        if not final_content:
            final_content = "（未获取到模型最终回答）"
        data = _parse_json(final_content)
        result = {
            "task": str(data.get("task", "")),
            "priority": str(data.get("priority", "")),
            "action": str(data.get("action", "")),
            "target": str(data.get("target", "")),
            "reason": str(data.get("reason", "")),
            "backend_name": self.backend_name,
            "model_name": self.model_name,
            "agent_status": self.status,
            "tool_calls": tool_calls_log,
            "evidence_sources": _collect_evidence(rag_result, tool_results),
            "llm_rounds": llm_rounds,
        }
        return result

    def answer_question(self, question, rag_result) -> dict:
        """基于 RAG 检索证据回答用户知识问题（中文，不编造来源/剂量）。"""
        if not self.available:
            raise RuntimeError("未配置 Agent API Key（DASHSCOPE_API_KEY / AGRI_AGENT_API_KEY）")
        client = self._ensure_client()
        rag_r = rag_result if isinstance(rag_result, dict) else {}
        evidence = {
            "disease_info": rag_r.get("disease_info", ""),
            "control_method": rag_r.get("control_method", ""),
            "precautions": rag_r.get("precautions", ""),
            "retrieval_status": rag_r.get("retrieval_status", ""),
            "sources": [{"title": s.get("title", ""), "publisher": s.get("publisher", ""), "url": s.get("url", "")}
                        for s in rag_r.get("sources", [])],
        }
        sys_prompt = (
            "你是农业知识问答助手，用中文简洁回答用户问题。"
            "只依据提供的检索资料回答，不得编造来源、文献、农药登记号或剂量。"
            "资料不足时明确说明证据不足；不输出具体药剂剂量或施药间隔（资料未经中国大陆登记核验）。"
            "回答中区分「资料内容」与「你的说明」。"
        )
        user_prompt = f"用户问题：{question}\n\n检索资料（JSON）：\n{json.dumps(evidence, ensure_ascii=False)}"
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.3, max_tokens=800,
        )
        answer = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        return {
            "answer": answer,
            "model_name": self.model_name,
            "agent_status": self.status,
            "backend_name": self.backend_name,
            "evidence_sources": rag_r.get("sources", []),
        }


# ---------------------------------------------------------------------------
# 默认后端工厂：有 Key 用真实 LLM 智能体，否则回退 Mock
# ---------------------------------------------------------------------------
def _build_default_backend() -> AgentBackend:
    try:
        b = LLMAgentBackend()
        if b.available:
            logger.info("Agent 真实后端已启用：%s（base_url=%s）", b.model_name, b.base_url)
            return b
        logger.info("Agent 未配置 API Key，使用 Mock 后端")
    except Exception as e:  # noqa: BLE001
        logger.warning("Agent 真实后端初始化失败，回退 Mock：%s", e)
    return MockAgentBackend()


_default_backend: AgentBackend = _build_default_backend()


def make_decision(detection_result, severity_result, vlm_result, rag_result,
                  vlm_status: str = "", rag_status: str = "") -> dict:
    """对外统一入口：综合多源结果输出农业智能决策建议。

    参数：
        detection_result PP-YOLOE 检测结果（Detection 对象列表或 dict 列表）
        severity_result  严重程度评估结果（severity.calculate_severity 返回值）
        vlm_result       VLM 视觉诊断结果（vlm.analyze_image 返回值）
        rag_result       RAG 知识增强结果（rag.query_knowledge 返回值）
        vlm_status       VLM 后端当前运行状态（vlm.get_vlm_status()），用于诚实标注依据来源
        rag_status       RAG 后端当前运行状态（rag.get_rag_status()），用于诚实标注依据来源

    返回：
        {
            "task":     当前任务,
            "priority": 优先级,
            "action":   推荐动作,
            "target":   目标区域,
            "reason":   决策依据,
            # 以下为向后兼容新增字段：
            "backend_name":  "llm" / "mock",
            "model_name":    实际模型名（mock 为空）,
            "agent_status":  Agent 运行状态说明,
            "tool_calls":    [{name, arguments}, ...] 实际发起的工具调用,
            "evidence_sources": [可追溯来源],
        }
    """
    backend = _default_backend
    try:
        result = dict(backend.decide(
            detection_result, severity_result, vlm_result, rag_result,
            vlm_status=vlm_status, rag_status=rag_status,
        ))
    except Exception as e:  # noqa: BLE001
        logger.error("Agent 真实调用失败，回退 Mock：%s", e)
        result = dict(_mock_decision(
            detection_result, severity_result, vlm_result, rag_result,
            vlm_status, rag_status,
        ))
        result["agent_status"] = f"{backend.status}（调用失败，已回退 Mock）"
        result["backend_name"] = "mock"
        result["model_name"] = ""
        result["tool_calls"] = []
        result["evidence_sources"] = []
    result.setdefault("backend_name", getattr(backend, "backend_name", "mock"))
    result.setdefault("model_name", getattr(backend, "model_name", ""))
    result.setdefault("agent_status", getattr(backend, "status", ""))
    result.setdefault("tool_calls", [])
    result.setdefault("evidence_sources", [])
    return result


def get_agent_status() -> str:
    """返回当前 Agent 后端状态（如 'mock模式（规则决策）' / '真实 LLM 智能体（qwen-plus）'）。"""
    return _default_backend.status


def get_agent_info() -> dict:
    """返回 Agent 运行信息（供页面展示与诊断，不含密钥）。"""
    return {
        "backend": _default_backend.backend_name,
        "model": getattr(_default_backend, "model_name", ""),
        "available": getattr(_default_backend, "available", False),
        "status": _default_backend.status,
    }


def answer_question(question, rag_result) -> dict:
    """对外统一入口：基于 RAG 检索证据回答用户知识问题（中文）。

    真实后端失败时回退为资料摘录（明确标注失败，不冒充 LLM 回答）。
    """
    try:
        return _default_backend.answer_question(question, rag_result)
    except Exception as e:  # noqa: BLE001
        logger.error("Agent 问答失败，回退资料摘录：%s", e)
        rag_r = rag_result if isinstance(rag_result, dict) else {}
        return {
            "answer": f"（问答失败，以下为检索资料原文摘录）"
                      f"{rag_r.get('disease_info', '')}；{rag_r.get('control_method', '')}；{rag_r.get('precautions', '')}",
            "model_name": "",
            "agent_status": f"{_default_backend.status}（问答失败）",
            "backend_name": "mock",
            "evidence_sources": [],
        }
