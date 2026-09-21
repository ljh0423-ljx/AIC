# -*- coding: utf-8 -*-
"""
web/services/vlm.py — VLM 视觉语言诊断模块（阶段2）
====================================================
在 PP-YOLOE 检测 + 严重程度评估基础上，提供自然语言化的视觉诊断描述。

后端选择（自动，无需改代码）：
    - 已配置 API Key（DASHSCOPE_API_KEY / AGRI_VLM_API_KEY）时 → QwenVLBackend
      通过 DashScope OpenAI 兼容接口调用 Qwen-VL 多模态模型（默认 qwen-vl-plus），
      真实读取原始图片并观察可见特征。
    - 未配置 / 调用失败时 → MockVLMBackend（仅基于检测结果拼接模拟解释）。

真实接入配置（环境变量，密钥不写入源码 / 日志 / 报告）：
    DASHSCOPE_API_KEY   Qwen 视觉模型 API Key（必填）
    AGRI_VLM_API_KEY    备用 Key 变量名（二者任一即可）
    AGRI_VLM_MODEL      模型名，默认 qwen3-vl-plus
    AGRI_VLM_BASE_URL   API 地址，默认 https://dashscope.aliyuncs.com/compatible-mode/v1
                        （百炼经典通用域名；若使用「工作空间」专用 Key，请改为
                          https://<WorkspaceId>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1）
    AGRI_VLM_TIMEOUT    请求超时秒数，默认 60

真实后端始终：
    - 读取原始图片（路径 / PIL / numpy 数组）并编码为 base64 送入模型；
    - 把 PP-YOLOE 检测类别 / 置信度 / 检测框与规则严重程度作为「参考上下文」，
      而不是当作唯一事实，要求模型观察图片后再描述可见特征；
    - 明确区分「图像观察事实 / 检测模型结果 / 规则严重程度 / 未经验证的建议」，
      不把输出表述为确定的病原学诊断。

本模块不依赖 Paddle、不依赖 inference 内部实现、不修改 PP-YOLOE 模型。
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger("agri.web.services.vlm")

_DEFAULT_MODEL = "qwen3-vl-plus"
_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DEFAULT_TIMEOUT = 60.0

_RESULT_KEYS = ("disease_description", "visual_evidence", "severity_analysis", "recommendation")


def _get(det, key: str, default=0.0):
    """归一化检测结果字段（兼容 Detection 对象与 dict）。"""
    if isinstance(det, dict):
        return det.get(key, default)
    return getattr(det, key, default)


class VLMBackend(ABC):
    """VLM 后端抽象接口：Qwen-VL / InternVL / GPT-4o Vision 等统一入口。"""

    backend_name: str = "base"
    status: str = "VLM未连接"
    model_name: str = ""

    @abstractmethod
    def analyze(self, image, detection_result, severity_result) -> dict:
        """对单张图片做视觉语言诊断，返回统一结果字典。

        返回：
            {
                "disease_description": 病害描述,
                "visual_evidence":     视觉依据,
                "severity_analysis":   风险分析,
                "recommendation":      处理建议,
            }
        可附加向后兼容字段："model"（实际模型名）、"status"、"error"。
        """


class MockVLMBackend(VLMBackend):
    """mock 后端：不真正调用 VLM，基于检测/严重程度结果生成模拟诊断。"""

    backend_name = "mock"
    status = "mock模式（VLM未连接）"
    model_name = ""

    def analyze(self, image, detection_result, severity_result) -> dict:
        return _mock_analyze(image, detection_result, severity_result)


def _mock_analyze(image, detection_result, severity_result) -> dict:
    """mock 实现：依据检测结果与严重程度评估拼接出模拟诊断文本。"""
    dets = list(detection_result or [])
    sev = severity_result if isinstance(severity_result, dict) else {}

    names: list[str] = []
    confs: list[float] = []
    for d in dets:
        nm = str(_get(d, "class_name", "") or "")
        if nm:
            names.append(nm)
        confs.append(float(_get(d, "confidence", 0.0)))

    unique = sorted(set(names))
    n = len(dets)
    max_conf = max(confs) if confs else 0.0
    severity = sev.get("severity", "轻度")
    ratio = float(sev.get("infection_ratio", 0.0))

    if not unique:
        disease_description = "未在图像中检测到明显病害特征，作物整体外观相对健康。"
        visual_evidence = "PP-YOLOE 未输出有效检测框，暂无可判读的病斑区域。"
    else:
        disease_description = (
            f"图像中检测到 {n} 个病害目标，涉及「{'、'.join(unique)}」等类别；"
            "结合病斑形态与分布，初步判断为叶部真菌/细菌性病害表现。"
        )
        visual_evidence = (
            f"检测框共 {n} 个，最高置信度 {max_conf:.2f}；"
            "病斑区域与健康组织存在颜色/纹理差异，符合典型叶部病害的视觉特征。"
        )

    severity_analysis = (
        f"病害区域占比（检测框代理）为 {ratio * 100:.1f}%，严重程度判定为「{severity}」。"
        + ("" if severity == "健康" else " 该比例为检测框覆盖区域估算，非像素级病斑面积。")
    )

    recommendation = sev.get("recommendation", "") or "建议结合田间实际情况，咨询植保专业人员进一步确认。"

    return {
        "disease_description": disease_description,
        "visual_evidence": visual_evidence,
        "severity_analysis": severity_analysis,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# 真实后端：Qwen-VL（DashScope OpenAI 兼容接口）
# ---------------------------------------------------------------------------
def _first_env(*names: str) -> str:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return ""


def _image_to_data_url(image) -> str:
    """把图片（文件路径 / PIL Image / numpy 数组）编码为 base64 data URL。"""
    if image is None:
        raise ValueError("未提供待分析图片")

    if isinstance(image, (str, Path)):
        p = Path(image)
        if not p.is_file():
            raise ValueError(f"图片文件不存在：{image}")
        raw = p.read_bytes()
        mime = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
            ".bmp": "image/bmp", ".webp": "image/webp", ".gif": "image/gif",
        }.get(p.suffix.lower(), "image/jpeg")
        return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")

    from PIL import Image
    import numpy as np

    if isinstance(image, np.ndarray):
        arr = image
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        pil = Image.fromarray(arr).convert("RGB")
    elif isinstance(image, Image.Image):
        pil = image.convert("RGB")
    else:
        raise ValueError(f"不支持的图片类型：{type(image)!r}")

    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _build_prompt(detection_result, severity_result) -> str:
    """构造发给 VLM 的提示：把检测/严重程度作为参考上下文，要求模型观察图片。"""
    dets = list(detection_result or [])
    sev = severity_result if isinstance(severity_result, dict) else {}

    lines = ["请仔细观察图片中的植物叶片，并结合以下参考信息输出 JSON。"]
    lines.append("")
    if dets:
        lines.append("【检测模型 PP-YOLOE+-m 给出的候选目标（仅供参考，非最终结论）】")
        for i, d in enumerate(dets, 1):
            cls = str(_get(d, "class_name", "") or "未知")
            conf = float(_get(d, "confidence", 0.0))
            x = _get(d, "x", None)
            box = ""
            if x is not None:
                box = (f"（框 x={float(_get(d, 'x', 0)):.0f}, y={float(_get(d, 'y', 0)):.0f}, "
                       f"w={float(_get(d, 'width', 0)):.0f}, h={float(_get(d, 'height', 0)):.0f}）")
            lines.append(f"  {i}. {cls}，置信度 {conf:.2f}{box}")
    else:
        lines.append("【检测模型未输出任何候选目标】")
    lines.append("")
    lines.append(
        f"【规则严重程度评估】等级={sev.get('severity', '未知')}，"
        f"检测框面积占比={float(sev.get('infection_ratio', 0)) * 100:.1f}%（规则估算，非像素级）"
    )
    lines.append("")
    lines.append("请输出一个 JSON 对象，只包含以下 4 个字段，内容用中文：")
    lines.append('  "disease_description": 仅描述你从图片中真实观察到的可见特征（颜色、纹理、病斑形态与分布、霉层/坏死等）。')
    lines.append('  "visual_evidence": 你的视觉依据，并明确区分「图片中直接可见的事实」与「检测模型给出的类别/置信度」。')
    lines.append('  "severity_analysis": 结合图片表现与上述规则严重程度评估风险，并说明该严重程度为规则评估而非像素级诊断。')
    lines.append('  "recommendation": 给出处理建议，并标注该建议未经植保知识库或专业人员验证。')
    lines.append("")
    lines.append("重要约束：不要把检测类别直接当作确定的病原学诊断；对不确定之处明确说明；只输出 JSON。")
    return "\n".join(lines)


def _parse_json(content: str) -> dict:
    """解析模型返回内容；非 JSON 时整段当作病害描述兜底。"""
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
    return {"disease_description": text, "visual_evidence": "", "severity_analysis": "", "recommendation": ""}


def _to_result(data: dict) -> dict:
    """把模型返回的任意 dict 规整为 4 字段结果。"""

    def s(key: str) -> str:
        v = data.get(key, "")
        if isinstance(v, str):
            return v
        if v is None:
            return ""
        return json.dumps(v, ensure_ascii=False)

    return {k: s(k) for k in _RESULT_KEYS}


def _is_response_format_unsupported(e) -> bool:
    """仅当错误为 400 参数类且与 response_format/JSON 相关时返回 True，
    避免对鉴权失败、限流或网络错误做盲目重试（产生不必要费用）。"""
    try:
        from openai import BadRequestError

        if isinstance(e, BadRequestError):
            msg = str(e).lower()
            return "response_format" in msg or "json" in msg
    except Exception:  # noqa: BLE001
        pass
    return False


class QwenVLBackend(VLMBackend):
    """真实 VLM 后端：通过 DashScope OpenAI 兼容接口调用 Qwen-VL。

    - 已配置 API Key 时 available=True；未配置时 available=False（工厂会改用 Mock）。
    - 每次调用读取原始图片并送入多模态模型；调用失败抛异常，由 analyze_image 兜底回退 Mock。
    """

    backend_name = "qwen_vl"

    def __init__(self, api_key: str = "", model: str = "", base_url: str = "", timeout=None) -> None:
        self.api_key = api_key or _first_env("DASHSCOPE_API_KEY", "AGRI_VLM_API_KEY", "QWEN_API_KEY")
        self.model_name = model or _first_env("AGRI_VLM_MODEL") or _DEFAULT_MODEL
        self.base_url = base_url or _first_env("AGRI_VLM_BASE_URL") or _DEFAULT_BASE_URL
        try:
            self.timeout = float(timeout or _first_env("AGRI_VLM_TIMEOUT") or _DEFAULT_TIMEOUT)
        except ValueError:
            self.timeout = _DEFAULT_TIMEOUT
        try:
            self.max_retries = int(_first_env("AGRI_API_MAX_RETRIES") or 2)
        except ValueError:
            self.max_retries = 2
        self.available = bool(self.api_key)
        self.status = (
            f"真实 VLM（{self.model_name}）" if self.available
            else "真实 VLM 未配置（缺 API Key，已回退 Mock）"
        )
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                                  timeout=self.timeout, max_retries=self.max_retries)
        return self._client

    def analyze(self, image, detection_result, severity_result) -> dict:
        if not self.available:
            raise RuntimeError("未配置 VLM API Key（DASHSCOPE_API_KEY / AGRI_VLM_API_KEY）")
        data_url = _image_to_data_url(image)
        prompt = _build_prompt(detection_result, severity_result)
        content = self._call(data_url, prompt)
        data = _parse_json(content)
        result = _to_result(data)
        result["model"] = self.model_name
        return result

    def _call(self, data_url: str, prompt: str) -> str:
        client = self._ensure_client()
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ],
        }]
        # 优先 JSON 输出模式；仅当模型/端点不支持 response_format（400 参数类）时回退普通文本，
        # 对鉴权失败(401)/限流(429)/网络错误不做盲目重试（这些由 SDK 的 max_retries 按策略处理）。
        try:
            resp = client.chat.completions.create(
                model=self.model_name, messages=messages,
                response_format={"type": "json_object"}, max_tokens=1024,
            )
        except Exception as e:  # noqa: BLE001
            if not _is_response_format_unsupported(e):
                raise
            logger.warning("VLM JSON 模式不被支持，回退普通文本：%s", e)
            resp = client.chat.completions.create(
                model=self.model_name, messages=messages, max_tokens=1024,
            )
        return (resp.choices[0].message.content or "") if resp.choices else ""


# ---------------------------------------------------------------------------
# 默认后端工厂：有 Key 用真实 Qwen-VL，否则回退 Mock
# ---------------------------------------------------------------------------
def _build_default_backend() -> VLMBackend:
    try:
        backend = QwenVLBackend()
        if backend.available:
            logger.info("VLM 真实后端已启用：%s（base_url=%s）", backend.model_name, backend.base_url)
            return backend
        logger.info("VLM 未配置 API Key，使用 Mock 后端（不影响检测流程）")
    except Exception as e:  # noqa: BLE001
        logger.warning("VLM 真实后端初始化失败，回退 Mock：%s", e)
    return MockVLMBackend()


# 当前默认后端（自动选择真实 / mock）。
_default_backend: VLMBackend = _build_default_backend()


def analyze_image(image, detection_result, severity_result) -> dict:
    """对外统一入口：调用当前 VLM 后端做视觉语言诊断。

    参数：
        image            原始图片（文件路径 / PIL Image / numpy 数组；真实后端会读取并送入模型）
        detection_result PP-YOLOE 检测结果（Detection 对象列表或 dict 列表，仅作参考上下文）
        severity_result  严重程度评估结果（web.services.severity.calculate_severity 返回值）

    返回：
        {
            "disease_description": 病害描述,
            "visual_evidence":     视觉依据,
            "severity_analysis":   风险分析,
            "recommendation":      处理建议,
            "model":  实际模型名（真实后端）/ ""（mock）,
            "status": 后端状态说明,
            # "error": 仅在真实调用失败回退 mock 时出现,
        }
    """
    backend = _default_backend
    try:
        result = dict(backend.analyze(image, detection_result, severity_result))
    except Exception as e:  # noqa: BLE001
        logger.error("VLM 分析失败，回退 Mock 结果：%s", e)
        result = dict(_mock_analyze(image, detection_result, severity_result))
        result["model"] = ""
        result["status"] = f"{backend.status}（调用失败，已回退 Mock）"
        result["error"] = str(e)[:200]
        return result
    result.setdefault("model", getattr(backend, "model_name", ""))
    result.setdefault("status", backend.status)
    return result


def get_vlm_status() -> str:
    """返回当前 VLM 后端状态（如 'mock模式（VLM未连接）' / '真实 VLM（qwen-vl-plus）'）。"""
    return _default_backend.status


def get_vlm_model() -> str:
    """返回当前 VLM 实际模型名（真实后端）；mock 时为空字符串。"""
    return getattr(_default_backend, "model_name", "")
