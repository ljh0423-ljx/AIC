# -*- coding: utf-8 -*-
"""
web/services/speech.py — 农业智能助手语音交互模块（阶段5：ASR + TTS）
====================================================================
提供语音识别（ASR，含文件与实时麦克风流）与语音合成（TTS）接口，实现
「麦克风实时语音输入 → ASR → Agent → VLM/RAG 诊断 → TTS 语音回答」的自然语言交互链路。

当前阶段：优先尝试真实 ASR（FunASR/Whisper），未安装时自动回退 mock。
    ASR：真实 ASR（FunASR/Whisper）→ 未安装则回退 MockASRBackend（固定模拟文本）
    TTS：生成占位提示音（非真实语音合成，MockTTSBackend）

未来接入（预留，保持接口签名不变）：
    ASR（文件 / 实时流）：Whisper、Whisper Streaming、FunASR、Paraformer
    TTS：Edge-TTS、CosyVoice
只需实现对应 Backend 子类并替换默认后端，Web 层与其余模块零改动。

本模块不依赖 Paddle、不依赖 inference 内部实现、不修改 PP-YOLOE 模型，
也不影响 severity.py / vlm.py / rag.py / agent.py 现有流程。
"""

from __future__ import annotations

import logging
import math
import struct
import time
import wave
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger("agri.web.services.speech")

# 输出目录：web/tmp/（与上传暂存同目录，Web 可直接读取）
_OUTPUT_DIR: Path = Path(__file__).resolve().parent.parent / "tmp"

# mock ASR 固定识别文本（模拟用户语音提问）
_MOCK_ASR_TEXT = "请帮我看看这片作物叶片得了什么病害，严重吗，该怎么处理？"


class ASRBackend(ABC):
    """语音识别后端抽象接口：Whisper / FunASR / Paraformer 等统一入口。"""

    backend_name: str = "base"
    status: str = "未连接"

    @abstractmethod
    def transcribe(self, audio) -> dict:
        """语音（文件）→ 文本，返回 {"text": ...}。"""

    @abstractmethod
    def transcribe_stream(self, audio_stream) -> dict:
        """实时麦克风语音流 → 文本，返回 {"text": ...}。"""


class TTSBackend(ABC):
    """语音合成后端抽象接口：Edge-TTS / CosyVoice 等统一入口。"""

    backend_name: str = "base"
    status: str = "未连接"

    @abstractmethod
    def synthesize(self, text) -> dict:
        """文本 → 音频，返回 {"audio_path": ...}。"""


class MockASRBackend(ASRBackend):
    """mock ASR：不真正识别语音，返回固定模拟文本。"""

    backend_name = "mock"
    status = "mock模式（ASR未连接）"

    def transcribe(self, audio) -> dict:
        if not audio:
            return {"text": ""}
        return {"text": _MOCK_ASR_TEXT}

    def transcribe_stream(self, audio_stream) -> dict:
        if not audio_stream:
            return {"text": ""}
        return {"text": _MOCK_ASR_TEXT}


_SENSEVOICE_REPO = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"


def _read_audio_samples(audio_path: str):
    """读取音频为 (sample_rate, float32 单声道样本)。"""
    import wave

    import numpy as np

    with wave.open(audio_path, "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        data = w.readframes(w.getnframes())
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        samples = samples.reshape(-1, nch).mean(axis=1)
    return sr, samples


def _normalize_audio(audio):
    """把 Gradio Audio 输入归一化为 (sample_rate, float32 单声道样本数组)。

    支持：
    - Gradio type="numpy" 的 (sample_rate, np.ndarray) 元组（麦克风/上传）
    - 旧的 type="filepath" 字符串路径（兼容，用 wave 读取）
    返回 None 表示无法解析（调用方回退 mock）。
    """
    import numpy as np

    # Gradio type="numpy"：返回 (sample_rate, np.ndarray)
    if isinstance(audio, (tuple, list)) and len(audio) == 2:
        sr, samples = audio
        arr = np.asarray(samples).astype(np.float32)
        if arr.ndim == 2:
            arr = arr.mean(axis=1)  # 立体声 → 单声道
        arr = arr.reshape(-1)
        # 若幅度明显 >1，说明是 16bit 整数采样，除以 32768 归一化到 [-1,1]
        if arr.size and float(np.abs(arr).max()) > 1.0:
            arr = arr / 32768.0
        return int(sr), arr

    # 兼容旧 filepath 字符串
    if isinstance(audio, (str, Path)):
        return _read_audio_samples(str(audio))

    return None


class FunASRBackend(ASRBackend):
    """真实 ASR 后端：优先 sherpa-onnx + SenseVoice（轻量 CPU），可选 funasr / whisper。

    - sherpa-onnx + SenseVoice 模型可用时：available=True，执行真实中文语音转写。
    - 未安装 / 模型缺失时：available=False，由默认后端工厂自动回退 MockASRBackend。
    """

    backend_name = "funasr"
    status = "真实 ASR（SenseVoice/FunASR/Whisper）"

    def __init__(self) -> None:
        self.available = False
        self._kind = None
        self._model = None
        self._model_path = None
        self._tokens_path = None
        self._load_engine()

    def _load_engine(self) -> None:
        """探测可用的真实 ASR 引擎（仅解析模型路径，不把模型读入内存）。"""
        try:
            import importlib.util

            # 1) 优先 sherpa-onnx + SenseVoice（轻量，CPU 可跑）
            if importlib.util.find_spec("sherpa_onnx") is not None:
                from huggingface_hub import hf_hub_download

                self._model_path = hf_hub_download(
                    repo_id=_SENSEVOICE_REPO, filename="model.onnx", local_files_only=True
                )
                self._tokens_path = hf_hub_download(
                    repo_id=_SENSEVOICE_REPO, filename="tokens.txt", local_files_only=True
                )
                self._kind = "sherpa_sensevoice"
                self.available = True
                return
            # 2) funasr（torch，重）
            if importlib.util.find_spec("funasr") is not None:
                self._kind = "funasr"
                self.available = True
                return
            # 3) whisper
            if importlib.util.find_spec("whisper") is not None:
                self._kind = "whisper"
                self.available = True
                return
        except Exception:
            pass
        self.available = False
        self._kind = None

    def _ensure_model(self):
        """懒加载真实识别器（首次转写时才把模型读入内存）。"""
        if self._model is None and self._kind == "sherpa_sensevoice":
            import sherpa_onnx

            self._model = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=self._model_path,
                tokens=self._tokens_path,
                num_threads=2,
                use_itn=True,
                debug=False,
            )
        elif self._model is None and self._kind == "funasr":
            from funasr import AutoModel

            self._model = AutoModel(model="paraformer-zh", disable_update=True)
        elif self._model is None and self._kind == "whisper":
            import whisper

            self._model = whisper.load_model("base")
        return self._model

    def _run(self, sr: int, samples) -> str:
        self._ensure_model()
        if self._kind == "sherpa_sensevoice":
            stream = self._model.create_stream()
            stream.accept_waveform(sr, samples)
            self._model.decode_stream(stream)
            return (stream.result.text or "").strip()
        # funasr / whisper 需要文件路径：写临时 wav
        import tempfile

        import numpy as np

        tmp = Path(tempfile.mktemp(suffix=".wav"))
        samples16 = (np.asarray(samples, dtype=np.float32) * 32768.0).clip(-32768, 32767).astype(np.int16)
        with wave.open(str(tmp), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(int(sr))
            w.writeframes(samples16.tobytes())
        if self._kind == "funasr":
            res = self._model.generate(input=str(tmp))
            return res[0].get("text", "") if res else ""
        result = self._model.transcribe(str(tmp))
        return result.get("text", "").strip()

    def transcribe(self, audio) -> dict:
        if not audio:
            return {"text": ""}
        if not self.available:
            return {"text": ""}  # 真实引擎不可用时返回空，不返回固定 mock 文本冒充识别结果
        try:
            normalized = _normalize_audio(audio)
            if normalized is None:
                logger.warning("[ASR] 无法解析音频输入：type=%s", type(audio).__name__)
                return {"text": ""}
            sr, samples = normalized
            logger.info(
                "[ASR] 收到音频 | 原始类型=%s | sample_rate=%s | 样本长度=%s | dtype=%s",
                type(audio).__name__, sr, samples.shape[0], samples.dtype,
            )
            text = self._run(sr, samples)
            return {"text": text or ""}  # 识别为空/失败时返回空，不冒充识别结果
        except Exception as e:  # noqa: BLE001
            logger.warning("[ASR] 转写异常：%s", e)
            return {"text": ""}

    def transcribe_stream(self, audio_stream) -> dict:
        return self.transcribe(audio_stream)


class MockTTSBackend(TTSBackend):
    """mock TTS：不真正合成语音，生成短促占位提示音。"""

    backend_name = "mock"
    status = "mock模式（TTS未连接）"

    def synthesize(self, text) -> dict:
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = _OUTPUT_DIR / f"tts_{ts}.wav"
        # 时长随文本长度微调，演示 text 被消费
        duration = min(3.0, max(0.5, len(text or "") / 15.0))
        _generate_mock_wav(path, duration=duration)
        return {"audio_path": str(path)}


class EdgeTTSBackend(TTSBackend):
    """真实 TTS 后端（预留）：Edge-TTS 微软在线语音合成。

    - 已安装 edge-tts 且网络可用时：available=True，生成真实语音（mp3）。
    - 未安装 / 合成失败时：自动回退 MockTTSBackend 的占位 wav。
    """

    backend_name = "edge_tts"
    status = "真实 TTS（Edge-TTS）"

    def __init__(self) -> None:
        self.available = False
        try:
            import importlib.util

            if importlib.util.find_spec("edge_tts") is not None:
                self.available = True
        except Exception:
            self.available = False

    def synthesize(self, text) -> dict:
        if not text:
            return {"audio_path": ""}
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")

        if self.available:
            mp3 = _OUTPUT_DIR / f"tts_{ts}.mp3"
            try:
                self._synthesize_edge(text, mp3)
                if mp3.is_file() and mp3.stat().st_size > 0:
                    return {"audio_path": str(mp3)}
            except Exception:
                pass

        # 回退 mock 占位 wav
        wav = _OUTPUT_DIR / f"tts_{ts}.wav"
        _generate_mock_wav(wav, duration=min(3.0, max(0.5, len(text) / 15.0)))
        return {"audio_path": str(wav)}

    def _synthesize_edge(self, text: str, path: Path) -> None:
        import asyncio

        import edge_tts

        async def _save() -> None:
            communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
            await communicate.save(str(path))

        asyncio.run(_save())


def _generate_mock_wav(path: Path, duration: float = 0.8, freq: float = 440.0) -> None:
    """生成短促提示音（mock TTS 占位音频，非真实语音合成）。"""
    framerate = 16000
    n = int(framerate * duration)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(framerate)
        frames = bytearray()
        for i in range(n):
            fade = 1.0 - (i / n)
            sample = int(0.25 * 32767 * math.sin(2 * math.pi * freq * i / framerate) * fade)
            frames += struct.pack("<h", sample)
        w.writeframes(bytes(frames))


def _build_default_asr() -> ASRBackend:
    """构建默认 ASR 后端：真实 ASR（FunASR/Whisper）可用则用之，否则回退 mock。"""
    try:
        backend = FunASRBackend()
        if backend.available:
            return backend
    except Exception:
        pass
    return MockASRBackend()


def _build_default_tts() -> TTSBackend:
    """构建默认 TTS 后端：真实 TTS（Edge-TTS）可用则用之，否则回退 mock。"""
    try:
        backend = EdgeTTSBackend()
        if backend.available:
            return backend
    except Exception:
        pass
    return MockTTSBackend()


# 默认后端：ASR/TTS 均优先真实（FunASR/Whisper、Edge-TTS），不可用回退 mock。
_default_asr: ASRBackend = _build_default_asr()
_default_tts: TTSBackend = _build_default_tts()


def audio_to_text(audio) -> dict:
    """对外统一入口：语音识别（ASR，文件）。

    参数：audio 用户上传的音频（文件路径或字节；mock 阶段不实际解析）。
    返回：{"text": 识别文本}
    """
    return _default_asr.transcribe(audio)


def microphone_to_text(audio_stream) -> dict:
    """对外统一入口：实时麦克风语音识别（ASR streaming，当前 mock）。

    参数：audio_stream 麦克风音频流（文件路径 / 采样数组；mock 阶段不实际解析）。
    返回：{"text": 识别文本}
    """
    return _default_asr.transcribe_stream(audio_stream)


def text_to_audio(text) -> dict:
    """对外统一入口：语音合成（TTS）。

    参数：text 待合成文本（Agent 生成的回答）。
    返回：{"audio_path": 音频文件路径}
    """
    return _default_tts.synthesize(text)


def get_speech_status() -> str:
    """返回当前语音模块状态（ASR / TTS）。"""
    return f"{_default_asr.status} / {_default_tts.status}"


def get_asr_status() -> str:
    """返回当前 ASR 运行模式：真实识别（SenseVoice/FunASR/Whisper）或 Mock fallback。"""
    if getattr(_default_asr, "available", False):
        return "真实识别（SenseVoice/FunASR/Whisper）"
    return "Mock fallback（ASR 未连接）"


def get_tts_status() -> str:
    """返回当前 TTS 运行模式：真实语音（Edge-TTS）或 Mock fallback。"""
    if getattr(_default_tts, "available", False):
        return "真实语音（Edge-TTS）"
    return "Mock fallback（TTS 未连接）"
