# -*- coding: utf-8 -*-
"""
detector.py — 检测后端封装（模型加载/推理与图片读取、结果处理、可视化解耦）
============================================================================
设计要点：
  1. `DetectorBackend` 为抽象基类，定义统一接口；当前实现 `PaddleBackend`（CPU/GPU），
     预留 GPU / NPU / RKNN 后端替换能力。后续 RK3588 只需新增 `RKNNBackend`
     并实现同一套接口，**无需重写整个系统**。
  2. 模型**只加载一次**（load()），批量推理复用同一实例，不重复加载。
  3. 本模块只接收已解码的 BGR 图像数组（图片读取由 batch_infer/infer 负责），
     输出统一的 `Detection` 列表（见 postprocess.py）。
  4. 不使用 PaddleDetection 源码修改；数据集路径通过 `-o` 覆盖注入（绝对路径），
     不依赖软链、不使用 test.json。
  5. 错误统一抛出带明确中文提示的自定义异常，交由上层捕获给出中文信息，而非裸 Traceback。

本阶段（Windows + Paddle 3.3.0 + CPU）已验证与官方 tools/infer.py 结果逐位一致。
"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

from . import config as app_config
from .postprocess import Detection

logger = logging.getLogger("agri.infer.detector")


# ---------------------------------------------------------------------------
# 自定义异常（中文信息，供上层直接展示）
# ---------------------------------------------------------------------------
class DetectorError(Exception):
    """检测后端错误基类。"""


class ModelNotFoundError(DetectorError):
    pass


class ConfigNotFoundError(DetectorError):
    pass


class PaddleNotReadyError(DetectorError):
    pass


class PaddleVersionError(DetectorError):
    pass


class InferenceError(DetectorError):
    pass


# ---------------------------------------------------------------------------
# 后端注册机制
# ---------------------------------------------------------------------------
_BACKEND_REGISTRY: dict[str, type] = {}


def register_backend(name: str):
    """后端注册装饰器：将后端类注册到 create_backend() 可用的名字下。

    例如：
        @register_backend("paddle_cpu")
        class PaddleBackend(DetectorBackend): ...

    后续 RK3588 只需实现 RKNNBackend 并注册为 "rknn_reserved"（或正式名 "rknn"），
    Web/推理/统计模块无需改动。
    """

    def deco(cls):
        _BACKEND_REGISTRY[name] = cls
        return cls

    return deco


def list_backends() -> list[str]:
    """返回当前已注册的后端名列表（供 UI 展示，如 paddle_cpu/paddle_gpu/rknn_reserved）。"""
    return sorted(_BACKEND_REGISTRY.keys())


def _ensure_gpu_available() -> None:
    """强制 GPU 前校验：Paddle 编译 CUDA 且有可用 GPU，否则抛出明确中文错误。"""
    try:
        import paddle

        if not paddle.is_compiled_with_cuda():
            raise DetectorError(
                "[设备错误] 当前 PaddlePaddle 未编译 CUDA，无法使用 GPU。\n"
                "请安装 GPU 版 PaddlePaddle（paddlepaddle-gpu），或改用 --device auto / --device cpu。"
            )
        if paddle.device.cuda.device_count() <= 0:
            raise DetectorError(
                "[设备错误] 未检测到可用 GPU（CUDA device count = 0）。\n"
                "请确认服务器已配置 GPU，或改用 --device auto / --device cpu。"
            )
    except DetectorError:
        raise
    except Exception as e:  # noqa: BLE001
        raise DetectorError(f"[设备错误] GPU 检测失败：{e}") from e


def resolve_device(mode: str = "auto") -> str:
    """解析推理设备模式 -> 'cpu' / 'gpu'。

    mode:
      - "auto": 启动时自动检测 —— Paddle 编译 CUDA 且有可用 GPU → "gpu"，否则 → "cpu"
      - "cpu" : 强制 CPU
      - "gpu" : 强制 GPU；无可用 GPU 时抛明确中文错误
    上层（Web/CLI）只需传模式，无需在调用处判断设备。
    """
    mode = (mode or "auto").lower()
    if mode == "cpu":
        return "cpu"
    if mode == "gpu":
        _ensure_gpu_available()
        return "gpu"
    # auto
    try:
        import paddle

        if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return "gpu"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def get_gpu_info() -> str:
    """返回可用 GPU 型号（如 'NVIDIA GeForce RTX 4090'）；无 GPU 返回空字符串（不虚构）。"""
    try:
        import paddle

        if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            name = paddle.device.cuda.get_device_name(0)
            if name:
                return str(name)
    except Exception:  # noqa: BLE001
        pass
    return ""


# ---------------------------------------------------------------------------
# 后端抽象基类
# ---------------------------------------------------------------------------
class DetectorBackend(ABC):
    """统一后端接口：后续 RKNNBackend 实现同一套方法即可无缝替换。

    约定接口：load() / predict() / predict_batch() / info() / close()。
    兼容旧名 infer() / infer_batch()（与 predict 行为一致）。
    输出统一为 postprocess.Detection 数据结构，与后端无关。
    """

    backend_name: str = "base"

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    @abstractmethod
    def load(self) -> None:
        """加载模型与映射（只调用一次）。"""

    @abstractmethod
    def predict(self, image_bgr: np.ndarray) -> list[Detection]:
        """对单张 BGR 图像推理，返回 Detection 列表（未过滤阈值）。"""

    @abstractmethod
    def predict_batch(self, images_bgr: list[np.ndarray]) -> list[list[Detection]]:
        """对多张 BGR 图像批量推理（模型不重复加载），返回逐图 Detection 列表。"""

    def infer(self, image_bgr: np.ndarray) -> list[Detection]:
        """兼容别名：等价于 predict()。"""
        return self.predict(image_bgr)

    def infer_batch(self, images_bgr: list[np.ndarray]) -> list[list[Detection]]:
        """兼容别名：等价于 predict_batch()。"""
        return self.predict_batch(images_bgr)

    @property
    def catid2name(self) -> dict[int, str]:
        """COCO 类别 id -> 英文名映射（与后端无关，由类别映射文件决定）。"""
        return getattr(self, "_catid2name", {})

    def info(self) -> dict:
        """返回后端/模型运行信息（供 UI 与统计模块展示，不触发加载）。"""
        return {
            "backend": self.backend_name,
            "device": self.device,
            "model_name": "PP-YOLOE+-m",
            "input_size": getattr(self, "input_size", None),
            "num_classes": len(self.catid2name),
            "paddle_version": getattr(self, "_paddle_version", ""),
            "model_path": str(getattr(self, "model_path", "")),
            "config_path": str(getattr(self, "config_path", "")),
            "dataset_dir": str(getattr(self, "dataset_dir", "")),
            "anno_path": str(getattr(self, "anno_path", "")),
        }

    @abstractmethod
    def close(self) -> None:
        """释放资源。"""


# ---------------------------------------------------------------------------
# Paddle / PP-YOLOE+-m 后端（默认后端，注册为 paddle_cpu / paddle_gpu）
# ---------------------------------------------------------------------------
@register_backend("paddle_cpu")
@register_backend("paddle_gpu")
class PaddleBackend(DetectorBackend):
    backend_name = "paddle"

    def __init__(
        self,
        model_path: Path | str,
        config_path: Path | str,
        dataset_dir: Path | str,
        anno_rel_path: str,
        device: str = "cpu",
        input_size: Optional[int] = None,
    ) -> None:
        super().__init__(device=device)
        # 描述性后端名：paddle_cpu / paddle_gpu（用于 UI 与统计展示）
        self.backend_name = "paddle_cpu" if device != "gpu" else "paddle_gpu"
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.dataset_dir = Path(dataset_dir)
        self.anno_rel_path = anno_rel_path
        self.anno_path = self.dataset_dir / anno_rel_path
        self.input_size = int(input_size or app_config.DEFAULT_INPUT_SIZE)

        self._model = None
        self._transforms = None
        self._clsid2catid: dict[int, int] = {}
        self._catid2name: dict[int, str] = {}
        self._catid2cn: dict[int, str] = {}
        self._paddle_version: str = ""
        self._loaded = False

    # ---------------- 加载 ----------------
    def load(self) -> None:
        if self._loaded:
            return

        if not self.model_path.is_file():
            raise ModelNotFoundError(
                f"[模型错误] 模型文件不存在：{self.model_path}\n"
                f"请确认最终模型 best_model.pdparams 路径是否正确。"
            )
        if not self.config_path.is_file():
            raise ConfigNotFoundError(
                f"[配置错误] 配置文件不存在：{self.config_path}\n"
                f"请确认 PP-YOLOE+-m 配置路径是否正确。"
            )
        if not self.dataset_dir.is_dir():
            raise ConfigNotFoundError(
                f"[数据错误] 数据集目录不存在：{self.dataset_dir}\n"
                f"本系统使用 dataset/processed_detection_exiffix（禁止使用 test.json / TEST 图片）。"
            )
        if not self.anno_path.is_file():
            raise ConfigNotFoundError(
                f"[数据错误] 类别映射标注文件不存在：{self.anno_path}\n"
                f"类别映射必须取自项目已有 val.json（非 TEST）。"
            )

        paddle = self._import_paddle()
        self._paddle_version = paddle.__version__
        self._check_paddle_version(paddle)
        self._setup_device(paddle)

        # ---- 构建配置（-o 绝对路径覆盖，与已验证的工具链一致）----
        from ppdet.core.workspace import create, load_config, merge_config

        cfg = load_config(str(self.config_path))
        overrides = {
            "weights": str(self.model_path),
            "use_gpu": (self.device == "gpu"),
            "TestDataset": {
                "dataset_dir": str(self.dataset_dir),
                "anno_path": self.anno_rel_path,
                "image_dir": "images/val",
            },
        }
        merge_config(overrides)
        if "TestReader" in cfg and "inputs_def" in cfg["TestReader"]:
            try:
                shape = cfg["TestReader"]["inputs_def"].get("image_shape", [])
                if len(shape) >= 2:
                    self.input_size = int(shape[1])
            except Exception:
                pass  # 取不到就保持默认 640

        # ---- 构建模型 + 加载权重（只一次）----
        try:
            model = create(cfg["architecture"])
        except Exception as e:  # noqa: BLE001
            raise InferenceError(
                f"[模型错误] 根据配置 {self.config_path.name} 构建模型失败：{e}"
            ) from e
        try:
            state = paddle.load(str(self.model_path))
            missing, unexpected = model.set_dict(state)
        except Exception as e:  # noqa: BLE001
            raise InferenceError(
                f"[模型错误] 加载权重失败：{e}\n"
                f"请确认 best_model.pdparams 与配置文件架构一致（PP-YOLOE+-m）。"
            ) from e
        if missing:
            raise InferenceError(
                f"[模型错误] 权重与模型结构不匹配，缺失 {len(missing)} 个键："
                f"{missing[:8]} ...\n请勿混用其他 checkpoint。"
            )
        model.eval()
        self._model = model

        # ---- 预处理算子链（与官方 reader 逐项一致，已逐位验证）----
        from ppdet.data.transform.operators import (
            NormalizeImage,
            Permute,
            Resize,
        )

        self._transforms = [
            Resize(
                target_size=[self.input_size, self.input_size],
                keep_ratio=False,
                interp=2,
            ),
            NormalizeImage(
                mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], norm_type="none"
            ),
            Permute(),
        ]

        # ---- 类别映射（严格取自 val.json）----
        from .config import build_class_mapping

        self._catid2name, self._clsid2catid = build_class_mapping(self.anno_path)
        self._catid2cn = dict(app_config.CLASS_NAMES_CN)

        self._loaded = True
        logger.info(
            "PaddleBackend 已就绪 | device=%s | paddle=%s | 13类映射来自 %s",
            self.device,
            self._paddle_version,
            self.anno_path.name,
        )

    def _import_paddle(self):
        try:
            import paddle
        except Exception as e:  # noqa: BLE001
            raise PaddleNotReadyError(
                f"[环境错误] 未能导入 PaddlePaddle：{e}\n"
                f"请先安装依赖：pip install -r requirements_infer.txt"
            ) from e
        return paddle

    def _check_paddle_version(self, paddle) -> None:
        try:
            major = int(paddle.__version__.split(".")[0])
        except Exception:  # noqa: BLE001
            major = 0
        if major < 2:
            raise PaddleVersionError(
                f"[环境错误] PaddlePaddle 版本过低（当前 {paddle.__version__}），"
                f"PP-YOLOE+ 推理需要 >= 2.4。请按 requirements_infer.txt 安装。"
            )
        if major not in (2, 3):
            logger.warning(
                "检测到 PaddlePaddle %s（主版本 %d），官方验证环境为 2.6.2。"
                "已尽力兼容，若结果异常请核对版本。",
                paddle.__version__,
                major,
            )

    def _setup_device(self, paddle) -> None:
        if self.device == "gpu":
            try:
                count = paddle.device.cuda.device_count()
            except Exception:  # noqa: BLE001
                count = 0
            if count <= 0:
                logger.warning(
                    "[环境提示] 请求 GPU 推理但未检测到可用 GPU，自动回退到 CPU。"
                )
                self.device = "cpu"
        try:
            paddle.set_device(self.device)
        except Exception as e:  # noqa: BLE001
            raise DetectorError(
                f"[环境错误] 无法设置推理设备 {self.device}：{e}"
            ) from e

    # ---------------- 预处理 ----------------
    def _preprocess(self, image_bgr: np.ndarray) -> dict:
        """BGR -> RGB -> Resize/Normalize/Permute，返回模型输入 dict。"""
        if image_bgr is None or image_bgr.size == 0:
            raise InferenceError("输入图像为空（解码失败）。")
        rgb = image_bgr
        if rgb.ndim == 3 and rgb.shape[2] == 3:
            # 假定输入为 BGR（cv2 读取约定），转 RGB 与模型一致
            rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        h, w = rgb.shape[:2]
        sample = {
            "image": rgb,
            "im_shape": np.array([h, w], dtype=np.float32),
            "scale_factor": np.array([1.0, 1.0], dtype=np.float32),
        }
        for op in self._transforms:
            sample = op(sample)
        return sample

    # ---------------- 推理（统一接口 predict / predict_batch）----------------
    def predict(self, image_bgr: np.ndarray) -> list[Detection]:
        if not self._loaded:
            raise DetectorError("[模型错误] 模型尚未加载，请先调用 load()。")
        try:
            sample = self._preprocess(image_bgr)
        except InferenceError:
            raise
        except Exception as e:  # noqa: BLE001
            raise InferenceError(f"[推理错误] 图像预处理失败：{e}") from e

        paddle = sys.modules.get("paddle")
        x = paddle.to_tensor(sample["image"][None].astype("float32"))
        sf = paddle.to_tensor(sample["scale_factor"][None].astype("float32"))
        im_shape = paddle.to_tensor(sample["im_shape"][None].astype("float32"))
        with paddle.no_grad():
            outs = self._model({"image": x, "scale_factor": sf, "im_shape": im_shape})
        return self._decode(outs, image_names=[""])[0]

    def predict_batch(self, images_bgr: list[np.ndarray]) -> list[list[Detection]]:
        """批量推理：模型只加载一次；逐图预处理，一次前向。"""
        if not self._loaded:
            raise DetectorError("[模型错误] 模型尚未加载，请先调用 load()。")

        n = len(images_bgr)
        if n == 0:
            return []
        samples = [self._preprocess(img) for img in images_bgr]

        paddle = sys.modules.get("paddle")
        x = paddle.to_tensor(
            np.stack([s["image"].astype("float32") for s in samples])
        )
        sf = paddle.to_tensor(
            np.stack([s["scale_factor"].astype("float32") for s in samples])
        )
        im_shape = paddle.to_tensor(
            np.stack([s["im_shape"].astype("float32") for s in samples])
        )
        with paddle.no_grad():
            outs = self._model({"image": x, "scale_factor": sf, "im_shape": im_shape})
        return self._decode(outs, image_names=[""] * n)

    def _decode(
        self, outs, image_names: list[str]
    ) -> list[list[Detection]]:
        """把模型输出解码为逐图 Detection 列表。

        模型输出 bbox 格式：[M,6] 每行 [cls_idx, score, xmin, ymin, xmax, ymax]，
        bbox_num 按图分片；坐标已被 scale_factor 还原到原图。
        """
        try:
            bbox = outs["bbox"].numpy()
        except Exception:  # noqa: BLE001
            bbox = np.asarray(outs["bbox"])
        bbox_num = np.asarray(outs["bbox_num"], dtype=np.int64)

        from ppdet.metrics.coco_utils import get_infer_results

        n = len(image_names)
        if n == 0:
            return []
        if bbox is None or len(bbox) == 0:
            return [[] for _ in range(n)]

        im_id = np.array([[i] for i in range(n)], dtype=np.int64)
        infer_res = get_infer_results(
            {"bbox": bbox, "bbox_num": bbox_num, "im_id": im_id},
            self._clsid2catid,
        )
        det_by_image: list[list[Detection]] = [[] for _ in range(n)]
        for rec in infer_res.get("bbox", []):
            img_idx = int(rec["image_id"])
            catid = int(rec["category_id"])
            x0, y0, w, h = rec["bbox"]
            name = self._catid2name.get(catid, f"class_{catid}")
            det_by_image[img_idx].append(
                Detection(
                    image_name=image_names[img_idx] if img_idx < n else "",
                    class_id=catid,
                    class_name=name,
                    confidence=float(rec["score"]),
                    x=float(x0),
                    y=float(y0),
                    width=float(w),
                    height=float(h),
                )
            )
        return det_by_image

    def close(self) -> None:
        self._model = None
        self._transforms = None
        self._loaded = False


# ---------------------------------------------------------------------------
# RKNN 后端占位（RK3588 迁移预留接口，本阶段不实现）
# ---------------------------------------------------------------------------
@register_backend("rknn_reserved")
class RKNNBackend(DetectorBackend):
    """RK3588 / RKNN 后端占位。

    仅作为接口占位与迁移说明，**本阶段未实现**，禁止声称已完成 RK3588 部署。
    后续实现时：完成 Paddle→ONNX→RKNN 转换链路，实现本类的
    load()/predict()/predict_batch()/info()/close()，输出统一 Detection，
    Web/推理/统计模块无需改动。
    """

    backend_name = "rknn_reserved"

    def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise NotImplementedError(
            "[后端占位] RKNNBackend 尚未实现（预留 RK3588 迁移接口）。\n"
            "当前系统仅支持 paddle_cpu / paddle_gpu；"
            "RK3588 部署需另行完成模型转换与后端实现。"
        )

    # ---- 接口占位（本类仅在 create_backend 中被识别，不会被实例化使用）----
    def load(self) -> None:  # pragma: no cover
        raise NotImplementedError("RKNNBackend 占位：load 未实现。")

    def predict(self, image_bgr):  # noqa: ANN001 - pragma: no cover
        raise NotImplementedError("RKNNBackend 占位：predict 未实现。")

    def predict_batch(self, images_bgr):  # noqa: ANN001 - pragma: no cover
        raise NotImplementedError("RKNNBackend 占位：predict_batch 未实现。")

    def close(self) -> None:  # pragma: no cover
        raise NotImplementedError("RKNNBackend 占位：close 未实现。")


# ---------------------------------------------------------------------------
# 后端工厂：通过注册表创建，后续新增 RKNN 后端只需注册，无需改本函数
# ---------------------------------------------------------------------------
def create_backend(
    kind: str = "paddle",
    *,
    model_path=None,
    config_path=None,
    dataset_dir=None,
    anno_rel_path=None,
    device: str = "cpu",
    input_size: Optional[int] = None,
) -> DetectorBackend:
    """后端工厂（注册表驱动）。

    kind:
      - "paddle"       : 别名，按 device 解析为 paddle_cpu / paddle_gpu
      - "paddle_cpu"   : PaddlePaddle CPU（默认）
      - "paddle_gpu"   : PaddlePaddle GPU（预留接口，无 GPU 自动回退 CPU）
      - "rknn_reserved": RK3588/RKNN 占位（本阶段未实现，调用即提示）
    """
    if kind == "paddle":
        kind = "paddle_cpu" if device != "gpu" else "paddle_gpu"

    if kind not in _BACKEND_REGISTRY:
        raise DetectorError(
            f"[配置错误] 未知推理后端：{kind}。当前支持 "
            f"paddle_cpu / paddle_gpu；rknn_reserved 为预留接口，本阶段未实现。"
        )

    # device 由后端名决定（paddle_cpu->cpu，paddle_gpu->gpu），除非显式传入
    resolved_device = device
    if kind == "paddle_cpu":
        resolved_device = "cpu"
    elif kind == "paddle_gpu":
        resolved_device = "gpu"

    cls = _BACKEND_REGISTRY[kind]
    try:
        return cls(
            model_path=model_path,
            config_path=config_path,
            dataset_dir=dataset_dir,
            anno_rel_path=anno_rel_path,
            device=resolved_device,
            input_size=input_size,
        )
    except NotImplementedError as e:
        raise DetectorError(str(e)) from e
