# -*- coding: utf-8 -*-
"""
web — 农业病害智能检测系统 Web 展示层（Gradio）
==============================================
模块结构：
    config.py     UI 常量 + 固定模型性能记录（复用 inference/config.py 路径）
    backend.py    DetectorBackend 单例 + 启动自检（Web 层不直接依赖 Paddle）
    app.py        Gradio 界面与事件

启动：
    python web/app.py            # 默认 http://127.0.0.1:7860

本模块仅做 Web 展示层与系统工程化，不改变模型推理逻辑；
推理复用 inference/ 已验证的 DetectorBackend / PaddleBackend / postprocess /
visualize / batch_infer。
"""

__version__ = "1.0.0"
