# -*- coding: utf-8 -*-
"""
web — 农业病害智能检测系统 Web 展示层（Gradio）
==============================================
模块结构：
    app.py        程序入口，只负责启动（环境变量 / 兼容修补 / 自检 / 启动服务器）
    ui.py         Gradio Blocks 页面组件构建 + 事件绑定
    handlers.py   检测 / 上传 / 清空 / 设备切换等事件处理 + 结果回读
    demo.py       比赛演示模式逻辑
    charts.py     matplotlib 图表生成（类别柱状图 / 空状态占位图）
    render.py     静态 HTML/CSS 与 markdown 卡片生成
    assets.py     图片暂存与演示资源解析
    backend.py    DetectorBackend 单例 + 启动自检（Web 层不直接依赖 Paddle）
    config.py     UI 常量 + 固定模型性能记录（复用 inference/config.py 路径）
    services/     服务层预留包（暂为空）

启动：
    python web/app.py            # 默认 http://127.0.0.1:7860
    web\\run_web.bat             # 已强制 PYTHONUTF8=1

本模块仅做 Web 展示层与系统工程化，不改变模型推理逻辑；
推理复用 inference/ 已验证的 DetectorBackend / PaddleBackend / postprocess /
visualize / batch_infer。
"""

__version__ = "1.0.0"
