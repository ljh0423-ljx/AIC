# -*- coding: utf-8 -*-
"""
inference — 农业病害检测系统（PP-YOLOE+-m）推理模块。

模块结构：
    config.py        路径 / 13 类映射 / 默认参数（全部 Path，不写死 Windows 路径）
    detector.py      后端抽象 + PaddleBackend（CPU/GPU），预留 RKNN 后端接口
    postprocess.py   统一 Detection 结构、过滤、统计、JSON/CSV/TXT
    visualize.py     可视化（原图比例、中英文标注、真实耗时）
    batch_infer.py   批量推理引擎（失败隔离、时间戳目录、真实 FPS）
    infer.py         命令行入口

用法：
    python inference/infer.py --image xxx.jpg
    python inference/infer.py --dir xxx_folder --conf 0.5
"""

__version__ = "1.0.0"
