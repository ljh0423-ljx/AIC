# -*- coding: utf-8 -*-
"""
web.services — 服务层
======================
承载与展示/检测解耦的可扩展服务模块：

    severity.py    病害严重程度评估（阶段1：基于检测框面积占比的规则评估；
                   预留病斑分割模型 / 深度学习严重程度分类模型替换接口）

    vlm.py         VLM 视觉语言诊断（阶段2：mock 模拟返回；预留 Qwen-VL /
                   InternVL / GPT-4o Vision 等真实 VLM 接入接口）

    rag.py         农业知识增强 RAG 诊断（阶段3：本地 mock 知识库；预留
                   FAISS / Chroma / Milvus / Elasticsearch 等向量库接入接口）

    agent.py       农业智能决策 Agent（阶段4：mock 规则决策；预留 DeepSeek /
                   GPT API / Qwen-Agent / LangGraph 等 LLM 接入接口）

    speech.py      农业智能助手语音交互（阶段5：ASR + TTS mock；预留
                   Whisper / FunASR（ASR）与 Edge-TTS / CosyVoice（TTS）接入）

    robot.py       SO-101（LeRobot）视觉引导机械臂执行（阶段6.1：mock 控制；
                   预留 Hugging Face LeRobot / SO-101 follower API 接入）

    camera.py      实时摄像头视觉检测（阶段7.1：Gradio webcam 单帧，复用 _detect；
                   预留 RK3588 / USB 工业相机等真实摄像头后端）

新增服务模块时，保持纯函数 / 无状态、不依赖 Paddle 与 inference 内部实现，
便于未来替换底层算法而不影响 Web 展示层。
"""
