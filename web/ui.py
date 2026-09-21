# -*- coding: utf-8 -*-
"""
ui.py — Gradio Blocks 页面组件构建
====================================
build_ui() 负责：
  - 页面组件树（Hero/流程图/左右双栏/结果 Tab/折叠技术信息）
  - 事件绑定（检测/选图/上传/清空/设备切换/演示模式）

组件装配在此，具体逻辑在 handlers / demo / html / assets / charts / backend。
本模块只做「搭积木」，不直接推理、不读文件。
"""

from __future__ import annotations

import gradio as gr

from web import backend
from web import config as web_config
from web import demo as demo_mod
from web import handlers
from web.assets import DEMO_CASES, _demo_gallery_items, _friendly_case_name
from web.render import (
    CSS,
    _class_empty_initial,
    _DET_HEADERS,
    _edge_status_markdown,
    _IMG_HEADERS,
    _render_sys_info,
    _upload_placeholder,
    flow_html,
    hero_html,
)


def build_ui(self_check_result: list[dict] | None = None) -> gr.Blocks:
    info = backend.detector_info()

    with gr.Blocks(
        title="AI 农业病害智能检测系统",
        theme=gr.themes.Base(primary_hue="green", neutral_hue="slate").set(
            # —— 深绿色农业科技统一配色（浅色模式，替换 slate 蓝灰中性色）——
            body_text_color="#EAF4EC",
            body_text_color_subdued="#9db8a6",
            background_fill_primary="#0d1b13",
            background_fill_secondary="#122018",
            block_background_fill="#14251D",
            block_border_color="#365943",
            block_label_text_color="#9db8a6",
            block_title_text_color="#9db8a6",
            border_color_primary="#365943",
            border_color_accent="#5FA878",
            input_background_fill="#20382B",
            input_background_fill_focus="#20382B",
            input_background_fill_hover="#20382B",
            input_border_color="#365943",
            input_border_color_focus="#5FA878",
            input_border_color_hover="#5FA878",
            input_placeholder_color="#7d9d88",
            panel_background_fill="#14251D",
            panel_border_color="#365943",
            link_text_color="#5FA878",
            link_text_color_hover="#7ee99a",
            button_secondary_background_fill="#20382B",
            button_secondary_background_fill_hover="#2c4a35",
            button_secondary_border_color="#365943",
            button_secondary_border_color_hover="#5FA878",
            button_secondary_text_color="#EAF4EC",
            table_odd_background_fill="#14251D",
            table_even_background_fill="#122018",
            table_border_color="#365943",
            table_text_color="#EAF4EC",
            checkbox_background_color="#20382B",
            checkbox_border_color="#365943",
            checkbox_border_color_focus="#5FA878",
            # —— 交互高亮（模式无关）——
            color_accent="#5FA878",
            color_accent_soft="#20382B",
            color_accent_soft_dark="#20382B",
            slider_color="#5FA878",
            # —— 深绿色（暗色模式，系统暗色下同样绿色）——
            body_text_color_dark="#EAF4EC",
            body_text_color_subdued_dark="#9db8a6",
            background_fill_primary_dark="#0d1b13",
            background_fill_secondary_dark="#122018",
            block_background_fill_dark="#14251D",
            block_border_color_dark="#365943",
            block_label_text_color_dark="#9db8a6",
            block_title_text_color_dark="#9db8a6",
            border_color_primary_dark="#365943",
            border_color_accent_dark="#5FA878",
            input_background_fill_dark="#20382B",
            input_background_fill_focus_dark="#20382B",
            input_background_fill_hover_dark="#20382B",
            input_border_color_dark="#365943",
            input_border_color_focus_dark="#5FA878",
            input_border_color_hover_dark="#5FA878",
            input_placeholder_color_dark="#7d9d88",
            panel_background_fill_dark="#14251D",
            panel_border_color_dark="#365943",
            link_text_color_dark="#5FA878",
            link_text_color_hover_dark="#7ee99a",
            button_secondary_background_fill_dark="#20382B",
            button_secondary_background_fill_hover_dark="#2c4a35",
            button_secondary_border_color_dark="#365943",
            button_secondary_border_color_hover_dark="#5FA878",
            button_secondary_text_color_dark="#EAF4EC",
            table_odd_background_fill_dark="#14251D",
            table_even_background_fill_dark="#122018",
            table_border_color_dark="#365943",
            table_text_color_dark="#EAF4EC",
            checkbox_background_color_dark="#20382B",
            checkbox_border_color_dark="#365943",
            checkbox_border_color_focus_dark="#5FA878",
        ),
        css=CSS,
    ) as demo:
        # ---------- 头部（主视觉 + 4 核心徽章，无开发环境信息） ----------
        with gr.Row():
            gr.HTML(hero_html())

        # ---------- AI 检测流程 ----------
        gr.HTML(flow_html())

        with gr.Row(equal_height=True, elem_classes="app-main"):
            # ============ 左列：检测操作卡（上传/阈值/按钮） ============
            with gr.Column(scale=1.6, elem_classes="card"):
                gr.Markdown("### ① 图片上传\n上传后直接预览作物图片缩略图。")
                # 隐藏 File 输入（检测数据流不变）；UploadButton + Gallery 作为主要界面
                upload = gr.Files(
                    label="上传图片（隐藏输入）",
                    file_types=["image"], type="filepath", file_count="multiple",
                    visible=False,
                )
                upload_btn = gr.UploadButton(
                    "📷 将作物图片拖拽到此处或点击选择",
                    file_types=["image"], file_count="multiple", type="filepath",
                    elem_classes="upload-btn",
                )
                upload_count = gr.Markdown(_upload_placeholder())
                preview_gallery = gr.Gallery(
                    label=None, columns=3, height=280, object_fit="contain",
                )
                conf = gr.Slider(
                    minimum=0.05, maximum=0.95, value=web_config.DEFAULT_CONF_THRESHOLD,
                    step=0.05, label="推理置信度阈值（默认 0.5）",
                    info="检测阈值，不改变模型参数。",
                )
                with gr.Row():
                    detect_btn = gr.Button("🚀 开始 AI 检测", variant="primary")
                    clear_btn = gr.Button("🗑 清空任务", variant="secondary")
                run_state = gr.State(value=None)
                session_state = gr.State(value=None)  # 会话级检测/VLM/RAG/Agent 上下文（隔离）

            # ============ 右列：系统信息卡（顶部对齐、底部随 Grid 拉伸） ============
            with gr.Column(scale=1, elem_classes="card"):
                sys_info_md = gr.Markdown(_render_sys_info(info))
                device_sel = gr.Dropdown(
                    ["auto", "cpu", "gpu"], value=backend.get_device_mode(),
                    label="推理设备（auto / cpu / gpu）",
                    info="auto=启动时自动检测；cpu=强制CPU；gpu=强制GPU（不可用报错）。切换需重新加载模型一次。",
                )
                device_note = gr.Markdown("")
                gr.Markdown(_edge_status_markdown())
                gr.Markdown(
                    "### 模型评测\n"
                    f"- VAL mAP@0.5:0.95 = **{web_config.MODEL_RECORDS['val_map']}**（模型选择依据）\n"
                    f"- TEST mAP@0.5:0.95 = **{web_config.MODEL_RECORDS['test_map']}**"
                    f"<span style='color:#ef9a9a'>（独立 TEST 最终评估结果）</span>\n"
                    f"- 参数量：**{web_config.MODEL_RECORDS['params_m']} M** · "
                    f"模型大小：**{web_config.MODEL_RECORDS['size_mb']} MB**\n"
                    f"<div class='card-note'>* 以上为固定记录，{web_config.MODEL_RECORDS['test_note']}</div>"
                )
                gr.Markdown(
                    "### 系统介绍\n"
                    "基于 **PP-YOLOE+-m** 实现 **13 类植物病害检测**：支持图片级 / 批量检测、"
                    "结果可视化与统计分析；通过统一 DetectorBackend 接口解耦推理后端，"
                    "为边缘设备（RK3588/NPU）部署预留接口。\n\n"
                    "流程：**图片上传 → AI 病害识别 → 结果可视化 → 统计分析**。"
                )

        # ---------- 实时摄像头检测（复用 _detect 流程） ----------
        gr.Markdown("### 📸 实时视觉检测")
        camera_input = gr.Image(
            label="摄像头输入（点击「开启摄像头」采集单帧）",
            sources=["webcam"],
            type="filepath",
        )
        camera_btn = gr.Button("📸 拍照并检测", variant="primary")

        # ---------- 结果区（全宽 Tab，保留原结构） ----------
        gr.Markdown("### ② 检测结果")
        stats_md = gr.Markdown(
            elem_classes="stats-strip",
            value="**图片** - · **目标数** - · **平均置信度** - · **批次 FPS** -",
        )
        agri_card = gr.Markdown("### 🌱 农业视觉检测结果\n\n（尚未检测）")
        vlm_card = gr.Markdown("### 🤖 AI 视觉诊断\n\n（尚未检测）")
        rag_card = gr.Markdown("### 🌾 农业知识辅助诊断\n\n（尚未检测）")
        agent_card = gr.Markdown("### 🧭 智能决策建议\n\n（尚未检测）")
        with gr.Tabs():
            with gr.Tab("📈 可视化结果"):
                gallery = gr.Gallery(
                    label="检测可视化（原图比例，不拉伸）",
                    columns=3, height=360, object_fit="contain",
                )
                gr.Markdown("选择图片查看 **原图 vs 检测结果** 对比：")
                img_selector = gr.Dropdown(label="选择图片", choices=[], value=None)
                with gr.Row(equal_height=True, elem_classes="img-pair"):
                    orig_img = gr.Image(label="原图", height=320)
                    vis_img = gr.Image(label="AI 检测结果", height=320)
                with gr.Row():
                    per_img_table = gr.Dataframe(label="该图片检测明细", headers=_IMG_HEADERS, interactive=False)
            with gr.Tab("📊 检测结果表格"):
                combined_table = gr.Dataframe(label="全部检测结果", headers=_DET_HEADERS, interactive=False)
            with gr.Tab("◕ 类别统计"):
                # 统一状态源：class_area 单组件 value 切换（有数据=指标卡，无数据=空状态卡）
                class_area = gr.HTML(value=_class_empty_initial())
                with gr.Row(equal_height=False, elem_classes="stats-row"):
                    # 左：柱状图（约 55%）；右：明细表（约 45%），顶部对齐、自然高度（数据少不撑高）
                    with gr.Column(scale=1.1, min_width=0):
                        bar_plot = gr.Plot(show_label=False, visible=False)
                    with gr.Column(scale=0.9, min_width=0):
                        class_stats_table = gr.Dataframe(
                            label="类别统计明细",
                            headers=["类别id", "类别", "图片数", "目标数", "平均置信度", "最高置信度"],
                            interactive=False,
                            value=[["-", "暂无类别统计数据", "-", "-", "-", "-"]],
                        )
            with gr.Tab("📋 批次汇总"):
                summary_md = gr.Markdown("### 批次汇总\n（尚未检测）")
            with gr.Tab("⬇️ 结果下载"):
                gr.Markdown("下载当前批次的检测结果图片、JSON、CSV 与统计文件。所有统计均基于实际推理结果。")
                download_files = gr.Files(label="下载文件", interactive=False)
                gr.Markdown("### 📄 AI 诊断验收报告")
                gr.Markdown("复用当前会话已生成的检测 / VLM / RAG / Agent 结果，不重复检测、不调用付费 API。")
                report_btn = gr.Button("📄 生成验收报告", variant="secondary")
                report_preview = gr.Markdown("（尚未生成报告）")
                report_pdf = gr.File(label="下载 PDF", interactive=False)
            with gr.Tab("🧭 诊断流程图 / 可视化记录图"):
                gr.Markdown("生成本次诊断的完整流程图，复用当前会话已有结果，不重复检测、不调用付费 API。")
                flow_btn = gr.Button("🧭 生成诊断流程图", variant="secondary")
                flow_preview = gr.Image(label="流程图预览", type="filepath", interactive=False, height=800)
                with gr.Row():
                    flow_png = gr.File(label="下载 PNG", interactive=False)
                    flow_pdf = gr.File(label="下载 PDF", interactive=False)
            with gr.Tab("🌿 农业病害场景"):
                gr.Markdown(
                    "基于 **15 张 VAL 演示案例**（番茄 8 / 苹果 4 / 葡萄 3）展示真实检测结果；"
                    "所有置信度均为模型原始输出，未人为提高。"
                )
                demo_gallery = gr.Gallery(
                    label="演示案例概览（原图比例）",
                    columns=5, height=320, object_fit="contain",
                    value=_demo_gallery_items(),
                )
                demo_choices = [
                    (f"{_friendly_case_name(c)} · {c['scene_label']}", c["demo_id"])
                    for c in DEMO_CASES
                ]
                demo_sel = gr.Dropdown(label="选择案例，查看 原图 vs 检测 与详情", choices=demo_choices, value=None)
                with gr.Row(equal_height=True, elem_classes="img-pair"):
                    demo_orig = gr.Image(label="原图", height=300)
                    demo_vis = gr.Image(label="AI 检测结果", height=300)
                demo_detail = gr.Markdown("（选择上方案例查看详情）")
            with gr.Tab("🎬 比赛演示模式"):
                _d0 = demo_mod._demo_case_view(0)
                gr.Markdown(
                    "使用 **web/demo_data** 的 15 张 VAL 演示图（顺序：番茄 → 苹果 → 葡萄 → 密集场景），"
                    "复用已生成结果，模型只加载一次；重新检测仅使用当前演示图片。"
                )
                with gr.Row():
                    demo_prev_btn = gr.Button("◀ 上一张")
                    demo_next_btn = gr.Button("下一张 ▶")
                    demo_play_btn = gr.Button("▶ 自动播放")
                    demo_stop_btn = gr.Button("⏹ 停止演示")
                    demo_redetect_btn = gr.Button("🔄 重新检测当前案例")
                with gr.Row(equal_height=True, elem_classes="img-pair"):
                    demo2_orig = gr.Image(label="演示原图", value=_d0[0], height=340)
                    demo2_vis = gr.Image(label="演示检测结果", value=_d0[1], height=340)
                demo2_info = gr.Markdown(_d0[2])
                demo_index = gr.State(value=0)
                demo_playing = gr.State(value=False)
                demo_timer = gr.Timer(2.0, active=True)

        # ---------- 🎤 实时语音助手（ASR + TTS） ----------
        gr.Markdown("### ③ 🎤 实时语音助手")
        audio_input = gr.Audio(
            label="麦克风语音输入（录音或上传）",
            type="numpy",
            sources=["microphone", "upload"],
        )
        # 会话级缓存：保存「最近一次完成」的录音/上传音频（gr.State，跨浏览器会话隔离）。
        # 麦克风录音值不会可靠持久化到 audio_input.value（独立按钮常读到 None），
        # 故通过 stop_recording/upload 事件把音频存入本 State，按钮改从这里读取。
        recorded_audio = gr.State(value=None)
        voice_btn = gr.Button("🎤 开始语音咨询", variant="primary")
        mic_status = gr.Markdown("麦克风状态：待机（点击麦克风开始监听）")
        voice_text = gr.Markdown("识别文本：-")
        voice_answer = gr.Markdown("AI回答：-")
        asr_status = gr.Markdown("ASR状态：-")
        tts_status = gr.Markdown("TTS状态：-")
        voice_output_status = gr.Markdown("语音输出状态：待合成")
        audio_output = gr.Audio(label="🔊 AI 语音回答", type="filepath", autoplay=True)

        # ---------- SO-101 机械臂执行 ----------
        gr.Markdown("### ④ SO-101 机械臂执行")
        robot_btn = gr.Button("🤖 执行机械臂任务", variant="primary")
        robot_card = gr.Markdown("### 🤖 SO-101 机械臂执行\n\n（尚未执行）")

        # ---------- 系统状态 / 技术信息（折叠） ----------
        with gr.Accordion("系统状态 / 技术信息", open=False):
            gr.Markdown("模型已加载 ✓ · 13 类病害 ✓ · 推理后端：Paddle CPU · RK3588/NPU：部署接口已预留")
            if self_check_result:
                lines = ["**启动自检**", ""]
                for c in self_check_result:
                    mark = "✅" if c["ok"] else "❌"
                    lines.append(f"- {mark} **{c['name']}**：{c['detail']}")
                gr.Markdown("\n".join(lines))
            gr.Markdown(
                f"**技术详情**：模型 PP-YOLOE+-m · 输入 {info['input_size']}×{info['input_size']} · "
                f"类别 {info['num_classes']} · 后端 {info['backend']}（{info['device']}） · "
                f"Paddle {info['paddle_version'] or '未知'}"
            )

        # ---------- 事件绑定 ----------
        detect_btn.click(
            handlers._detect, inputs=[upload, conf, session_state],
            outputs=[gallery, img_selector, orig_img, vis_img, per_img_table,
                     combined_table, bar_plot, class_stats_table, class_area,
                     stats_md, agri_card, vlm_card, rag_card, agent_card, summary_md, download_files, run_state, session_state,
                     report_preview, report_pdf, flow_preview, flow_png, flow_pdf],
        )
        camera_btn.click(
            handlers._camera_detect, inputs=[camera_input, conf, session_state],
            outputs=[gallery, img_selector, orig_img, vis_img, per_img_table,
                     combined_table, bar_plot, class_stats_table, class_area,
                     stats_md, agri_card, vlm_card, rag_card, agent_card, summary_md, download_files, run_state, session_state,
                     report_preview, report_pdf, flow_preview, flow_png, flow_pdf],
        )
        img_selector.change(
            handlers._select_image, inputs=[img_selector, run_state],
            outputs=[orig_img, vis_img, per_img_table],
        )
        demo_sel.change(
            demo_mod._demo_select, inputs=[demo_sel],
            outputs=[demo_orig, demo_vis, demo_detail],
        )
        demo_prev_btn.click(
            demo_mod._demo_prev, inputs=[demo_index],
            outputs=[demo_index, demo2_orig, demo2_vis, demo2_info],
        )
        demo_next_btn.click(
            demo_mod._demo_next, inputs=[demo_index],
            outputs=[demo_index, demo2_orig, demo2_vis, demo2_info],
        )
        demo_play_btn.click(demo_mod._demo_play, outputs=[demo_playing])
        demo_stop_btn.click(demo_mod._demo_stop, outputs=[demo_playing])
        demo_timer.tick(
            demo_mod._demo_timer_tick, inputs=[demo_index, demo_playing],
            outputs=[demo_index, demo2_orig, demo2_vis, demo2_info, demo_playing],
        )
        demo_redetect_btn.click(
            demo_mod._demo_redetect, inputs=[demo_index],
            outputs=[demo2_orig, demo2_vis, demo2_info],
        )
        device_sel.change(
            handlers._switch_device, inputs=[device_sel],
            outputs=[sys_info_md, device_note],
        )
        upload_btn.upload(
            handlers._on_upload, inputs=[upload_btn],
            outputs=[upload, preview_gallery, upload_count],
        )
        clear_btn.click(
            handlers._clear, inputs=[session_state],
            outputs=[upload, preview_gallery, upload_count, gallery, img_selector,
                     orig_img, vis_img, per_img_table, combined_table, bar_plot,
                     class_stats_table, class_area, stats_md, agri_card, vlm_card,
                     rag_card, agent_card, summary_md, download_files, run_state, session_state,
                     report_preview, report_pdf, flow_preview, flow_png, flow_pdf],
        )
        voice_btn.click(
            handlers._voice_stream_interact, inputs=[recorded_audio, session_state],
            outputs=[mic_status, voice_text, voice_answer, asr_status, tts_status, voice_output_status, audio_output, session_state],
        )
        # 音频缓存事件链：新录音开始→清空；录音结束/上传成功→保存；清空/移除→清空。
        # 默认并发限制为 1，事件按 FIFO 串行，stop_recording 保存完成后才轮到按钮点击读取，无竞态。
        audio_input.start_recording(
            handlers._on_audio_record_start, outputs=[recorded_audio],
        )
        audio_input.stop_recording(
            handlers._on_audio_record_stop, inputs=[audio_input], outputs=[recorded_audio],
        )
        audio_input.upload(
            handlers._on_audio_upload, inputs=[audio_input], outputs=[recorded_audio],
        )
        audio_input.clear(
            handlers._on_audio_clear, outputs=[recorded_audio],
        )
        robot_btn.click(
            handlers._robot_execute, inputs=[session_state],
            outputs=[robot_card],
        )
        report_btn.click(
            handlers._generate_report, inputs=[session_state],
            outputs=[report_preview, report_pdf, session_state],
        )
        flow_btn.click(
            handlers._generate_flowchart, inputs=[session_state],
            outputs=[flow_preview, flow_png, flow_pdf],
        )

    return demo
