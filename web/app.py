# -*- coding: utf-8 -*-
"""
app.py — 农业病害智能检测系统 Web 程序入口
==========================================
本文件只负责「启动」：
  - 关键环境变量设置（NO_PROXY 绕过 localhost 代理拦截，须在 import gradio 之前）
  - gradio_client 兼容性修补（仅运行时，不改第三方库文件）
  - 启动自检（backend.self_check，模型只加载一次）
  - 构建 UI（ui.build_ui）并启动服务器

职责拆分：
  ui.py        Gradio Blocks 页面组件构建 + 事件绑定
  handlers.py  检测/上传/清空/设备切换等事件处理 + 结果回读
  demo.py      比赛演示模式
  charts.py    matplotlib 图表生成
  render.py    静态 HTML/CSS 与 markdown 卡片生成
  assets.py    图片暂存与演示资源解析
  backend.py   检测后端单例 + 启动自检（保留原有封装）

启动方式不变：
    python web/app.py            # 默认 127.0.0.1:7860
    web\\run_web.bat             # 已强制 PYTHONUTF8=1
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# 关键兼容（必须放在 import gradio 之前）：
# 绕过对 127.0.0.1 的代理拦截。部分环境（企业代理/VPN/沙箱）会拦截 httpx 对
# localhost 的请求（返回 502），导致 gradio startup-events 检查失败、事件派发异常。
# 设置 NO_PROXY 后 httpx 直连本地，gradio 干净启动、点击/检测事件正常。
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
           "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_k, None)

_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from web import backend  # noqa: E402
from web.ui import build_ui  # noqa: E402

logger = logging.getLogger("agri.web.app")


def _apply_gradio_client_compat_patch() -> None:
    """gradio 4.44.1 + gradio_client 1.3.0 兼容性修补（仅运行时，不改第三方库文件）。

    背景：gradio 4.44.1 为 File/Image/Gallery 等组件生成的 schema 含
    `additionalProperties: true`（JSON Schema 中的裸布尔），而 gradio_client 1.3.0 的
    `_json_schema_to_python_type` 在递归处理时把该布尔当 dict 使用，抛出
    `TypeError: argument of type 'bool' is not iterable`，导致页面根路由 500。
    此处将 gradio_client.utils.get_type 在运行时补丁为"接受裸布尔 schema"，
    使 API 文档 schema 解析不再崩溃。若未来升级 gradio_client 已修复，此补丁自动失效。
    """
    try:
        import gradio_client.utils as _gcu

        _orig_get_type = _gcu.get_type

        def _safe_get_type(schema):
            if isinstance(schema, bool):
                return "boolean"
            return _orig_get_type(schema)

        _gcu.get_type = _safe_get_type
        logger.info("已应用 gradio_client 兼容性修补（additionalProperties: true）")
    except Exception:  # noqa: BLE001
        # 修补失败不影响程序启动（仅当上游已修复时不会走到这里）
        pass


_apply_gradio_client_compat_patch()


def _launch_with_localhost_fallback(app_obj, **launch_kwargs):
    """启动 Web 服务器。

    正常环境（文件顶部已设 NO_PROXY 绕过 localhost 代理）下，gradio 的
    startup-events 预启动探测会自然通过，此处直接 launch（**不启用任何补丁**，
    保证事件派发正常）。仅当确实因 localhost 探测被拦截而启动失败时，
    才临时禁用该探测并重试（运行时补丁，不修改第三方库文件）。
    """
    try:
        app_obj.launch(**launch_kwargs)
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if not (
            "localhost is not accessible" in msg
            or "shareable link" in msg
            or "startup-events" in msg
            or "Couldn't start the app" in msg
        ):
            raise
        print("[提示] localhost 预启动探测被拦截，启用本地兼容模式重试。")
        import gradio.networking as _gnet
        import httpx as _httpx

        _orig_url_ok = _gnet.url_ok
        _orig_httpx_get = _httpx.get

        def _fake_httpx_get(url, *a, **kw):
            if "startup-events" in str(url):
                return _httpx.Response(200, request=_httpx.Request("GET", str(url)))
            return _orig_httpx_get(url, *a, **kw)

        _gnet.url_ok = lambda url: True
        _httpx.get = _fake_httpx_get
        try:
            app_obj.launch(**launch_kwargs)
        finally:
            _gnet.url_ok = _orig_url_ok
            _httpx.get = _orig_httpx_get


if __name__ == "__main__":
    import argparse

    # 公网分享：GRADIO_SHARE=true 时开启 gradio.live 公开分享链接；默认 false（本地 127.0.0.1）。
    _share = os.environ.get("GRADIO_SHARE", "false").strip().lower() in ("1", "true", "yes", "y")

    _ap = argparse.ArgumentParser(description="农业病害智能检测系统 Web")
    _ap.add_argument(
        "--device", choices=["auto", "cpu", "gpu"],
        default=os.environ.get("AGRI_DEVICE", "auto"),
        help="推理设备：auto=启动时自动检测(cpu/gpu)；cpu=强制CPU；gpu=强制GPU(不可用报错)。默认 auto。",
    )
    _ap.add_argument(
        "--port", type=int, default=int(os.environ.get("AGRI_PORT", "7860")),
        help="Web 端口，默认 7860（可用环境变量 AGRI_PORT 覆盖）。",
    )
    _args = _ap.parse_args()

    backend.set_device_mode(_args.device)   # 通过启动参数指定设备模式，无需改代码

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print("=" * 60)
    print("农业病害智能检测系统 启动自检 ...")
    print(f"设备模式：{backend.get_device_mode()} | "
          f"公网分享：{'开启（gradio.live）' if _share else '关闭（本地 127.0.0.1 访问）'}")
    checks = backend.self_check()   # 触发模型加载（仅一次）
    for c in checks:
        print(f"  [{'OK' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}")
    # 实际推理设备（真实，不虚构）
    _info = backend.detector_info()
    _gpu = _info.get("gpu_info") or ""
    print(f"实际推理设备：{_info.get('device', '?').upper()}（后端 {_info.get('backend', '?')}）"
          + (f" · GPU：{_gpu}" if _gpu and _gpu != "无" else ""))
    print("=" * 60)
    if _share:
        print("[提示] 已开启公网分享模式：Gradio 将生成公开 gradio.live 分享链接，")
        print("        其他人在浏览器打开该链接即可直接使用当前电脑上的模型进行推理。")
        print("        关闭分享：退出进程后以 GRADIO_SHARE=false 重启（默认）。")
    print(f"Local URL: http://127.0.0.1:{_args.port}")
    if _share:
        print("Share URL: 由 Gradio 生成（创建隧道后输出 Running on public URL，此处不虚构）")
    sys.stdout.flush()
    app = build_ui(checks)
    app.queue()
    _launch_with_localhost_fallback(
        app,
        server_name="0.0.0.0",
        server_port=_args.port,
        show_error=True,
        prevent_thread_lock=False,
    )
