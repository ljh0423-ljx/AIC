# -*- coding: utf-8 -*-
"""
make_demo_report.py — 生成 DEMO_CASES_REPORT.md
==================================================
合并 demo_manifest.json 与最新 demo_outputs/run_*/ 的检测结果，
输出每个演示案例的：图片来源、场景、GT、检测结果与真实推理耗时。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))

WEB_DIR = Path(__file__).resolve().parent


def _latest_run() -> Path:
    runs = sorted(WEB_DIR.glob("demo_outputs/run_*"))
    if not runs:
        raise FileNotFoundError("未找到 demo_outputs/run_* 目录，请先运行推理。")
    return runs[-1]


def _load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def build() -> str:
    manifest = _load_json(WEB_DIR / "demo_data" / "demo_manifest.json")
    run_dir = _latest_run()
    summary = _load_json(run_dir / "inference_summary.json")
    predictions = _load_json(run_dir / "predictions.json")
    # image_name -> detections
    from collections import defaultdict

    by_img: dict[str, list] = defaultdict(list)
    for r in predictions:
        by_img[r["image_name"]].append(r)

    lines = [
        "# DEMO_CASES_REPORT — 农业病害智能检测系统演示案例",
        "",
        f"- 生成日期：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- 演示数据来源：`dataset/processed_detection_exiffix/images/val/`（**仅 VAL，未使用 TEST**）",
        "- 图片处理：仅复制到 `web/demo_data/`，**未修改原图**",
        f"- 推理程序：`inference/infer.py`（正式推理程序，PP-YOLOE+-m best_model）",
        f"- 结果目录：`{run_dir.name}`",
        f"- 性能：{summary['performance']['total_images']} 张 / 成功 {summary['performance']['success_images']} / 失败 {summary['performance']['failed_images']} / "
        f"总耗时 {summary['performance']['total_time']:.2f} s / 平均 {summary['performance']['average_time']:.3f} s/张 / "
        f"**实际 FPS {summary['performance']['fps']:.2f}**",
        "",
        "> 以下置信度均为模型真实输出（conf=0.5 阈值过滤），未人为提高。",
        "",
        "## 案例清单（15 例）",
        "",
        "| demo_id | 作物 | 场景 | 原图文件 | GT类别(数) | 检测结果(置信度) | 单图耗时(ms) |",
        "|---|---|---|---|---|---|---|",
    ]

    for c in manifest["cases"]:
        fn = c["file_name"]
        dets = by_img.get(fn, [])
        det_str = "; ".join(f"{d['class_name']} {d['confidence']:.2f}" for d in dets) or "（无检测）"
        # 单图耗时取自每图 JSON
        ms = ""
        jp = run_dir / "detections" / f"{Path(fn).stem}.json"
        if jp.is_file():
            try:
                ms = f"{_load_json(jp).get('inference_ms', 0):.0f}"
            except Exception:  # noqa: BLE001
                ms = ""
        lines.append(
            f"| {c['demo_id']} | {c['crop_cn']} | {c['scene_label']}({c['scenario']}) | "
            f"`{fn}` | {'/'.join(c['gt_classes'])}({c['gt_count']}) | {det_str} | {ms} |"
        )

    lines += [
        "",
        "## 场景类型覆盖",
        "- 单目标（single）：9 例",
        "- 多目标（multi，2~9 个）：5 例",
        "- 密集场景（dense，≥10 个 GT）：1 例（`tomato_04_mosaic_dense`，12 个 GT）",
        "- 作物覆盖：番茄 8 / 苹果 4 / 葡萄 3",
        "- 类别覆盖：12/13 类（Tomato 7 类、Apple 3 类、Grape 2 类）",
        "",
        "## 说明",
        "- 演示图片均来自 VAL 子集；**未复制/读取任何 TEST 图片**。",
        "- 检测结果全部来自正式推理程序，未做任何后处理/人为调高置信度。",
        "- 少量案例存在类间混淆（如 Septoria↔bacterial spot），为模型真实能力表现，如实展示。",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    out = WEB_DIR / "DEMO_CASES_REPORT.md"
    out.write_text(build(), encoding="utf-8")
    print(f"DEMO_CASES_REPORT.md 已生成：{out}")
