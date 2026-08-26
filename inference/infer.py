# -*- coding: utf-8 -*-
"""
infer.py — 农业病害检测系统命令行入口
======================================
支持单图 / 批量 / 文件夹递归推理，例如：

    python inference/infer.py --image xxx.jpg
    python inference/infer.py --dir xxx_folder
    python inference/infer.py --image xxx.jpg --conf 0.5
    python inference/infer.py --dir xxx_folder --conf 0.5
    python inference/infer.py --image a.jpg --image b.jpg --conf 0.5      # 批量
    python inference/infer.py --dir xxx --recursive --limit 20            # 递归+限量

支持格式：jpg / jpeg / png / bmp / webp。
单张失败不会终止整个批次，失败文件与原因记录在输出目录。
结果目录按时间戳自动创建（run_YYYYMMDD_HHMMSS/），不覆盖历史结果。
所有统计（耗时 / FPS）基于真实运行时间，不虚构。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 使脚本可以从任意目录以 `python inference/infer.py` 直跑：
# 将工程根目录加入 sys.path，再以包形式导入 inference 子模块。
_BASE_DIR = Path(__file__).resolve().parent.parent
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

from inference import config as app_config  # noqa: E402
from inference.batch_infer import BatchError, run_batch  # noqa: E402
from inference.detector import DetectorError, create_backend  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="infer",
        description="农业病害检测系统（PP-YOLOE+-m）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image", action="append", default=None, metavar="IMG",
        help="单张图片路径（可多次指定以批量推理）。",
    )
    parser.add_argument(
        "--dir", action="append", default=None, metavar="DIR",
        help="图片目录（可多次指定；默认非递归，见 --recursive）。",
    )
    parser.add_argument(
        "--conf", type=float, default=app_config.DEFAULT_CONF_THRESHOLD,
        help=f"置信度阈值，默认 {app_config.DEFAULT_CONF_THRESHOLD}。",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="对 --dir 目录递归搜索子目录图片。",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="最多处理前 N 张图片（小规模测试用）。",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help=f"输出根目录，默认 {app_config.OUTPUT_ROOT}（其下按时间戳建 run_* 子目录）。",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "gpu"],
        help="推理设备，默认 cpu；gpu 预留接口（无 GPU 时自动回退 CPU）。",
    )
    parser.add_argument(
        "--backend", type=str, default="paddle",
        help="推理后端，当前仅支持 paddle；rknn 为预留接口。",
    )
    parser.add_argument(
        "--input-size", type=int, default=None,
        help="模型输入边长（默认读取配置 640）。",
    )
    parser.add_argument(
        "--no-visualize", action="store_true",
        help="不生成可视化图（仅保留检测结果与统计）。",
    )
    return parser.parse_args(argv)


def _setup_console_logger() -> logging.Logger:
    logger = logging.getLogger("agri.infer")
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)
    return logger


def main(argv=None) -> int:
    logger = _setup_console_logger()
    args = parse_args(argv)

    image_paths: list[Path] = []
    if args.image:
        image_paths += [Path(p) for p in args.image]
    if args.dir:
        image_paths += [Path(p) for p in args.dir]
    if not image_paths:
        print(
            "[参数错误] 请至少指定一个输入：\n"
            "  --image <图片路径>   单张/多张\n"
            "  --dir   <目录路径>   文件夹推理\n"
            "例如：python inference/infer.py --image xxx.jpg"
        )
        return 2

    # ---- 参数校验 ----
    if not (0.0 < args.conf <= 1.0):
        print(f"[参数错误] 置信度阈值需在 (0,1] 区间，当前：{args.conf}")
        return 2
    output_root = Path(args.output) if args.output else app_config.OUTPUT_ROOT

    try:
        # ---- 快速失败：输入为空/不存在时，不加载模型直接报错 ----
        from inference.batch_infer import discover_images

        discover_images(image_paths, recursive=args.recursive, limit=args.limit)

        # ---- 加载后端（模型只加载一次）----
        logger.info("正在加载模型：%s", app_config.MODEL_PATH.name)
        detector = create_backend(
            kind=args.backend,
            model_path=app_config.MODEL_PATH,
            config_path=app_config.CONFIG_PATH,
            dataset_dir=app_config.DATASET_DIR,
            anno_rel_path=app_config.ANNO_REL_PATH,
            device=args.device,
            input_size=args.input_size,
        )
        detector.load()

        # ---- 执行推理（单图/批量/文件夹统一走批次引擎）----
        summary = run_batch(
            detector,
            image_paths,
            conf=args.conf,
            output_root=output_root,
            recursive=args.recursive,
            limit=args.limit,
            enable_visualization=not args.no_visualize,
            catid2cn=app_config.CLASS_NAMES_CN,
            paddle_version=detector._paddle_version,
        )
        detector.close()
    except (DetectorError, BatchError) as e:
        print(f"\n{str(e)}")
        return 1
    except KeyboardInterrupt:
        print("\n[已取消] 用户中断推理。")
        return 130
    except Exception as e:  # noqa: BLE001
        print(
            f"\n[运行错误] 推理过程中发生未预期异常：{e}\n"
            f"请检查日志/输出目录后重试。如需详细堆栈，请在原命令后加 PYTHONFAULTHANDLER=1。"
        )
        logger.debug("traceback", exc_info=True)
        return 1

    # ---- 中文汇总 ----
    print("\n==================== 推理完成 ====================")
    print(f"输出目录    : {summary['output_dir']}")
    print(f"图片总数    : {summary['total_images']}")
    print(f"成功        : {summary['success_images']}")
    print(f"失败        : {summary['failed_images']}")
    print(f"总耗时      : {summary['total_time']:.2f} s")
    print(f"平均耗时    : {summary['average_time']:.3f} s/张")
    print(f"FPS         : {summary['fps']:.2f}")
    if summary["failed"]:
        print("失败明细:")
        for f in summary["failed"]:
            print(f"  - {f['image_name']}: {f['error']}")
    print("==================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
