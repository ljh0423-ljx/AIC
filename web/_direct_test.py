# -*- coding: utf-8 -*-
"""临时验证脚本：直接调用 web.app 的处理器（模拟上传），不启动服务器。"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from web import app as webapp  # noqa: E402
from web import backend  # noqa: E402

VAL = BASE / "dataset/processed_detection_exiffix/images/val"
OK = lambda s: print(f"  [OK] {s}")  # noqa: E731


def val_of(u):
    """解包 gr.update 对象取真实 value。"""
    if isinstance(u, dict) and "value" in u:
        return u.get("value")
    return u


def main():
    print("=== 自检 ===")
    checks = backend.self_check()
    for c in checks:
        print(f"  [{'OK' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail'][:70]}")
    assert all(c["ok"] for c in checks), "自检未全部通过"
    OK("启动自检全部通过（模型已加载一次）")

    # ---------- 单图 ----------
    print("=== 单图上传检测 ===")
    single = str(VAL / "TRAIN_000004_apple_scab.jpg")
    r = webapp._detect([single], 0.5)
    gal, sel, orig, vis, per_tbl, comb, bar, cls_tbl, class_area, stats_md, agri, summ, dl, state = r
    comb_rows = val_of(comb)
    assert gal, "gallery 为空"
    assert state, "run_state 为空"
    run_dir = Path(state)
    import json
    records = json.load(open(run_dir / "predictions.json", encoding="utf-8"))
    print(f"  输出目录: {run_dir.name}; gallery={len(gal)}; 表格行数={len(comb_rows)}")
    print(f"  目标数: {len(records)}; 首目标 {records[0]['class_name']} {records[0]['confidence']:.3f}")
    assert records and records[0]["class_name"] == "Apple Scab Leaf", "类别映射/检测异常"
    assert len(comb_rows) == 1, "单图表格行数异常"
    assert "FPS" in summ, "汇总缺少 FPS"
    assert "农业视觉检测结果" in agri and "苹果" in agri, "农业结果卡异常"
    assert "不等同于专业植保诊断" in agri, "结果卡应含免责声明"
    OK("单图检测: 1 目标 Apple Scab Leaf, 表格 1 行, 汇总含 FPS, 农业结果卡正常")

    # ---------- 单图对比 ----------
    print("=== 单图对比/表格 ===")
    first_name = Path(gal[0][0]).name
    o, v, t = webapp._select_image(first_name, state)
    t = val_of(t)
    assert o and v and t, "单图对比数据为空"
    assert len(t) == 1, "单图明细行数异常"
    OK(f"原图={Path(o).name} 检测图={Path(v).name} 明细行数={len(t)}")

    # ---------- 批量 8 张 ----------
    print("=== 批量 8 张 ===")
    imgs = [
        "TRAIN_000004_apple_scab.jpg", "TEST_000018_0605_Rust-induced_leafspot.jpg",
        "TEST_000028_early_blight1-150x150.jpg", "TEST_000079_9511.img.jpg",
        "TRAIN_000037_Tomato+Problems+Septoria+Leaf+Spot.jpg",
        "TRAIN_000029_early-blight-of-tomato-tomato-1.jpg",
        "TEST_000106_depositphotos_3443387-stock-photo-the-green-grape-leaf-on.jpg",
        "TRAIN_000040_Septoria_leaf_spot_tomato.jpg",
    ]
    r = webapp._detect([str(VAL / i) for i in imgs], 0.5)
    gal, sel, orig, vis, per_tbl, comb, bar, cls_tbl, class_area, stats_md, agri, summ, dl, state = r
    comb_rows = val_of(comb)
    run_dir = Path(state)
    sm = json.load(open(run_dir / "inference_summary.json", encoding="utf-8"))
    perf = sm["performance"]
    records = json.load(open(run_dir / "predictions.json", encoding="utf-8"))
    print(f"  total={perf['total_images']} success={perf['success_images']} failed={perf['failed_images']}")
    print(f"  目标数={len(records)} total_time={perf['total_time']} FPS={perf['fps']}")
    print(f"  类别统计行数={len(sm['class_statistics'])}; gallery={len(gal)}; 下载文件={len(val_of(dl))}")
    assert perf["total_images"] == 8 and perf["failed_images"] == 0
    assert len(comb_rows) == len(records)
    assert sm["class_statistics"], "类别统计为空"
    OK(f"批量 8 张: 成功=8, 目标={len(records)}, FPS={perf['fps']}")

    # ---------- 无检测目标 ----------
    print("=== 无检测目标图片 ===")
    r = webapp._detect([str(VAL / "TEST_000028_early_blight1-150x150.jpg")], 0.5)
    gal, sel, orig, vis, per_tbl, comb, bar, cls_tbl, class_area, stats_md, agri, summ, dl, state = r
    assert len(val_of(comb)) == 1 and "暂无检测目标" in str(val_of(comb)[0]), "应显示暂无检测目标空状态"
    assert gal, "gallery 应保留原图"
    assert "未检出" in summ, "应提示未检出"
    assert "未检测到目标" in agri, "无检测时农业结果卡应提示未检测到目标"
    OK("无检测目标: 表格为空、有提示、原图保留、结果卡提示未检测")

    # ---------- 损坏图片隔离 ----------
    print("=== 损坏图片 + 2 正常（失败隔离） ===")
    corrupt = BASE / "inference/fixtures/corrupt_test.jpg"
    r = webapp._detect([str(corrupt), str(VAL / "TRAIN_000004_apple_scab.jpg"),
                        str(VAL / "TEST_000018_0605_Rust-induced_leafspot.jpg")], 0.5)
    gal, sel, orig, vis, per_tbl, comb, bar, cls_tbl, class_area, stats_md, agri, summ, dl, state = r
    sm = json.load(open(Path(state) / "inference_summary.json", encoding="utf-8"))
    perf = sm["performance"]
    print(f"  total={perf['total_images']} success={perf['success_images']} failed={perf['failed_images']}")
    assert perf["failed_images"] == 1 and perf["success_images"] == 2
    assert sm["failed"][0]["error"], "失败原因未记录"
    assert "失败明细" in summ
    OK(f"失败隔离: failed=1, 原因='{sm['failed'][0]['error'][:30]}...', 批次继续")

    # ---------- 空输入 ----------
    print("=== 空输入校验 ===")
    r = webapp._detect([], 0.5)
    assert val_of(r[0]) == [] and r[-1] is None, "空输入应返回默认结果"
    OK("空输入: 友好提示(gr.Warning)+默认结果")

    # ---------- 清空 ----------
    print("=== 清空任务 ===")
    r = webapp._clear()
    assert r[0] is None and val_of(r[1]) == [] and r[-1] is None
    OK("清空成功（上传区/结果区/状态已重置）")

    # ---------- 统计条 & 农业演示场景 ----------
    print("=== 统计条 & 农业演示场景 ===")
    assert "FPS" in stats_md, "统计条应含真实 FPS"
    OK(f"统计条: {stats_md}")
    assert webapp.DEMO_CASES and len(webapp.DEMO_CASES) >= 10, "演示案例应 >=10"
    gi = webapp._demo_gallery_items()
    assert len(gi) == len(webapp.DEMO_CASES), "演示 gallery 项数异常"
    d_id = webapp.DEMO_CASES[0]["demo_id"]
    o, v, d = webapp._demo_select(d_id)
    assert o and v and d, "演示案例选择异常"
    OK(f"演示场景: {len(webapp.DEMO_CASES)} 案例, 选择 {d_id} 正常")

    # ---------- 比赛演示模式 ----------
    print("=== 比赛演示模式 ===")
    assert len(webapp._DEMO_ORDER) == len(webapp.DEMO_CASES) == 15, "演示顺序应为15"
    order_ids = [webapp.DEMO_CASES[i]["demo_id"] for i in webapp._DEMO_ORDER]
    assert order_ids[-1] == "tomato_04_mosaic_dense", "密集场景应最后"
    idx, o, v, info = webapp._demo_next(0)
    assert idx == 1 and "案例 2" in info
    idx, o, v, info = webapp._demo_prev(idx)
    assert idx == 0 and "案例 1" in info
    idx, o, v, info, playing = webapp._demo_timer_tick(0, True)
    assert idx == 1 and playing is True
    idx, o, v, info, playing = webapp._demo_timer_tick(5, False)
    assert idx == 5 and playing is False
    o, v, info = webapp._demo_redetect(0)
    assert "重新检测" in info and "耗时" in info, "重新检测应含真实耗时"
    OK("比赛演示模式: 顺序/上下张/自动播放tick/停止/重新检测全部正常")
    assert "RK3588" in webapp._edge_status_markdown() and "尚未部署" in webapp._edge_status_markdown()
    OK("边缘部署状态卡: RK3588 标注接口预留、尚未部署")

    print("\n全部直接处理器验证通过 ✅")


if __name__ == "__main__":
    main()
