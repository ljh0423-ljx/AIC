# -*- coding: utf-8 -*-
"""
make_assets.py — 第二阶段比赛文档素材生成脚本（只读，不改动任何原始数据/模型/代码）
仅从现有训练日志、label_list.txt、val.json、demo_outputs、最终报告数据生成图表素材。
输出统一写入 docs/assets/。
"""
import re
import csv
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import cv2

BASE = Path(__file__).resolve().parent.parent          # AIC2026_AgriVision/
ASSETS = Path(__file__).resolve().parent / "assets"     # docs/assets/
ASSETS.mkdir(parents=True, exist_ok=True)

# 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 13 类（顺序与 label_list.txt / 报告一致）
CLASSES = [
    "Tomato Early blight leaf", "Tomato Septoria leaf spot", "Tomato leaf",
    "Tomato leaf bacterial spot", "Tomato leaf late blight", "Tomato leaf mosaic virus",
    "Tomato leaf yellow virus", "Tomato mold leaf", "Apple Scab Leaf", "Apple leaf",
    "Apple rust leaf", "grape leaf", "grape leaf black rot",
]
CROP = ["Tomato"] * 8 + ["Apple"] * 3 + ["Grape"] * 2
CROP_COLOR = {"Tomato": "#d9534f", "Apple": "#5cb85c", "Grape": "#8e44ad"}


def log(msg):
    print("[make_assets]", msg)


# ============================================================
# 1. 训练 loss / mAP 曲线（解析 formal_train.out）
# ============================================================
def parse_train_log():
    log_path = BASE / "experiments/direction_m/M_SCALEUP_100e/logs/formal_train.out"
    text = log_path.read_text(encoding="utf-8", errors="replace")

    iter_re = re.compile(
        r"Epoch: \[(\d+)\]\s+\[\s*(\d+)/57\]\s+learning_rate:\s+([\d.eE+-]+)\s+"
        r"loss:\s+([\d.]+)\s+loss_cls:\s+([\d.]+)\s+loss_iou:\s+([\d.]+)\s+"
        r"loss_dfl:\s+([\d.]+)\s+loss_l1:\s+([\d.]+)"
    )
    map_re = re.compile(
        r"Average Precision\s+\(AP\)\s+@\[\s*IoU=0\.50:0\.95\s+\|\s*area=\s+all\s+\|\s*maxDets=100\s*\]\s*=\s*([\d.]+)"
    )

    iters = []      # (epoch, iter, lr, loss, loss_cls, loss_iou, loss_dfl, loss_l1)
    maps = []       # (epoch, mAP)
    cur_epoch = 0
    for line in text.splitlines():
        m = iter_re.search(line)
        if m:
            cur_epoch = int(m.group(1))
            iters.append((cur_epoch, int(m.group(2)), float(m.group(3)),
                          float(m.group(4)), float(m.group(5)), float(m.group(6)),
                          float(m.group(7)), float(m.group(8))))
            continue
        m = map_re.search(line)
        if m:
            maps.append((cur_epoch, float(m.group(1))))
    return iters, maps


def gen_training_curves():
    iters, maps = parse_train_log()

    # 原始逐迭代 CSV（不聚合，不补数据）
    raw_csv = ASSETS / "training_loss_raw.csv"
    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "iter", "learning_rate", "loss", "loss_cls",
                    "loss_iou", "loss_dfl", "loss_l1"])
        for e, i, lr, lo, lc, li, ld, l1 in iters:
            w.writerow([e, i, lr, lo, lc, li, ld, l1])

    # 逐 epoch 平均 loss（仅对日志已记录的采样点取均值）
    from collections import defaultdict
    ep_loss = defaultdict(list)
    for e, i, lr, lo, lc, li, ld, l1 in iters:
        ep_loss[e].append(lo)
    epochs = sorted(ep_loss)
    mean_loss = [float(np.mean(ep_loss[e])) for e in epochs]

    ep_csv = ASSETS / "training_loss_epoch.csv"
    with open(ep_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "mean_loss"])
        for e, ml in zip(epochs, mean_loss):
            w.writerow([e + 1, round(ml, 6)])

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.plot([e + 1 for e in epochs], mean_loss, color="#c0392b", linewidth=1.8)
    ax.set_xlabel("Epoch（已训练轮数）")
    ax.set_ylabel("Training Loss（每 epoch 均值）")
    ax.set_title("PP-YOLOE+-m 训练 Loss 曲线（EXIF-fix, 100 epochs, bs=16, lr=0.002）")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "training_loss_curve.png")
    plt.close(fig)

    # mAP 随 epoch（每 5 个 epoch 评估一次，20 个点）
    map_csv = ASSETS / "training_map_epoch.csv"
    with open(map_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint_epoch(0-based)", "epoch(已训练轮数)", "val_mAP@0.5:0.95"])
        for e, ap in maps:
            w.writerow([e, e + 1, round(ap, 4)])

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    xs = [e + 1 for e, _ in maps]
    ys = [ap for _, ap in maps]
    ax.plot(xs, ys, marker="o", color="#2980b9", linewidth=1.8, markersize=4)
    best_e, best_ap = max(maps, key=lambda t: t[1])
    ax.scatter([best_e + 1], [best_ap], color="red", zorder=5, s=60)
    ax.annotate(f"best {best_ap:.3f}\n(epoch {best_e + 1})",
                xy=(best_e + 1, best_ap), xytext=(best_e + 1, best_ap - 0.08),
                ha="center", color="red", fontsize=9)
    ax.set_xlabel("Epoch（已训练轮数，每 5 轮评估一次）")
    ax.set_ylabel("VAL mAP@0.5:0.95")
    ax.set_title("PP-YOLOE+-m 验证 mAP 随 epoch 变化（VAL 113 图）")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "training_map_curve.png")
    plt.close(fig)

    log(f"loss 曲线：{len(iters)} 个迭代采样 → training_loss_raw.csv / training_loss_epoch.csv / training_loss_curve.png")
    log(f"mAP 曲线：{len(maps)} 个评估点 → training_map_epoch.csv / training_map_curve.png")


# ============================================================
# 2. 13 类分布柱状图（label_list.txt）
# ============================================================
def gen_class_distribution():
    src = BASE / "dataset/processed_detection_exiffix/label_list.txt"
    rows = []
    for line in src.read_text(encoding="utf-8").strip().splitlines():
        cid, name, crop, nimg, ntar = line.split("\t")
        rows.append((int(cid), name, crop, int(nimg), int(ntar)))
    assert len(rows) == 13, f"label_list 行数异常：{len(rows)}"

    dist_csv = ASSETS / "class_distribution.csv"
    with open(dist_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name", "crop", "train_images", "train_targets"])
        for cid, name, crop, nimg, ntar in rows:
            w.writerow([cid, name, crop, nimg, ntar])

    names = [r[1] for r in rows]
    imgs = [r[3] for r in rows]
    tars = [r[4] for r in rows]
    colors = [CROP_COLOR[r[2]] for r in rows]
    y = np.arange(len(names))[::-1]  # 自上而下

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), dpi=150)
    ax1.barh(y, imgs, color=colors, edgecolor="white")
    ax1.set_yticks(y); ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel("Train 图片数")
    ax1.set_title("13 类 · Train 图片数")
    ax1.grid(True, axis="x", alpha=0.3)
    for yi, v in zip(y, imgs):
        ax1.text(v + 1, yi, str(v), va="center", fontsize=7)

    ax2.barh(y, tars, color=colors, edgecolor="white")
    ax2.set_yticks(y); ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel("Train 目标数（bbox）")
    ax2.set_title("13 类 · Train 目标数")
    ax2.grid(True, axis="x", alpha=0.3)
    for yi, v in zip(y, tars):
        ax2.text(v + 1, yi, str(v), va="center", fontsize=7)

    fig.suptitle("PlantDoc 13 类数据分布（来源：label_list.txt，Train 子集）", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(ASSETS / "class_distribution.png")
    plt.close(fig)
    log("13 类分布 → class_distribution.csv / class_distribution.png")


# ============================================================
# 3. 13 类 AP 柱状图（M_FINAL_TEST_REPORT.md §3.3 / §5.3）
# ============================================================
def gen_class_ap():
    # 逐项对应 M_FINAL_TEST_REPORT.md（VAL 来自 §5.3，TEST 来自 §3.3/§5.3）
    AP = {
        "Tomato Early blight leaf":        (0.282, 0.268),
        "Tomato Septoria leaf spot":       (0.468, 0.515),
        "Tomato leaf":                     (0.241, 0.256),
        "Tomato leaf bacterial spot":      (0.296, 0.154),
        "Tomato leaf late blight":         (0.461, 0.565),
        "Tomato leaf mosaic virus":        (0.231, 0.095),
        "Tomato leaf yellow virus":        (0.208, 0.248),
        "Tomato mold leaf":                (0.199, 0.302),
        "Apple Scab Leaf":                 (0.547, 0.592),
        "Apple leaf":                      (0.850, 0.620),
        "Apple rust leaf":                 (0.658, 0.550),
        "grape leaf":                      (0.712, 0.594),
        "grape leaf black rot":            (0.414, 0.658),
    }
    ap_csv = ASSETS / "class_ap_val_test.csv"
    with open(ap_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "class_name", "crop", "AP_VAL", "AP_TEST"])
        for i, name in enumerate(CLASSES):
            v, t = AP[name]
            w.writerow([i, name, CROP[i], v, t])

    names = CLASSES
    val = [AP[n][0] for n in names]
    test = [AP[n][1] for n in names]
    y = np.arange(len(names))[::-1]
    h = 0.36
    colors = [CROP_COLOR[CROP[i]] for i in range(13)]

    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    ax.barh(y + h / 2, val, height=h, label="VAL mAP@0.5:0.95", color="#aeb6bf")
    ax.barh(y - h / 2, test, height=h, label="TEST mAP@0.5:0.95", color="#2980b9")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("AP @ IoU=0.5:0.95")
    ax.set_title("PP-YOLOE+-m 13 类 AP（VAL vs 独立 TEST）")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ASSETS / "class_ap_val_test.png")
    plt.close(fig)
    log("13 类 AP → class_ap_val_test.csv / class_ap_val_test.png")


# ============================================================
# 4. 系统技术架构图
# ============================================================
def box(ax, x, y, w, h, text, fc, ec="black", fs=9, dashed=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                       linewidth=1.2 if not dashed else 1.0,
                       linestyle="--" if dashed else "-",
                       facecolor=fc, edgecolor=ec)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def arrow(ax, x1, y1, x2, y2, dashed=False):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                        linewidth=1.4, linestyle="--" if dashed else "-",
                        color="#555555")
    ax.add_patch(a)


def gen_architecture():
    fig, ax = plt.subplots(figsize=(11, 12.5), dpi=150)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # 颜色
    c_input = "#fef3cd"; c_web = "#d6eaf8"; c_bk = "#d5f5e3"; c_model = "#fadbd8"
    c_out = "#ebdef0"; c_reserved = "#f2f3f4"

    # 布局（自上而下数据流）
    box(ax, 0.30, 0.955, 0.40, 0.030, "图片输入（单图 / 批量 / 文件夹 / 拖拽）", c_input, fs=9)
    box(ax, 0.30, 0.885, 0.40, 0.030, "web/app.py — Gradio UI（置信度滑块 / 演示模式）", c_web, fs=9)
    box(ax, 0.30, 0.815, 0.40, 0.030, "web/backend.py — get_detector() 单例\nset_device_mode(auto) · self_check()", c_web, fs=8)
    box(ax, 0.30, 0.745, 0.40, 0.030, "inference/detector.py — DetectorBackend 接口\nload/predict/predict_batch/info/close · create_backend 注册表", c_bk, fs=8)
    box(ax, 0.30, 0.675, 0.40, 0.030, "PaddleBackend（paddle_cpu / paddle_gpu）\n按 AGRI_DEVICE=auto 解析 → GPU:0 或 CPU", c_bk, fs=8)
    box(ax, 0.30, 0.605, 0.40, 0.030, "预处理（letterbox 640×640 · normalize · BGR→RGB → Tensor）", c_input, fs=9)
    box(ax, 0.30, 0.535, 0.40, 0.030, "PP-YOLOE+-m（best_model.pdparams · 23.57M · 13 类 · 640 输入）", c_model, fs=9)
    box(ax, 0.30, 0.465, 0.40, 0.030, "后处理 postprocess.py（NMS · 置信度阈值 0.5 → Detection 列表）", c_input, fs=9)
    box(ax, 0.30, 0.395, 0.40, 0.030, "可视化 visualize.py（bbox+标签）· batch_infer.py run_batch（逐图隔离+统计）", c_input, fs=8)
    box(ax, 0.30, 0.325, 0.40, 0.030, "结果输出：可视化 gallery / 结果表格 / 类别统计柱状图\n批次汇总 / 农业结果卡 / 结果下载(JSON·CSV·图)", c_out, fs=8)

    # RKNN 预留旁路（虚线）
    box(ax, 0.79, 0.675, 0.19, 0.030, "RKNNBackend\n(rknn_reserved 占位)", c_reserved, fs=8, dashed=True)
    box(ax, 0.79, 0.615, 0.19, 0.030, "RK3588 / RKNN\n预留接口 · 未部署", c_reserved, fs=8, dashed=True)
    arrow(ax, 0.79, 0.645, 0.79, 0.615, dashed=True)

    # 主数据流箭头（自上而下）
    ys = [0.955, 0.885, 0.815, 0.745, 0.675, 0.605, 0.535, 0.465, 0.395, 0.325]
    for i in range(len(ys) - 1):
        arrow(ax, 0.50, ys[i], 0.50, ys[i + 1] + 0.030)

    # 预留接口旁路连接（虚线）
    arrow(ax, 0.70, 0.690, 0.79, 0.690, dashed=True)

    ax.text(0.50, 0.285, "（RKNN/RK3588 为预留接口，尚未实现/部署；虚线表示未启用）",
            ha="center", fontsize=8, color="#888888", style="italic")
    ax.set_title("农业病害检测系统技术架构图（依据 web/ 与 inference/ 实际代码结构）", fontsize=13, pad=12)
    fig.savefig(ASSETS / "system_architecture.png", bbox_inches="tight")
    plt.close(fig)
    log("架构图 → system_architecture.png")


# ============================================================
# 5. 数据集样例 + GT bbox 可视化（val.json + images/val）
# ============================================================
def imread_any(p):
    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def save_grid(img_paths, out_path, cols=3, cell_h=360, gap=8):
    imgs = [imread_any(p) for p in img_paths]
    imgs = [im for im in imgs if im is not None]
    if not imgs:
        return
    # 统一每张到相同高度
    imgs = [cv2.resize(im, (int(im.shape[1] * cell_h / im.shape[0]), cell_h)) for im in imgs]
    rows = [imgs[i:i + cols] for i in range(0, len(imgs), cols)]
    row_widths = [sum(im.shape[1] for im in row) + gap * (len(row) - 1) for row in rows]
    gmax_w = max(row_widths)
    out_rows = []
    for row in rows:
        canvas = np.full((cell_h, gmax_w, 3), 255, dtype=np.uint8)
        x = 0
        for im in row:
            canvas[:, x:x + im.shape[1]] = im
            x += im.shape[1] + gap
        out_rows.append(canvas)
    grid = np.vstack(out_rows)
    cv2.imencode(".jpg", grid)[1].tofile(str(out_path))


def gen_dataset_samples():
    import json
    val_json = BASE / "dataset/processed_detection_exiffix/annotations/val.json"
    data = json.loads(val_json.read_text(encoding="utf-8"))
    cat_id2name = {c["id"]: c["name"] for c in data["categories"]}
    img_meta = {im["id"]: im for im in data["images"]}
    ann_by_img = {}
    for a in data["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)

    # 选取 6 张代表性 VAL 图（覆盖番茄/苹果/葡萄 + 单/多/密集）
    picks = [
        ("TRAIN_000004_apple_scab.jpg", "apple_scab_multi"),
        ("TEST_000079_9511.img.jpg", "tomato_mosaic_dense"),
        ("TEST_000088_tylcv-seminar-1-638.jpg", "tomato_yellow_virus_multi"),
        ("TRAIN_000014_stock-photo-green-apple-leaf-clipping-path-258728936.jpg", "apple_leaf_single"),
        ("TEST_000106_depositphotos_3443387-stock-photo-the-green-grape-leaf-on.jpg", "grape_leaf_single"),
        ("TEST_000108_5-29black-rot-chardRR.jpg", "grape_black_rot_single"),
    ]

    # 找到每张图对应的 image_id
    fname2id = {im["file_name"]: im["id"] for im in data["images"]}
    out_files = []
    for idx, (fname, tag) in enumerate(picks, 1):
        img_id = fname2id.get(fname)
        if img_id is None:
            log(f"  跳过（val.json 中无 {fname}）"); continue
        img_path = BASE / "dataset/processed_detection_exiffix/images/val" / fname
        img = imread_any(img_path)
        if img is None:
            log(f"  跳过（无法读取 {fname}）"); continue
        for a in ann_by_img.get(img_id, []):
            x, y, w, h = [int(round(v)) for v in a["bbox"]]
            cat = cat_id2name.get(a["category_id"], "?")
            crop = next((CROP[i] for i, n in enumerate(CLASSES) if n == cat), "?")
            color = {"Tomato": (80, 80, 220), "Apple": (80, 200, 80), "Grape": (150, 80, 200)}[crop]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            label = cat
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x, y - th - 6), (x + tw + 4, y), color, -1)
            cv2.putText(img, label, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
        out = ASSETS / f"dataset_sample_{idx:02d}_{tag}.jpg"
        cv2.imencode(".jpg", img)[1].tofile(str(out))
        out_files.append(out)
        log(f"  样例 {idx}: {tag} (GT {len(ann_by_img.get(img_id, []))} bbox)")

    # 组合网格
    if out_files:
        save_grid(out_files, ASSETS / "dataset_samples_grid.jpg")
        log("  数据集样例网格 → dataset_samples_grid.jpg")


# ============================================================
# 6. 演示检测结果样例（从 demo_outputs/visualized 复制，不修改）
# ============================================================
def gen_demo_results():
    src_dir = BASE / "web/demo_outputs/run_20260817_173307/visualized"
    picks = [
        ("TRAIN_000004_apple_scab.jpg", "demo_result_01_apple_scab_multi.jpg"),
        ("TEST_000079_9511.img.jpg", "demo_result_02_tomato_mosaic_dense.jpg"),
        ("TEST_000088_tylcv-seminar-1-638.jpg", "demo_result_03_tomato_yellow_virus_multi.jpg"),
        ("TRAIN_000014_stock-photo-green-apple-leaf-clipping-path-258728936.jpg", "demo_result_04_apple_leaf_single.jpg"),
        ("TEST_000106_depositphotos_3443387-stock-photo-the-green-grape-leaf-on.jpg", "demo_result_05_grape_leaf_single.jpg"),
        ("TEST_000108_5-29black-rot-chardRR.jpg", "demo_result_06_grape_black_rot_single.jpg"),
    ]
    copied = []
    for src_name, dst_name in picks:
        s = src_dir / src_name
        if s.exists():
            shutil.copy2(s, ASSETS / dst_name)
            copied.append(ASSETS / dst_name)
            log(f"  复制 {dst_name} (来源 {src_name})")
        else:
            log(f"  缺失 {src_name}")

    # 组合网格（能读则生成）
    if copied:
        save_grid(copied, ASSETS / "demo_results_grid.jpg")
        log("  演示样例网格 → demo_results_grid.jpg")


if __name__ == "__main__":
    log("开始生成素材 → docs/assets/")
    gen_training_curves()
    gen_class_distribution()
    gen_class_ap()
    gen_architecture()
    gen_dataset_samples()
    gen_demo_results()
    log("完成。")
