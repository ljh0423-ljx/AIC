#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
错误案例可视化 — VAL 集, model_final
产出:
  error_cases/             逐图标注 (GT 绿 / TP 蓝 / FP 红 / 混淆橙)
  diagnostic_visualizations/  汇总图表 (per-class AP, 混淆热图, 尺寸分布, 作物对比, 错误构成)
"""
import json, os, glob
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pycocotools.coco import COCO

BASE = '/root/autodl-tmp/AIC2026_AgriVision'
VAL_IMG = f'{BASE}/dataset/processed_detection_clean/images/val'
GT = f'{BASE}/dataset/processed_detection_clean/annotations/val.json'
PRED = f'{BASE}/experiments/baseline_v1/metrics/val_eval/bbox.json'
OUT_CASES = f'{BASE}/experiments/baseline_v1/error_cases'
OUT_VIZ = f'{BASE}/experiments/baseline_v1/diagnostic_visualizations'
os.makedirs(OUT_CASES, exist_ok=True); os.makedirs(OUT_VIZ, exist_ok=True)

coco = COCO(GT)
cat_info = {c['id']: c['name'] for c in coco.dataset['categories']}
imgs = {im['id']: im for im in coco.dataset['images']}
gt_by_img = {}
for a in coco.dataset['annotations']:
    gt_by_img.setdefault(a['image_id'], []).append(a)
preds = json.load(open(PRED))
by_img = {}
for p in preds:
    by_img.setdefault(p['image_id'], []).append(p)

CONF_DRAW = 0.30

def iou_wh(bb1, bb2):
    x1, y1, w1, h1 = bb1; x2, y2, w2, h2 = bb2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    iw, ih = max(0, xi2-xi1), max(0, yi2-yi1)
    inter = iw*ih; u1, u2 = w1*h1, w2*h2
    return inter/(u1+u2-inter+1e-9)

def match_image(imid):
    """返回 gt_matched(gt_idx->pred), pred_role: list of (pred, role, gt_idx)"""
    gts = gt_by_img.get(imid, [])
    dets = sorted([p for p in by_img.get(imid, []) if p['score'] >= CONF_DRAW],
                  key=lambda d: -d['score'])
    matched_gt = set()
    pred_role = []
    for d in dets:
        best_i, best_iou, best_gc = -1, 0.0, None
        for gi in range(len(gts)):
            if gi in matched_gt: continue
            iou = iou_wh(gts[gi]['bbox'], d['bbox'])
            if iou > best_iou:
                best_i, best_iou, best_gc = gi, iou, gts[gi]['category_id']
        if best_i >= 0 and best_iou >= 0.5:
            matched_gt.add(best_i)
            role = 'TP' if d['category_id'] == best_gc else 'CONF'
            pred_role.append((d, role, best_i))
        else:
            pred_role.append((d, 'FP', -1))
    gt_role = [('missed' if gi not in matched_gt else 'hit') for gi in range(len(gts))]
    return gts, pred_role, gt_role, [gi for gi in range(len(gts)) if gi not in matched_gt]

def draw_case(imid, title, fname, show_det=True):
    im = imgs[imid]
    p = os.path.join(VAL_IMG, im['file_name'])
    img = cv2.imread(p)
    if img is None:
        print(f"  !! 无法读取 {p}"); return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    scale = min(1.0, 1400/max(img.shape[:2]))
    if scale < 1:
        img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)),
                         interpolation=cv2.INTER_AREA)
    S = 1/scale if scale < 1 else 1.0
    gts, pred_role, gt_role, missed = match_image(imid)
    thick = max(2, int(2*S)); fs = max(0.55, 0.85*S)
    # GT 框
    for gi, a in enumerate(gts):
        x, y, w, h = [v*scale for v in a['bbox']]
        color = (0, 180, 0) if gt_role[gi] == 'hit' else (0, 255, 0)
        cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), color, thick)
        label = cat_info[a['category_id']] + (' (missed)' if gt_role[gi]=='missed' else '')
        cv2.putText(img, label, (int(x), max(18, int(y)-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, color, max(1, thick-1), cv2.LINE_AA)
    # 预测框
    if show_det:
        for d, role, gi in pred_role:
            x, y, w, h = [v*scale for v in d['bbox']]
            color = {'TP': (0, 0, 255), 'FP': (255, 0, 0), 'CONF': (255, 128, 0)}[role]
            cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), color, thick)
            lbl = f"{cat_info[d['category_id']]}:{d['score']:.2f}"
            if role == 'CONF':
                lbl = f"GT[{cat_info[gts[gi]['category_id']][:14]}]->{lbl}"
            cv2.putText(img, lbl, (int(x), max(18, int(y+h)+16)),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.4, fs*0.75), color, max(1, thick-1), cv2.LINE_AA)
    # 图注
    cv2.putText(img, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)
    out = os.path.join(OUT_CASES, fname)
    plt.imsave(out, img)
    print(f"  保存 {out}")

# ============ A. 错误案例 ============
print("生成错误案例...")
# 漏检: 密集/大面积漏检
draw_case(92, "漏检典型: Tomato yellow virus 19目标仅检出1 (18漏检)", "case_missed_dense_yellowvirus_img92.jpg")
draw_case(7,  "漏检典型: 密集 mosaic 场景 12目标漏检10", "case_missed_dense_mosaic_img7.jpg")
draw_case(88, "漏检典型: 205x216 小图 mosaic 7目标全漏", "case_missed_smallimg_mosaic_img88.jpg")
# 误检
draw_case(64, "误检典型: 密集番茄叶场景 8目标/22检测/17误检", "case_fp_leaf_dense_img64.jpg")
draw_case(114,"误检典型: grape black rot 过度分割 (7目标/16检测/10误检)", "case_fp_blackrot_overseg_img114.jpg")
# 类别混淆
draw_case(63, "类别混淆: Tomato leaf GT 被判为 late blight / mold", "case_conf_leaf_vs_blight_mold_img63.jpg")
draw_case(5,  "类别混淆: Tomato bacterial spot GT 被判为 Septoria (score 0.92)", "case_conf_bactspot_vs_septoria_img5.jpg")
draw_case(67, "类别混淆: bacterial spot -> Septoria (另一例)", "case_conf_bactspot_vs_septoria_img67.jpg")
# EXIF 旋转影响
draw_case(65, "EXIF旋转影响: 标注尺寸与像素对调, 7目标全误检", "case_exif_rotation_img65.jpg")
print()

# ============ B. 汇总图表 ============
print("生成诊断图表...")
per_class = list(csv.reader if False else [])
# 读 per-class CSV
import csv
with open(f'{BASE}/experiments/baseline_v1/metrics/val_per_class_metrics.csv') as f:
    pc = list(csv.DictReader(f))
def cls_col(crop):
    return {'Tomato':'#d62728', 'Apple':'#1f77b4', 'Grape':'#2ca02c'}[crop]

# B1. per-class AP50:95 条形图
pc_sorted = sorted(pc, key=lambda r: r['AP50_95'])
names = [r['class'] for r in pc_sorted]
aps = [float(r['AP50_95']) for r in pc_sorted]
colors = [cls_col(r['crop']) for r in pc_sorted]
plt.figure(figsize=(11, 6))
bars = plt.barh(names, aps, color=colors)
plt.xlabel('AP@[0.5:0.95] (VAL)')
plt.title('Per-class AP (VAL, model_final) — Tomato=红 Apple=蓝 Grape=绿')
for b, v in zip(bars, aps):
    plt.text(v+0.01, b.get_y()+b.get_height()/2, f'{v:.3f}', va='center', fontsize=9)
plt.tight_layout(); plt.savefig(f'{OUT_VIZ}/per_class_AP50_95.png', dpi=120); plt.close()

# B2. 混淆矩阵热图
conf = np.zeros((13, 13))
gt_anns = coco.dataset['annotations']
# 重新做全局匹配统计 (conf=0.3 显示)
cats = sorted(cat_info)
idx = {c: i for i, c in enumerate(cats)}
for imid in gt_by_img:
    gts, pred_role, gt_role, _ = match_image(imid)
    for d, role, gi in pred_role:
        if role == 'CONF':
            conf[idx[gts[gi]['category_id']], idx[d['category_id']]] += 1
plt.figure(figsize=(9.5, 8.5))
plt.imshow(conf, cmap='YlOrRd')
lab = [cat_info[c] for c in cats]
plt.xticks(range(13), lab, rotation=90, fontsize=8)
plt.yticks(range(13), lab, fontsize=8)
for i in range(13):
    for j in range(13):
        v = conf[i, j]
        if v > 0:
            plt.text(j, i, int(v), ha='center', va='center', fontsize=7,
                     color='black' if v < 8 else 'white')
plt.xlabel('Predicted'); plt.ylabel('Ground Truth')
plt.title('Confusion matrix (VAL, conf=0.3, 仅跨类)')
plt.tight_layout(); plt.savefig(f'{OUT_VIZ}/confusion_heatmap_val.png', dpi=120); plt.close()

# B3. bbox 尺寸分布
with open(f'{BASE}/experiments/baseline_v1/metrics/val_bbox_size_dist.csv') as f:
    sz = list(csv.DictReader(f))[:-1]
sz_sorted = sorted(sz, key=lambda r: int(r['objects']))
n = len(sz_sorted)
plt.figure(figsize=(11, 6))
bott = np.zeros(n)
for key, lab in [('small', 'small(<32²)'), ('medium', 'medium(32²-96²)'), ('large', 'large(≥96²)')]:
    vals = np.array([int(r[key]) for r in sz_sorted])
    c = {'small':'#9467bd', 'medium':'#ff7f0e', 'large':'#1f77b4'}[key]
    plt.bar(range(n), vals, bottom=bott, label=lab, color=c, width=0.62)
    bott += vals
plt.xticks(range(n), [r['class'] for r in sz_sorted], rotation=75, fontsize=8)
plt.ylabel('# objects'); plt.title('GT bbox 尺寸分布 (VAL)')
plt.legend(); plt.tight_layout(); plt.savefig(f'{OUT_VIZ}/bbox_size_distribution_val.png', dpi=120); plt.close()

# B4. 作物对比
with open(f'{BASE}/experiments/baseline_v1/metrics/val_crop_summary.csv') as f:
    cr = list(csv.DictReader(f))
labels = [r['crop'] for r in cr]
metric_keys = ['mAP50_95', 'mPrecision', 'mRecall', 'mF1']
x = np.arange(3); width = 0.2
plt.figure(figsize=(9, 5.5))
for i, k in enumerate(metric_keys):
    vals = [float(r[k]) for r in cr]
    plt.bar(x+i*width, vals, width, label=k.replace('m', 'm'))
    for j, v in enumerate(vals):
        plt.text(j+i*width, v+0.01, f'{v:.2f}', ha='center', fontsize=8)
plt.xticks(x+width*1.5, labels)
plt.ylim(0, 0.9)
plt.ylabel('score'); plt.title('按作物汇总 (VAL) — Tomato 显著最弱')
plt.legend(); plt.tight_layout(); plt.savefig(f'{OUT_VIZ}/crop_summary_val.png', dpi=120); plt.close()

# B5. 错误构成 (FN/FP/Confusion per class, conf=0.5)
with open(f'{BASE}/experiments/baseline_v1/metrics/val_per_class_metrics.csv') as f:
    pc2 = list(csv.DictReader(f))
pc2s = sorted(pc2, key=lambda r: int(r['FN']))
n = len(pc2s)
plt.figure(figsize=(11, 6))
x = np.arange(n)
for k, c, lab in [('FN', '#d62728', 'FN (漏检)'), ('FP', '#ff7f0e', 'FP (误检)')]:
    vals = np.array([int(r[k]) for r in pc2s])
    plt.bar(x, vals, color=c, label=lab, width=0.6)
plt.xticks(x, [r['class'] for r in pc2s], rotation=75, fontsize=8)
plt.ylabel('count'); plt.legend(); plt.title('FN/FP per class (VAL, conf=0.5)')
plt.tight_layout(); plt.savefig(f'{OUT_VIZ}/error_composition_val.png', dpi=120); plt.close()
print("全部可视化完成。")
