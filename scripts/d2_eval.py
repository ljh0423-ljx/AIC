#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
d2_eval.py — D2 统一评估: A0 原始推理 vs D2(锁定参数) 在 VAL 上对比
输出 (experiments/direction4/d2_results/):
  A0_D2_metrics.csv           总体指标 (mAP@.5:.95, mAP@.5, P/R/F1@conf.5, 漏检, loc-ok-类错, FP)
  density_d2_comparison.csv   密度分组 Recall (1 / 2-4 / 5-9 / >=10)
  per_class_d2_comparison.csv 13 类 AP + Recall (A0 vs D2)
  可视化 PNG + D2_REPORT.md
TEST 封闭; 参数来自 params_locked.json (TRAIN 锁定, 已冻结, 不依据 VAL 调整)。
"""
import json, os, sys
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d2_operator import apply_operator

BASE = '/root/autodl-tmp/AIC2026_AgriVision'
OUT = f'{BASE}/experiments/direction4/d2_results'
VAL_GT = f'{BASE}/dataset/processed_detection_exiffix/annotations/val.json'
BBOX_A0 = f'{BASE}/experiments/ablation_dir3_B1/val_eval/A0_best.json/bbox.json'
CONF_TH, IOU_TH = 0.5, 0.5
FOCUS = {2: 'Septoria', 4: 'bact', 6: 'mosaic', 7: 'yellow', 8: 'mold'}
os.makedirs(OUT, exist_ok=True)

coco = json.load(open(VAL_GT))
cats = {c['id']: c['name'] for c in coco['categories']}
img_meta = {im['id']: (im['width'], im['height']) for im in coco['images']}
gt_by = defaultdict(list)
for a in coco['annotations']:
    gt_by[a['image_id']].append(a)
gt_totals = defaultdict(int)
for a in coco['annotations']:
    gt_totals[a['category_id']] += 1
gt_img_density = {imid: len(v) for imid, v in gt_by.items()}

locked = json.load(open(f'{OUT}/params_locked.json'))
assert locked['enabled'], 'params_locked.json 显示 D2 未启用,不应进入评估'


def iou(bb1, bb2):
    x1, y1, w1, h1 = bb1; x2, y2, w2, h2 = bb2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    iw, ih = max(0, xi2 - xi1), max(0, yi2 - yi1)
    inter = iw * ih
    return inter / (w1 * h1 + w2 * h2 - inter + 1e-9)


def greedy_metrics(preds_):
    """conf0.5 贪心匹配: 返回 (tp, fp, fn, loc_ok, cls_ok, cls_tp, fp_by_class)"""
    det_by = defaultdict(list)
    for p in preds_:
        det_by[p['image_id']].append(p)
    tp = fp = loc_ok = cls_ok = 0
    cls_tp = defaultdict(int)
    fp_by_class = defaultdict(int)
    for imid, anns in gt_by.items():
        dets = sorted([d for d in det_by.get(imid, []) if d['score'] >= CONF_TH],
                      key=lambda d: -d['score'])
        used = set()
        for d in dets:
            bi, biou, bgc = -1, 0.0, None
            for gi, a in enumerate(anns):
                if gi in used: continue
                io = iou(a['bbox'], d['bbox'])
                if io > biou: bi, biou, bgc = gi, io, a['category_id']
            if bi >= 0 and biou >= IOU_TH:
                used.add(bi); tp += 1; loc_ok += 1
                cls_tp[bgc] += 1
                if d['category_id'] == bgc: cls_ok += 1
            else:
                fp += 1
                fp_by_class[d['category_id']] += 1
    fn = len(coco['annotations']) - loc_ok
    return tp, fp, fn, loc_ok, cls_ok, cls_tp, fp_by_class


def density_recall(preds_):
    det_by = defaultdict(list)
    for p in preds_:
        det_by[p['image_id']].append(p)
    buckets = {'1': [0, 0], '2-4': [0, 0], '5-9': [0, 0], '>=10': [0, 0]}
    for imid, anns in gt_by.items():
        d = gt_img_density[imid]
        key = '1' if d == 1 else ('2-4' if d <= 4 else ('5-9' if d <= 9 else '>=10'))
        dets = sorted([x for x in det_by.get(imid, []) if x['score'] >= CONF_TH],
                      key=lambda x: -x['score'])
        used = set(); n = 0
        for det in dets:
            bi, biou = -1, 0.0
            for gi, a in enumerate(anns):
                if gi in used: continue
                io = iou(a['bbox'], det['bbox'])
                if io > biou: bi, biou = gi, io
            if bi >= 0 and biou >= IOU_TH:
                used.add(bi); n += 1
        buckets[key][0] += n
        buckets[key][1] += len(anns)
    return {k: (round(v[0] / v[1], 4), v[0], v[1]) for k, v in buckets.items()}


def map_coco(preds_, per_class=False):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    cc = COCO(VAL_GT)
    ev = COCOeval(cc, cc.loadRes(preds_), 'bbox')
    ev.evaluate(); ev.accumulate(); ev.summarize()
    stats = ev.stats
    per = {}
    if per_class:
        p = ev.eval['precision']  # [T,R,K,A,M], M 索引: 0=maxDets1, 1=maxDets10, 2=maxDets100
        for cid in range(1, 14):
            ap = p[:, :, cid - 1, 0, 2]   # area=all, maxDets=100 (标准每类 AP)
            ap = ap[ap >= 0]
            per[cid] = float(ap.mean()) if ap.size else 0.0
    return stats[0], stats[1], per


def analyze(name, preds_):
    m, m50, per_ap = map_coco(preds_, per_class=True)
    tp, fp, fn, loc_ok, cls_ok, cls_tp, fp_by_class = greedy_metrics(preds_)
    p_, r_ = tp / (tp + fp), tp / (tp + fn)
    f1 = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0
    dr = density_recall(preds_)
    return {'name': name, 'mAP': m, 'mAP50': m50, 'precision': p_, 'recall': r_, 'f1': f1,
            'missed': fn, 'loc_ok': loc_ok, 'cls_ok': cls_ok, 'loc_but_wrong_cls': loc_ok - cls_ok,
            'tp': tp, 'fp': fp, 'fn': fn, 'cls_tp': cls_tp, 'fp_by_class': fp_by_class,
            'per_ap': per_ap, 'density': dr}


# ---- 载入 A0 原始 VAL 推理 ----
preds_a0 = json.load(open(BBOX_A0))

# ---- D2: 应用锁定参数 ----
preds_d2, op_stats = apply_operator(
    preds_a0, img_meta,
    alpha=locked['alpha'], straggler=locked['straggler'],
    use_rescore=locked['use_rescore'], tau=locked['tau'],
    lam=locked['lam'], min_agg=locked['min_agg'])

# ---- 统一评估 ----
res_a0 = analyze('A0', preds_a0)
res_d2 = analyze('D2', preds_d2)

# ---- CSV 1: 总体指标 ----
rows = []
for r in (res_a0, res_d2):
    rows.append({'arm': r['name'],
                 'mAP@0.5:0.95': round(r['mAP'], 4), 'mAP@0.5': round(r['mAP50'], 4),
                 'Precision@conf0.5': round(r['precision'], 4),
                 'Recall@conf0.5': round(r['recall'], 4), 'F1@conf0.5': round(r['f1'], 4),
                 '总漏检(FN)': r['missed'], '定位正确loc_ok': r['loc_ok'],
                 '类正确cls_ok': r['cls_ok'], '定位对但类错': r['loc_but_wrong_cls'],
                 'TP': r['tp'], 'FP': r['fp']})
df_m = pd.DataFrame(rows)
df_m.to_csv(f'{OUT}/A0_D2_metrics.csv', index=False)

# ---- CSV 2: 密度分组 Recall ----
rows = []
for k in ['1', '2-4', '5-9', '>=10']:
    a = res_a0['density'][k]; d = res_d2['density'][k]
    rows.append({'density_bucket': k, 'A0_recall': a[0], 'A0_tp': a[1], 'A0_gt': a[2],
                 'D2_recall': d[0], 'D2_tp': d[1], 'D2_gt': d[2],
                 'dRecall': round(d[0] - a[0], 4)})
df_d = pd.DataFrame(rows)
df_d.to_csv(f'{OUT}/density_d2_comparison.csv', index=False)

# ---- CSV 3: 每类 AP + Recall ----
rows = []
for cid in range(1, 14):
    a_tp = res_a0['cls_tp'][cid]; d_tp = res_d2['cls_tp'][cid]
    gt = gt_totals[cid]
    rows.append({'category': cats[cid],
                 'A0_AP': round(res_a0['per_ap'][cid], 4), 'D2_AP': round(res_d2['per_ap'][cid], 4),
                 'dAP': round(res_d2['per_ap'][cid] - res_a0['per_ap'][cid], 4),
                 'A0_Recall': round(a_tp / gt, 4), 'D2_Recall': round(d_tp / gt, 4),
                 'dRecall': round(d_tp / gt - a_tp / gt, 4), 'GT': gt})
df_pc = pd.DataFrame(rows)
df_pc.to_csv(f'{OUT}/per_class_d2_comparison.csv', index=False)

# ---- 可视化 ----
plt.rcParams['figure.dpi'] = 110
labels = ['mAP@.5:.95', 'mAP@.5', 'P@conf.5', 'R@conf.5', 'F1@conf.5']
a_vals = [res_a0['mAP'], res_a0['mAP50'], res_a0['precision'], res_a0['recall'], res_a0['f1']]
d_vals = [res_d2['mAP'], res_d2['mAP50'], res_d2['precision'], res_d2['recall'], res_d2['f1']]
x = np.arange(len(labels)); w = 0.36
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w/2, a_vals, w, label='A0 (raw)', color='#8ecae6')
ax.bar(x + w/2, d_vals, w, label='D2 (locked params)', color='#f4a261')
for xi, a, d in zip(x, a_vals, d_vals):
    ax.text(xi - w/2, a + 0.008, f'{a:.3f}', ha='center', fontsize=7)
    ax.text(xi + w/2, d + 0.008, f'{d:.3f}', ha='center', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.0)
ax.set_ylabel('score'); ax.legend(); ax.set_title('A0 vs D2 overall metrics (VAL)')
fig.tight_layout(); fig.savefig(f'{OUT}/overall_metrics_A0_vs_D2.png'); plt.close(fig)

dk = list(df_d['density_bucket'])
xd = np.arange(len(dk))
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(xd - w/2, df_d['A0_recall'], w, label='A0')
ax.bar(xd + w/2, df_d['D2_recall'], w, label='D2')
for xi, a, d in zip(xd, df_d['A0_recall'], df_d['D2_recall']):
    ax.text(xi - w/2, a + 0.01, f'{a:.2f}', ha='center', fontsize=7)
    ax.text(xi + w/2, d + 0.01, f'{d:.2f}', ha='center', fontsize=7)
ax.set_xticks(xd); ax.set_xticklabels(dk); ax.set_ylim(0, 1.0)
ax.set_ylabel('Recall@conf0.5'); ax.legend(); ax.set_title('Density-bucket Recall: A0 vs D2 (VAL)')
fig.tight_layout(); fig.savefig(f'{OUT}/density_recall_A0_vs_D2.png'); plt.close(fig)

x = np.arange(13)
fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x - w/2, df_pc['A0_Recall'], w, label='A0')
ax.bar(x + w/2, df_pc['D2_Recall'], w, label='D2')
ax.set_xticks(x); ax.set_xticklabels([f'{i+1}:{n.split()[-1]}' for i, n in enumerate(df_pc['category'])], rotation=90, fontsize=6)
ax.set_ylabel('Recall@conf0.5'); ax.legend(); ax.set_title('Per-class Recall: A0 vs D2 (VAL)')
fig.tight_layout(); fig.savefig(f'{OUT}/per_class_recall_A0_vs_D2.png'); plt.close(fig)

# ---- 控制台汇总 ----
print('=' * 70)
print('VAL 统一评估: A0(原始) vs D2(锁定参数)')
print('=' * 70)
for r in (res_a0, res_d2):
    print(f"[{r['name']}] mAP@.5:.95={r['mAP']:.4f} mAP@.5={r['mAP50']:.4f} "
          f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
          f"漏检={r['missed']} loc-ok-类错={r['loc_but_wrong_cls']} FP={r['fp']}")
print(f"\nD2 算子: 重标={op_stats['relabeled']} 重分={op_stats['rescored']} "
      f"(主导类图像数={op_stats['images_with_dom']})")
print(f"锁定参数: {json.dumps(locked, ensure_ascii=False)}")
print('\n密度分组 Recall (A0 -> D2):')
for k in ['1', '2-4', '5-9', '>=10']:
    a = res_a0['density'][k]; d = res_d2['density'][k]
    print(f"  {k:>5s}: A0={a[0]:.3f} ({a[1]}/{a[2]})  D2={d[0]:.3f} ({d[1]}/{d[2]})  Δ={d[0]-a[0]:+.3f}")
print('\n重点类 Recall@conf0.5 (A0 -> D2, Δ) 与 AP:')
for cid, short in FOCUS.items():
    a_tp = res_a0['cls_tp'][cid]; d_tp = res_d2['cls_tp'][cid]
    gt = gt_totals[cid]
    print(f"  {short:8s}: R {a_tp/gt:.3f}->{d_tp/gt:.3f} (Δ{(d_tp-a_tp)/gt:+.3f})  "
          f"AP {res_a0['per_ap'][cid]:.3f}->{res_d2['per_ap'][cid]:.3f}")
print('\nFP@conf0.5: A0=', res_a0['fp'], ' D2=', res_d2['fp'], ' Δ=', res_d2['fp'] - res_a0['fp'])
print('完成: d2_results/ (CSVs + PNGs)')
