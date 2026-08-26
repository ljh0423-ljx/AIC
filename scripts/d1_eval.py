#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
d1_eval.py — D1 统一评估: A0(exiffix best) vs D1(density-aware VFL best) 在 VAL 上对比
与 d2_eval.py 使用完全相同的度量函数 (map_coco / greedy_metrics / density_recall),
保证 A0 与 D1 用同一套代码重算。新增低置信正确目标统计。

输出 (experiments/direction4/d1_formal/):
  A0_D1_metrics.csv            总体指标 (mAP@.5:.95, mAP@.5, P/R/F1@conf.5, 漏检, loc-ok-类错, FP)
  density_recall_comparison.csv 密度分组 Recall (1 / 2-4 / 5-9 / >=10)
  per_class_D1_comparison.csv   13 类 AP + Recall (A0 vs D1)
  low_confidence_comparison.csv 低置信正确目标数量 (按置信带)
TEST 封闭; 仅 VAL。
"""
import json, os, sys
from collections import defaultdict
import numpy as np
import pandas as pd

BASE = '/root/autodl-tmp/AIC2026_AgriVision'
OUT = f'{BASE}/experiments/direction4/d1_formal'
VAL_GT = f'{BASE}/dataset/processed_detection_exiffix/annotations/val.json'
BBOX_A0 = f'{BASE}/experiments/ablation_dir3_B1/val_eval/A0_best.json/bbox.json'
BBOX_D1 = f'{OUT}/val_eval/D1_best.json/bbox.json'
CONF_TH, IOU_TH = 0.5, 0.5
FOCUS = {2: 'Septoria', 4: 'bact', 6: 'mosaic', 7: 'yellow', 8: 'mold'}
os.makedirs(OUT, exist_ok=True)

coco = json.load(open(VAL_GT))
cats = {c['id']: c['name'] for c in coco['categories']}
gt_by = defaultdict(list)
for a in coco['annotations']:
    gt_by[a['image_id']].append(a)
gt_totals = defaultdict(int)
for a in coco['annotations']:
    gt_totals[a['category_id']] += 1
gt_img_density = {imid: len(v) for imid, v in gt_by.items()}


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


def low_conf_correct(preds_):
    """低置信正确目标统计:
       对每个 GT, 取其 IoU>=0.5 的检测中最高 score, 若该 score < 0.5,
       按置信带累计 (该目标被判对位置、但置信不足 conf0.5, 在 conf0.5 下会漏检)。"""
    det_by = defaultdict(list)
    for p in preds_:
        det_by[p['image_id']].append(p)
    bands = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]
    matched = defaultdict(int)
    for imid, anns in gt_by.items():
        dets = det_by.get(imid, [])
        for gi, a in enumerate(anns):
            best_score = -1.0
            for d in dets:
                if iou(a['bbox'], d['bbox']) >= IOU_TH and d['score'] > best_score:
                    best_score = d['score']
            if 0.0 <= best_score < 0.5:
                for (lo, hi) in bands:
                    if lo <= best_score < hi:
                        matched[(lo, hi)] += 1
    return {k: matched.get(k, 0) for k in bands}


def analyze(name, preds_):
    m, m50, per_ap = map_coco(preds_, per_class=True)
    tp, fp, fn, loc_ok, cls_ok, cls_tp, fp_by_class = greedy_metrics(preds_)
    p_, r_ = tp / (tp + fp), tp / (tp + fn)
    f1 = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0
    dr = density_recall(preds_)
    lc = low_conf_correct(preds_)
    return {'name': name, 'mAP': m, 'mAP50': m50, 'precision': p_, 'recall': r_, 'f1': f1,
            'missed': fn, 'loc_ok': loc_ok, 'cls_ok': cls_ok, 'loc_but_wrong_cls': loc_ok - cls_ok,
            'tp': tp, 'fp': fp, 'fn': fn, 'cls_tp': cls_tp, 'fp_by_class': fp_by_class,
            'per_ap': per_ap, 'density': dr, 'low_conf': lc}


# ---- 载入 A0 与 D1 的 VAL 推理 (同一套代码重算) ----
preds_a0 = json.load(open(BBOX_A0))
preds_d1 = json.load(open(BBOX_D1))
res_a0 = analyze('A0', preds_a0)
res_d1 = analyze('D1', preds_d1)

# ---- CSV 1: 总体指标 ----
rows = []
for r in (res_a0, res_d1):
    rows.append({'arm': r['name'],
                 'mAP@0.5:0.95': round(r['mAP'], 4), 'mAP@0.5': round(r['mAP50'], 4),
                 'Precision@conf0.5': round(r['precision'], 4),
                 'Recall@conf0.5': round(r['recall'], 4), 'F1@conf0.5': round(r['f1'], 4),
                 '总漏检(FN)': r['missed'], '定位正确loc_ok': r['loc_ok'],
                 '类正确cls_ok': r['cls_ok'], '定位对但类错': r['loc_but_wrong_cls'],
                 'TP': r['tp'], 'FP': r['fp']})
df_m = pd.DataFrame(rows)
df_m.to_csv(f'{OUT}/A0_D1_metrics.csv', index=False)

# ---- CSV 2: 密度分组 Recall ----
rows = []
for k in ['1', '2-4', '5-9', '>=10']:
    a = res_a0['density'][k]; d = res_d1['density'][k]
    rows.append({'density_bucket': k, 'A0_recall': a[0], 'A0_tp': a[1], 'A0_gt': a[2],
                 'D1_recall': d[0], 'D1_tp': d[1], 'D1_gt': d[2],
                 'dRecall': round(d[0] - a[0], 4)})
df_d = pd.DataFrame(rows)
df_d.to_csv(f'{OUT}/density_recall_comparison.csv', index=False)

# ---- CSV 3: 每类 AP + Recall ----
rows = []
for cid in range(1, 14):
    a_tp = res_a0['cls_tp'][cid]; d_tp = res_d1['cls_tp'][cid]
    gt = gt_totals[cid]
    rows.append({'category': cats[cid],
                 'A0_AP': round(res_a0['per_ap'][cid], 4), 'D1_AP': round(res_d1['per_ap'][cid], 4),
                 'dAP': round(res_d1['per_ap'][cid] - res_a0['per_ap'][cid], 4),
                 'A0_Recall': round(a_tp / gt, 4), 'D1_Recall': round(d_tp / gt, 4),
                 'dRecall': round(d_tp / gt - a_tp / gt, 4), 'GT': gt})
df_pc = pd.DataFrame(rows)
df_pc.to_csv(f'{OUT}/per_class_D1_comparison.csv', index=False)

# ---- CSV 4: 低置信正确目标 ----
rows = []
for k in [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]:
    a = res_a0['low_conf'][k]; d = res_d1['low_conf'][k]
    rows.append({'confidence_band': f'[{k[0]},{k[1]})', 'A0_correct_lowconf': a,
                 'D1_correct_lowconf': d, 'dCount': d - a})
tot_a = sum(res_a0['low_conf'].values()); tot_d = sum(res_d1['low_conf'].values())
rows.append({'confidence_band': '<0.5 total', 'A0_correct_lowconf': tot_a,
             'D1_correct_lowconf': tot_d, 'dCount': tot_d - tot_a})
df_lc = pd.DataFrame(rows)
df_lc.to_csv(f'{OUT}/low_confidence_comparison.csv', index=False)

# ---- 成功标准判定 ----
A0_MAP = res_a0['mAP']
GATE = round(A0_MAP + 0.005, 4)
dense_a = res_a0['density']['>=10'][0] if '>=10' in res_a0['density'] else 0.0
dense_d = res_d1['density']['>=10'][0]
dense_59_a = res_a0['density']['5-9'][0]; dense_59_d = res_d1['density']['5-9'][0]
m_pass = res_d1['mAP'] >= GATE
dense_pass = (dense_d > dense_a) or (dense_59_d > dense_59_a)
strong_a = sorted([(res_a0['per_ap'][c], c) for c in range(1, 14)], reverse=True)[:5]
strong_d = {c: res_d1['per_ap'][c] for _, c in strong_a}
strong_drop = [c for _, c in strong_a if strong_d[c] < res_a0['per_ap'][c] - 0.03]
strong_pass = len(strong_drop) == 0
verdict = 'PASS (D1 有效)' if (m_pass and dense_pass and strong_pass) else 'FAIL (D1 无效)'

print('=' * 76)
print('D1 统一评估 (VAL): A0(exiffix best) vs D1(density-aware VFL best)')
print('=' * 76)
for r in (res_a0, res_d1):
    print(f"[{r['name']}] mAP@.5:.95={r['mAP']:.4f} mAP@.5={r['mAP50']:.4f} "
          f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
          f"漏检={r['missed']} loc-ok-类错={r['loc_but_wrong_cls']} FP={r['fp']}")
print(f"\n成功标准: D1 mAP >= A0({A0_MAP:.4f}) + 0.005 = {GATE}")
print(f"  ① mAP 达标:   D1={res_d1['mAP']:.4f} >= {GATE} -> {'PASS' if m_pass else 'FAIL'}")
print(f"  ② 密集 Recall: >=10图 A0={dense_a:.3f}->D1={dense_d:.3f}, 5-9图 A0={dense_59_a:.3f}->D1={dense_59_d:.3f} -> {'PASS(有提升)' if dense_pass else 'FAIL(无提升)'}")
print(f"  ③ 强类回退:   强5类 AP 降>0.03 的类数 = {len(strong_drop)} -> {'PASS' if strong_pass else 'FAIL'}")
print(f"  => 判定: {verdict}")

print('\n密度分组 Recall (A0 -> D1):')
for k in ['1', '2-4', '5-9', '>=10']:
    a = res_a0['density'][k]; d = res_d1['density'][k]
    print(f"  {k:>5s}: A0={a[0]:.3f} ({a[1]}/{a[2]})  D1={d[0]:.3f} ({d[1]}/{d[2]})  Δ={d[0]-a[0]:+.3f}")
print('\n重点类 AP (A0 -> D1, Δ) 与 Recall@conf0.5:')
for cid, short in FOCUS.items():
    a_tp = res_a0['cls_tp'][cid]; d_tp = res_d1['cls_tp'][cid]
    gt = gt_totals[cid]
    print(f"  {short:8s}: AP {res_a0['per_ap'][cid]:.3f}->{res_d1['per_ap'][cid]:.3f} (Δ{res_d1['per_ap'][cid]-res_a0['per_ap'][cid]:+.3f})  "
          f"R {a_tp/gt:.3f}->{d_tp/gt:.3f} (Δ{(d_tp-a_tp)/gt:+.3f})")
print('\n低置信正确目标 (IoU>=0.5 但 score<0.5, A0 -> D1):')
for k in [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5)]:
    print(f"  [{k[0]:.1f},{k[1]:.1f}): A0={res_a0['low_conf'][k]}  D1={res_d1['low_conf'][k]}  Δ={res_d1['low_conf'][k]-res_a0['low_conf'][k]:+d}")
print(f"  <0.5 合计: A0={tot_a}  D1={tot_d}  Δ={tot_d-tot_a:+d}")
print(f"\nFP@conf0.5: A0={res_a0['fp']}  D1={res_d1['fp']}  Δ={res_d1['fp']-res_a0['fp']:+d}")
print('完成: d1_formal/ (CSVs)')
