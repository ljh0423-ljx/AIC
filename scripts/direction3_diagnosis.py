#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
direction3_diagnosis.py — 方向3 深度诊断 (B1 设计依据)
基准 = A0 (EXIF-Fix + VFL best checkpoint)。仅 VAL; TEST 封闭。
目标: 量化漏检问题 (yellow virus/mosaic/mold)、类别不平衡、密集场景。
输出 (experiments/direction3/):
  d3_diagnostic.json   全部数值
  DIRECTION3_DIAGNOSTIC.md  诊断报告
"""
import json, os
import numpy as np
from collections import Counter

BASE = '/root/autodl-tmp/AIC2026_AgriVision'
GT_PATH = f'{BASE}/dataset/processed_detection_exiffix/annotations/val.json'
TR_PATH = f'{BASE}/dataset/processed_detection_exiffix/annotations/train.json'
BBOX = f'{BASE}/experiments/direction2/val_eval/A0_best.json/bbox.json'
OUT = f'{BASE}/experiments/direction3'
CONF_TH = 0.5
IOU_TH = 0.5
os.makedirs(OUT, exist_ok=True)

coco = json.load(open(GT_PATH))
cats = {c['id']: c['name'] for c in coco['categories']}
imgs = {im['id']: im for im in coco['images']}
name2id = {v: k for k, v in cats.items()}
gt_by_img = {}
for a in coco['annotations']:
    gt_by_img.setdefault(a['image_id'], []).append(a)
preds = json.load(open(BBOX))
det_by_img = {}
for p in preds:
    det_by_img.setdefault(p['image_id'], []).append(p)

# ---------- TRAIN 统计 ----------
tr = json.load(open(TR_PATH))
tr_count = Counter(a['category_id'] for a in tr['annotations'])
tr_img = {}
for a in tr['annotations']:
    tr_img.setdefault(a['category_id'], set()).add(a['image_id'])

# ---------- 匹配 ----------
def iou_wh(bb1, bb2):
    x1, y1, w1, h1 = bb1; x2, y2, w2, h2 = bb2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    iw, ih = max(0, xi2-xi1), max(0, yi2-yi1)
    inter = iw*ih
    return inter / (w1*h1 + w2*h2 - inter + 1e-9)

# 每类: fn@conf0.5, "完全无检测覆盖" (无 IoU>0.1 的检测), 以及最低 conf 检测的 IoU
per_cls = {}
dense_ctx = {}
AREA = {'s': 0, 'm': 0, 'l': 0}
for cid in range(1, 14):
    per_cls[cid] = {'gt_total': 0, 'fn_conf05': 0, 'fn_no_det': 0,
                    'miss_img_density': [], 'miss_area': Counter(),
                    'miss_min_iou_any': []}
for imid, gt_anns in gt_by_img.items():
    dens = len(gt_anns)
    dets = det_by_img.get(imid, [])
    dets50 = [d for d in dets if d['score'] >= CONF_TH]
    for a in gt_anns:
        cid = a['category_id']
        per_cls[cid]['gt_total'] += 1
        # 面积: 相对图像
        im = imgs[imid]; W, H = im['width'], im['height']
        x, y, w, h = a['bbox']; area = w*h/(W*H)
        if area < 0.005: sz = 's'
        elif area < 0.02: sz = 'm'
        else: sz = 'l'
        AREA[sz] += 1
        best50, best_any = 0.0, 0.0
        for d in dets50:
            best50 = max(best50, iou_wh(a['bbox'], d['bbox']))
        for d in dets:
            best_any = max(best_any, iou_wh(a['bbox'], d['bbox']))
        if best50 < IOU_TH:
            per_cls[cid]['fn_conf05'] += 1
            per_cls[cid]['miss_img_density'].append(dens)
            per_cls[cid]['miss_area'][sz] += 1
            per_cls[cid]['miss_min_iou_any'].append(best_any)
            if best_any < 0.1:
                per_cls[cid]['fn_no_det'] += 1

tot = {'gt': sum(v['gt_total'] for v in per_cls.values()),
       'fn_conf05': sum(v['fn_conf05'] for v in per_cls.values()),
       'fn_no_det': sum(v['fn_no_det'] for v in per_cls.values())}

# ---------- 密集 vs 稀疏 场景的漏检率 ----------
miss_dense = miss_sparse = gt_dense = gt_sparse = 0
for imid, gt_anns in gt_by_img.items():
    dens = len(gt_anns)
    dets = det_by_img.get(imid, [])
    dets50 = [d for d in dets if d['score'] >= CONF_TH]
    for a in gt_anns:
        best50 = max([iou_wh(a['bbox'], d['bbox']) for d in dets50], default=0.0)
        if dens >= 5:
            gt_dense += 1
            if best50 < IOU_TH: miss_dense += 1
        else:
            gt_sparse += 1
            if best50 < IOU_TH: miss_sparse += 1

# ---------- 漏检目标的"最高覆盖分数"分布 (若降低阈值可恢复多少) ----------
# 对每个 conf0.5 漏检的 GT: 找 IoU>=0.5 的检测中 score 最高者 (即"差一点"的置信度)
miss_score = {cid: [] for cid in range(1, 14)}
miss_score_all = []
RECALL_AT = {0.1: 0, 0.3: 0, 0.5: 0}
for imid, gt_anns in gt_by_img.items():
    dets = det_by_img.get(imid, [])
    for a in gt_anns:
        cid = a['category_id']
        best_iou, best_score = 0.0, 0.0
        for d in dets:
            iou = iou_wh(a['bbox'], d['bbox'])
            if iou >= 0.5 and d['score'] > best_score:
                best_score = d['score']
            best_iou = max(best_iou, iou)
        if best_score < CONF_TH:  # conf0.5 漏检: 无 score>=0.5 且 IoU>=0.5 的检测
            miss_score[cid].append(best_score)
            miss_score_all.append(best_score)
# 注意: 上面 best_iou 基于全检测集, 需在 conf 尺度下重算 recall@t
recall_at = {}
for t in RECALL_AT:
    hit = 0
    for imid, gt_anns in gt_by_img.items():
        dets = [d for d in det_by_img.get(imid, []) if d['score'] >= t]
        used = set()
        for a in gt_anns:
            for d in sorted(dets, key=lambda x: -x['score']):
                gi = gt_anns.index(a)
                if gi in used: continue
                if iou_wh(a['bbox'], d['bbox']) >= IOU_TH:
                    hit += 1; used.add(gi); break
    recall_at[t] = hit / tot['gt']

# ---------- 保存 ----------
def bucket(scores):
    b = {'0.01-0.1': 0, '0.1-0.3': 0, '0.3-0.5': 0}
    for s in scores:
        if s < 0.1: b['0.01-0.1'] += 1
        elif s < 0.3: b['0.1-0.3'] += 1
        else: b['0.3-0.5'] += 1
    return b

out = {'per_class': {}, 'total': tot,
       'dense_vs_sparse': {'dense_gt': gt_dense, 'dense_miss': miss_dense, 'dense_miss_rate': round(miss_dense/gt_dense, 3),
                            'sparse_gt': gt_sparse, 'sparse_miss': miss_sparse, 'sparse_miss_rate': round(miss_sparse/gt_sparse, 3)},
       'area_gt': dict(AREA), 'train_count': dict(tr_count), 'train_img': {str(k): len(v) for k, v in tr_img.items()},
       'recall_at_conf': {str(k): round(v, 3) for k, v in recall_at.items()},
       'miss_score_all': bucket(miss_score_all)}
for cid, v in per_cls.items():
    gt = v['gt_total']
    out['per_class'][str(cid)] = {
        'name': cats[cid], 'gt_total': gt, 'train_count': tr_count[cid],
        'train_img': len(tr_img[cid]),
        'fn_conf05': v['fn_conf05'],
        'miss_rate_conf05': round(v['fn_conf05']/gt, 3),
        'fn_no_det': v['fn_no_det'],
        'no_det_rate': round(v['fn_no_det']/gt, 3),
        'miss_mean_img_density': round(np.mean(v['miss_img_density']), 2) if v['miss_img_density'] else None,
        'miss_area': dict(v['miss_area']),
        'miss_min_iou_mean': round(np.mean(v['miss_min_iou_any']), 3) if v['miss_min_iou_any'] else None,
        'miss_cover_score_bucket': bucket(miss_score[cid]) if miss_score[cid] else {}}
json.dump(out, open(f'{OUT}/d3_diagnostic.json', 'w'), ensure_ascii=False, indent=1)

# ---------- 控制台 ----------
print('==== 方向3 深度诊断 (A0 best, VAL) ====')
print(f'总 GT={tot["gt"]}  漏检@conf0.5={tot["fn_conf05"]} ({(tot["fn_conf05"]/tot["gt"])*100:.1f}%)  完全无检测={tot["fn_no_det"]}')
print(f'密集(≥5目标/图): 漏检 {miss_dense}/{gt_dense} = {miss_dense/gt_dense:.1%} | 稀疏: {miss_sparse}/{gt_sparse} = {miss_sparse/gt_sparse:.1%}')
print()
print('--- 若降低置信阈值, recall@IoU0.5 变化 ---')
for t, r in recall_at.items():
    print(f'  conf>={t}: Recall={r:.1%}')
print(f'  漏检目标中, 覆盖检测(IoU>=0.5)的最高分数分布: {bucket(miss_score_all)} (总漏检 {len(miss_score_all)})')
print()
print('--- 每类漏检 (重点 6 番茄类) ---')
for cid in [1,2,3,4,5,6,7,8]:
    v = per_cls[cid]; gt = v['gt_total']
    b = bucket(miss_score[cid]) if miss_score[cid] else {}
    print(f'{cats[cid]:30s} GT={gt:3d} 漏={v["fn_conf05"]:3d} ({v["fn_conf05"]/gt:.0%}) '
          f'漏检覆盖分数: {b}')
