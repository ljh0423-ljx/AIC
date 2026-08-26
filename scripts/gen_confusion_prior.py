#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_confusion_prior.py — 生成 train-only confusion prior (CGPM 输入, 冻结一次)
混淆关系 ONLY 来自 TRAIN (Arm B best 在 train 集评估), 不接触 VAL/TEST。
输出 experiments/direction2/confusion_prior_train_only.json
  pairs: 对每个 GT 类 g, top-K 混淆对手 r (0-based idx) + 权重 weight (归一化混淆率)
A0/A1/A2 共用同一 prior; A2 只用 pairs, 忽略 weight (统一固定 margin)。
"""
import json, os
import numpy as np

BASE = '/root/autodl-tmp/AIC2026_AgriVision'
GT_PATH = f'{BASE}/dataset/processed_detection_exiffix/annotations/train.json'
PRED_PATH = f'{BASE}/experiments/direction2/train_prior_eval/train_eval.json/bbox.json'
OUT = f'{BASE}/experiments/direction2/confusion_prior_train_only.json'
CONF_TH = 0.5
IOU_TH = 0.5
TOP_K = 2          # 每 GT 类取 top-K 混淆对手
MIN_COUNT = 1      # 最小混淆次数
MARGIN_SCALE = 0.15  # 训练时 margin 基量 (A1: m=scale*weight; A2: m=scale)

coco = json.load(open(GT_PATH))
cats = {c['id']: c['name'] for c in coco['categories']}  # 1-based
gt_by_img = {}
for a in coco['annotations']:
    gt_by_img.setdefault(a['image_id'], []).append(a)
imgs = {im['id']: im for im in coco['images']}
preds = json.load(open(PRED_PATH))

det_by_img = {}
for p in preds:
    det_by_img.setdefault(p['image_id'], []).append(p)

def iou_wh(bb1, bb2):
    x1, y1, w1, h1 = bb1; x2, y2, w2, h2 = bb2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    iw, ih = max(0, xi2-xi1), max(0, yi2-yi1)
    inter = iw*ih
    return inter / (w1*h1 + w2*h2 - inter + 1e-9)

NC = len(cats)
conf_mat = np.zeros((NC, NC), dtype=int)
for imid, gt_anns in gt_by_img.items():
    dets = sorted(det_by_img.get(imid, []), key=lambda d: -d['score'])
    dets = [d for d in dets if d['score'] >= CONF_TH]
    matched = set()
    for d in dets:
        dc = d['category_id']
        bi, biou, bgc = -1, 0.0, None
        for gi, a in enumerate(gt_anns):
            if gi in matched: continue
            iou = iou_wh(a['bbox'], d['bbox'])
            if iou > biou: bi, biou, bgc = gi, iou, a['category_id']
        if bi >= 0 and biou >= IOU_TH:
            matched.add(bi)
            conf_mat[bgc-1][dc-1] += 1

# 提取 Top-K 混淆对手 (0-based)
pairs = {}   # g(0-based) -> [{'rival': r, 'count': n, 'weight': w}]
gt_total = {}
for a in coco['annotations']:
    gt_total[a['category_id']-1] = gt_total.get(a['category_id']-1, 0) + 1
gt_total = {g: gt_total.get(g, 0) for g in range(NC)}
gt_matched = {g: int(conf_mat[g].sum()) for g in range(NC)}
for g in range(NC):
    rivals = []
    for r in range(NC):
        if r == g and conf_mat[g][r] > 0:
            pass
        if r != g and conf_mat[g][r] >= MIN_COUNT:
            rivals.append((r, int(conf_mat[g][r])))
    rivals.sort(key=lambda x: -x[1])
    rivals = rivals[:TOP_K]
    if not rivals:
        pairs[g] = []
        continue
    maxc = rivals[0][1]
    pairs[g] = [{'rival': r, 'count': c, 'weight': round(c / maxc, 4)} for r, c in rivals]

prior = {
    'meta': {
        'source': 'TRAIN only (Arm B best on train, conf=0.5, IoU=0.5)',
        'num_classes': NC,
        'top_k': TOP_K,
        'min_count': MIN_COUNT,
        'conf_th': CONF_TH,
        'iou_th': IOU_TH,
        'margin_scale': MARGIN_SCALE,
        'note': 'A1: m[g][r]=margin_scale*weight; A2: m=margin_scale (忽略 weight, 用同一 pairs)',
        'class_order': [cats[i] for i in range(1, NC+1)],
    },
    'class_id_0based_to_name': [cats[i] for i in range(1, NC+1)],
    'class_id_1based_to_name': {k: v for k, v in cats.items()},
    'gt_total_0based': {str(g): gt_total[g] for g in range(NC)},
    'gt_matched_0based': {str(g): gt_matched[g] for g in range(NC)},
    'pairs_0based': {str(g): pairs[g] for g in range(NC)},
    'confusion_matrix_train_0based': conf_mat.tolist(),
}

with open(OUT, 'w') as f:
    json.dump(prior, f, ensure_ascii=False, indent=1)

print('=== train-only confusion prior ===')
print(f'TRAIN GT 总数: {sum(gt_total.values())} (conf=0.5 匹配 {sum(gt_matched.values())}), 类数 {NC}, TOP_K={TOP_K}, MIN_COUNT={MIN_COUNT}')
print('被错分的 GT 类 → Top-K 混淆对手 (权重):')
for g in range(NC):
    if pairs[g]:
        s = '; '.join(f"{prior['class_id_0based_to_name'][p['rival']]} (c={p['count']}, w={p['weight']})" for p in pairs[g])
        print(f"  [{g}] {prior['class_id_0based_to_name'][g]:32s} GT={gt_total[g]:4d} matched={gt_matched[g]:4d} -> {s}")
print(f'保存: {OUT}')
