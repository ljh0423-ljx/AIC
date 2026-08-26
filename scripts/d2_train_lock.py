#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
d2_train_lock.py — D2 参数在 TRAIN 上锁定(网格搜索, 仅 TRAIN)
规则(ABLATION_DIRECTION4_PLAN.md §3): 只接受 TRAIN mAP 不降且 Recall 提升的组合, 取 TRAIN mAP 最高者冻结。
禁止依据 VAL 选择参数。
输出: experiments/direction4/d2_results/params_locked.json + train_lock_results.csv
"""
import json, os, sys, itertools
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from d2_operator import apply_operator

BASE = '/root/autodl-tmp/AIC2026_AgriVision'
OUT = f'{BASE}/experiments/direction4/d2_results'
os.makedirs(OUT, exist_ok=True)
TRAIN_GT = f'{BASE}/dataset/processed_detection_exiffix/annotations/train.json'
BBOX = f'{OUT}/train_preds/train_preds.json/bbox.json'
CONF_TH, IOU_TH = 0.5, 0.5

# ---- 加载 ----
coco = json.load(open(TRAIN_GT))
img_meta = {im['id']: (im['width'], im['height']) for im in coco['images']}
preds = json.load(open(BBOX))
print(f'TRAIN: {len(coco["images"])} 图 / {len(coco["annotations"])} GT / {len(preds)} 候选框')


def iou(bb1, bb2):
    x1, y1, w1, h1 = bb1; x2, y2, w2, h2 = bb2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    iw, ih = max(0, xi2 - xi1), max(0, yi2 - yi1)
    inter = iw * ih
    return inter / (w1 * h1 + w2 * h2 - inter + 1e-9)


def recall_conf05(preds_):
    from collections import defaultdict
    by_img = defaultdict(list)
    for p in preds_:
        by_img[p['image_id']].append(p)
    gt_by = defaultdict(list)
    for a in coco['annotations']:
        gt_by[a['image_id']].append(a)
    tp = 0
    for imid, anns in gt_by.items():
        dets = sorted([d for d in by_img.get(imid, []) if d['score'] >= CONF_TH],
                      key=lambda d: -d['score'])
        used = set()
        for d in dets:
            bi, biou = -1, 0.0
            for gi, a in enumerate(anns):
                if gi in used: continue
                io = iou(a['bbox'], d['bbox'])
                if io > biou: bi, biou = gi, io
            if bi >= 0 and biou >= IOU_TH:
                used.add(bi); tp += 1
    return tp / len(coco['annotations'])


def map_coco(preds_):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    cc = COCO(TRAIN_GT)
    ev = COCOeval(cc, cc.loadRes(preds_), 'bbox')
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev.stats[0], ev.stats[1]


# ---- 基线 ----
base_map, base_map50 = map_coco(preds)
base_rec = recall_conf05(preds)
print(f'[基线] TRAIN mAP@.5:.95={base_map:.4f} mAP@.5={base_map50:.4f} Recall@conf.5={base_rec:.4f}')

rows = []
def test(name, params, use_relabel, use_rescore, alpha, straggler, tau, lam, min_agg):
    np_, st = apply_operator(preds, img_meta, alpha=alpha, straggler=straggler,
                             use_rescore=use_rescore, tau=tau, lam=lam, min_agg=min_agg)
    m, m50 = map_coco(np_)
    r = recall_conf05(np_)
    rows.append({'name': name, 'alpha': alpha, 'straggler': straggler, 'tau': tau,
                 'lam': lam, 'min_agg': min_agg, 'use_rescore': use_rescore,
                 'relabeled': st['relabeled'], 'rescored': st['rescored'],
                 'mAP': round(m, 4), 'mAP50': round(m50, 4), 'recall': round(r, 4),
                 'dMAP': round(m - base_map, 4), 'dRec': round(r - base_rec, 4)})
    flag = '  <- 候选' if (m >= base_map and r > base_rec) else ''
    print(f'{name:34s} mAP={m:.4f}(Δ{m-base_map:+.4f}) R={r:.4f}(Δ{r-base_rec:+.4f}) '
          f'重标={st["relabeled"]} 重分={st["rescored"]}{flag}')

# 阶段A: 仅保守重标
for alpha, straggler in itertools.product([0.5, 0.6, 0.7, 0.8], [0.2, 0.3, 0.5]):
    test(f'relabel a={alpha} s={straggler}', {}, True, False, alpha, straggler, 0.2, 0.1, 0.8)

# 阶段B: 仅重打分
for tau, lam, min_agg in itertools.product([0.15, 0.2, 0.25], [0.05, 0.1], [0.5, 0.8]):
    test(f'rescore t={tau} l={lam} m={min_agg}', {}, False, True, 0.6, 0.3, tau, lam, min_agg)

df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/train_lock_results.csv', index=False)

# ---- 选择: TRAIN mAP 不降 且 Recall 提升, 取最高 mAP ----
cand = df[(df['mAP'] >= base_map) & (df['recall'] > base_rec)]
if cand.empty:
    print('\n[锁定结果] TRAIN 上无任何组合同时满足"mAP 不降且 Recall 提升" → D2 退化为恒等算子(no-op), 参数不启用。')
    locked = {'enabled': False, 'alpha': None, 'straggler': None, 'tau': None,
              'lam': None, 'min_agg': None, 'use_rescore': False,
              'reason': 'TRAIN 网格搜索无候选组合(TRAIN mAP 不降且 Recall 提升)', 'train_base': base_map}
else:
    best = cand.loc[cand['mAP'].idxmax()]
    locked = {'enabled': True,
              'alpha': float(best['alpha']), 'straggler': float(best['straggler']),
              'tau': float(best['tau']), 'lam': float(best['lam']),
              'min_agg': float(best['min_agg']),
              'use_rescore': bool(best['use_rescore']),
              'name': best['name'], 'train_mAP': float(best['mAP']),
              'train_recall': float(best['recall']), 'train_base_mAP': base_map}
    print(f'\n[锁定结果] 采用 {best["name"]}: TRAIN mAP={best["mAP"]:.4f} (Δ{best["dMAP"]:+.4f}) '
          f'Recall={best["recall"]:.4f} (Δ{best["dRec"]:+.4f})')

locked['train_base'] = {'mAP': base_map, 'mAP50': base_map50, 'recall': base_rec}
json.dump(locked, open(f'{OUT}/params_locked.json', 'w'), ensure_ascii=False, indent=1)
print('保存: params_locked.json + train_lock_results.csv')
