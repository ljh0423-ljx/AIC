#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VAL 集深度诊断分析脚本 — 基于 model_final 在 VAL 集的预测
仅使用 VAL (诊断), TEST 完全不触碰。
产出:
  metrics/val_per_class_metrics.csv   每类 AP50/AP50:95/P/R/F1/目标数/图片数
  metrics/val_crop_summary.csv        按作物汇总
  metrics/val_confusion_analysis.csv  混淆矩阵 + 混淆方向分析
  metrics/val_bbox_size_dist.csv      尺寸分布 (小/中/大 + 每类)
  metrics/val_image_errors.json       逐图错误分解 (供错误案例筛选)
"""
import json, os, sys
import numpy as np
import csv
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

BASE = '/root/autodl-tmp/AIC2026_AgriVision'
GT = f'{BASE}/dataset/processed_detection_clean/annotations/val.json'
PRED = f'{BASE}/experiments/baseline_v1/metrics/val_eval/bbox.json'
OUT = f'{BASE}/experiments/baseline_v1/metrics'
CONF_TH = 0.5   # P/R 与混淆矩阵的置信度阈值
IOU_TH = 0.5    # 匹配 IoU 阈值

os.makedirs(OUT, exist_ok=True)

# ---------- 1. 加载 ----------
coco = COCO(GT)
cat_info = {c['id']: c['name'] for c in coco.dataset['categories']}
cats = sorted(cat_info.keys())
name2id = {v: k for k, v in cat_info.items()}

preds = json.load(open(PRED))
# 过滤: 部分预测可能低于阈值, 全部保留由 COCOeval 自行处理; P/R 用 CONF_TH
imgs = {im['id']: im for im in coco.dataset['images']}
gt_by_img = {}
for a in coco.dataset['annotations']:
    gt_by_img.setdefault(a['image_id'], []).append(a)

# ---------- 2. COCOeval: per-class AP50 / AP50:95 ----------
coco_dt = coco.loadRes(preds)
ev = COCOeval(coco, coco_dt, 'bbox')
ev.evaluate()
ev.accumulate()

# precision[T=10iou, R=101rec, K=cls, A=4area, M=3maxDets(1,10,100)]
prec = ev.eval['precision']  # [T,R,K,A,M]
ap_all = {}
ap50 = {}
for i, cid in enumerate(cats):
    # IoU=0.5:0.95, area=all(A=0), maxDets=100(M=2) -> mean over T, R (ignore -1)
    p = prec[:, :, i, 0, 2]
    p = p[p > -1]
    ap_all[cid] = float(p.mean()) if len(p) else 0.0
    # IoU=0.50 (T=0), area=all, maxDets=100
    p50 = prec[0, :, i, 0, 2]
    p50 = p50[p50 > -1]
    ap50[cid] = float(p50.mean()) if len(p50) else 0.0

# ---------- 3. 置信度阈值下的逐图匹配 (greedy, 按 score 降序) ----------
def iou_wh(bb1, bb2):
    x1, y1, w1, h1 = bb1; x2, y2, w2, h2 = bb2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    iw, ih = max(0, xi2-xi1), max(0, yi2-yi1)
    inter = iw*ih
    u1, u2 = w1*h1, w2*h2
    return inter / (u1+u2-inter+1e-9)

# 统计
tp = {c: 0 for c in cats}; fp = {c: 0 for c in cats}; fn = {c: 0 for c in cats}
confusion = {g: {p: 0 for p in cats} for g in cats}
img_errors = {}  # image_id -> dict

for imid, gt_anns in gt_by_img.items():
    im_det = [p for p in preds if p['image_id'] == imid]
    im_det = sorted(im_det, key=lambda d: -d['score'])
    im_det = [d for d in im_det if d['score'] >= CONF_TH]
    gt_ids = list(range(len(gt_anns)))
    matched_gt = set()
    per_img = {'image_id': imid, 'file': imgs[imid]['file_name'],
               'n_gt': len(gt_anns), 'n_det': len(im_det),
               'tp': 0, 'fp': 0, 'fn': len(gt_anns),
               'confusion': [], 'missed': [], 'false': []}
    for d in im_det:
        dc = d['category_id']
        best_i, best_iou, best_gc = -1, 0.0, None
        for gi in range(len(gt_anns)):
            if gi in matched_gt: continue
            iou = iou_wh(gt_anns[gi]['bbox'], d['bbox'])
            if iou > best_iou:
                best_i, best_iou, best_gc = gi, iou, gt_anns[gi]['category_id']
        if best_i >= 0 and best_iou >= IOU_TH:
            matched_gt.add(best_i)
            if dc == best_gc:
                tp[dc] += 1; per_img['tp'] += 1; per_img['fn'] -= 1
            else:
                # 跨类匹配: 对 GT 类计 FN, 对预测类计 FP
                confusion[best_gc][dc] += 1
                fn[best_gc] += 1; fp[dc] += 1
                per_img['tp'] += 1; per_img['fn'] -= 1
                per_img['confusion'].append({'gt': best_gc, 'pred': dc,
                                             'iou': round(best_iou, 3),
                                             'score': round(d['score'], 3)})
        else:
            fp[dc] += 1; per_img['fp'] += 1
            per_img['false'].append({'pred': dc, 'score': round(d['score'], 3),
                                     'best_iou': round(best_iou, 3)})
    for gi in range(len(gt_anns)):
        if gi not in matched_gt:
            a = gt_anns[gi]
            fn[a['category_id']] += 1
            per_img['missed'].append({'gt': a['category_id'],
                                      'area': a['bbox'][2]*a['bbox'][3]})
    img_errors[imid] = per_img

# ---------- 4. 汇总 per-class ----------
def area_size(area):
    if area < 32*32: return 'small'
    if area < 96*96: return 'medium'
    return 'large'

# GT 统计 (每类目标数/图片数/尺寸分布)
gt_stat = {c: {'objects': 0, 'images': set(), 'small': 0, 'medium': 0, 'large': 0,
               'area_sum': 0.0, 'area_min': 1e9, 'area_max': 0} for c in cats}
for a in coco.dataset['annotations']:
    c = a['category_id']; b = a['bbox']; ar = b[2]*b[3]
    s = gt_stat[c]
    s['objects'] += 1; s['images'].add(a['image_id'])
    s['area_sum'] += ar; s['area_min'] = min(s['area_min'], ar); s['area_max'] = max(s['area_max'], ar)
    sz = area_size(ar); s[sz] += 1

rows = []
for cid in cats:
    p = tp[cid]/(tp[cid]+fp[cid]) if (tp[cid]+fp[cid]) else 0.0
    r = tp[cid]/(tp[cid]+fn[cid]) if (tp[cid]+fn[cid]) else 0.0
    f1 = 2*p*r/(p+r) if (p+r) else 0.0
    g = gt_stat[cid]
    rows.append({
        'class': cat_info[cid], 'cat_id': cid,
        'crop': 'Tomato' if cid <= 8 else ('Apple' if cid <= 11 else 'Grape'),
        'n_objects': g['objects'], 'n_images': len(g['images']),
        'AP50': round(ap50[cid], 3), 'AP50_95': round(ap_all[cid], 3),
        'Precision': round(p, 3), 'Recall': round(r, 3), 'F1': round(f1, 3),
        'TP': tp[cid], 'FP': fp[cid], 'FN': fn[cid],
        'small': g['small'], 'medium': g['medium'], 'large': g['large'],
        'small_ratio': round(g['small']/g['objects'], 3) if g['objects'] else 0,
        'mean_area': round(g['area_sum']/g['objects'], 1) if g['objects'] else 0,
        'min_area': round(g['area_min'], 1), 'max_area': round(g['area_max'], 1),
    })

# 写 per_class_metrics.csv
with open(f'{OUT}/val_per_class_metrics.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# 作物汇总
crop_rows = []
for crop in ['Tomato', 'Apple', 'Grape']:
    sub = [r for r in rows if r['crop'] == crop]
    n_objs = sum(r['n_objects'] for r in sub)
    n_imgs = len(set().union(*[set()]) )  # placeholder
    # 图片数按类别集合
    imset = set()
    for cid, cname in cat_info.items():
        crop_of = 'Tomato' if cid <= 8 else ('Apple' if cid <= 11 else 'Grape')
        if crop_of == crop:
            imset |= gt_stat[cid]['images']
    crop_rows.append({
        'crop': crop, 'n_classes': len(sub),
        'n_objects': n_objs, 'n_images': len(imset),
        'mAP50': round(np.mean([r['AP50'] for r in sub]), 3),
        'mAP50_95': round(np.mean([r['AP50_95'] for r in sub]), 3),
        'mPrecision': round(np.mean([r['Precision'] for r in sub]), 3),
        'mRecall': round(np.mean([r['Recall'] for r in sub]), 3),
        'mF1': round(np.mean([r['F1'] for r in sub]), 3),
        'small_ratio': round(np.mean([r['small_ratio'] for r in sub]), 3),
        'class_AP50_95': {r['class']: r['AP50_95'] for r in sub},
    })
with open(f'{OUT}/val_crop_summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(crop_rows[0].keys()))
    w.writeheader(); w.writerows(crop_rows)

# 混淆矩阵分析 csv: 每对 (gt,pred) 计数>0
conf_rows = []
for g in cats:
    for p in cats:
        n = confusion[g][p]
        if n > 0:
            # 归一化: 相对 GT 类总数
            gtot = gt_stat[g]['objects']
            conf_rows.append({'gt_class': cat_info[g], 'pred_class': cat_info[p],
                              'count': n,
                              'frac_of_gt': round(n/gtot, 3) if gtot else 0})
conf_rows.sort(key=lambda r: -r['count'])
with open(f'{OUT}/val_confusion_analysis.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['gt_class', 'pred_class', 'count', 'frac_of_gt'])
    w.writeheader(); w.writerows(conf_rows)

# 尺寸分布 csv
size_rows = []
for cid in cats:
    g = gt_stat[cid]
    size_rows.append({'class': cat_info[cid], 'objects': g['objects'],
                      'small': g['small'], 'medium': g['medium'], 'large': g['large'],
                      'small_ratio': round(g['small']/g['objects'], 3) if g['objects'] else 0})
size_rows.append({'class': 'ALL', 'objects': sum(g['objects'] for g in gt_stat.values()),
                  'small': sum(g['small'] for g in gt_stat.values()),
                  'medium': sum(g['medium'] for g in gt_stat.values()),
                  'large': sum(g['large'] for g in gt_stat.values())})
with open(f'{OUT}/val_bbox_size_dist.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(size_rows[0].keys()))
    w.writeheader(); w.writerows(size_rows)

# 逐图错误分解
json.dump({k: v for k, v in sorted(img_errors.items())},
          open(f'{OUT}/val_image_errors.json', 'w'), ensure_ascii=False, indent=1)

print("== per-class (VAL, model_final, conf=0.5) ==")
print(f"{'class':38s} {'objs':>4s} {'imgs':>4s} {'AP50':>5s} {'AP50:95':>7s} {'P':>5s} {'R':>5s} {'F1':>5s} {'TP':>3s} {'FP':>3s} {'FN':>3s} {'smallR':>6s}")
for r in rows:
    print(f"{r['class']:38s} {r['n_objects']:4d} {r['n_images']:4d} {r['AP50']:5.3f} {r['AP50_95']:7.3f} {r['Precision']:5.3f} {r['Recall']:5.3f} {r['F1']:5.3f} {r['TP']:3d} {r['FP']:3d} {r['FN']:3d} {r['small_ratio']:6.3f}")
print()
print("== 作物汇总 ==")
for r in crop_rows:
    print(f"{r['crop']:8s} n_obj={r['n_objects']:5d} n_img={r['n_images']:4d} mAP50={r['mAP50']:.3f} mAP50:95={r['mAP50_95']:.3f} mP={r['mPrecision']:.3f} mR={r['mRecall']:.3f} mF1={r['mF1']:.3f}")
print()
print("== 混淆(gt->pred, count>=3) ==")
for r in conf_rows:
    if r['count'] >= 3:
        print(f"  {r['gt_class']:32s} -> {r['pred_class']:28s} {r['count']} (占GT {r['frac_of_gt']:.1%})")
print()
print("== 逐图错误 TOP (按 n_missed+fp 排序) ==")
ierrs = sorted(img_errors.values(), key=lambda e: -(e['fn']+e['fp']))
for e in ierrs[:8]:
    print(f"  img={e['image_id']} file={e['file'][:40]:40s} gt={e['n_gt']:2d} det={e['n_det']:3d} tp={e['tp']} fp={e['fp']} fn={e['fn']}")
print("\nDONE")
