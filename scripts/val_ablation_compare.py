#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
val_ablation_compare.py — Arm A (clean) vs Arm B (exiffix) 的 VAL 消融对比
仅 VAL; TEST 完全封闭。两臂 val GT 逐字节相同 (JSON 未改), 唯一差异 = 数据修复因子。

输入 (自动定位):
  GT  = dataset/processed_detection_clean/annotations/val.json
  preds = experiments/ablation_data_{clean,exiffix}/metrics/val_eval_{best,final}/bbox.json
输出 (experiments/ablation_compare/):
  ablation_exif_perclass.csv     每臂每类 AP50/AP50:95/P/R/F1
  ablation_exif_img65.csv        img65 (TRAIN_000054) 两臂逐检测
  ABLATION_VAL_SUMMARY.md        mAP/AR/P/R/F1 + per-class 差异 + 结论建议
  ablation_perclass_ap.png       13 类 AP50:95 双臂条形对比
  ablation_exif_recall.png       EXIF 图 recall 对比
"""
import json, os, sys, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

BASE = '/root/autodl-tmp/AIC2026_AgriVision'
GT_PATH = f'{BASE}/dataset/processed_detection_clean/annotations/val.json'
OUT = f'{BASE}/experiments/ablation_compare'
CONF_TH = 0.5
IOU_TH = 0.5
ARMS = ['clean', 'exiffix']
TAGS = ['best', 'final']
EXIF_VAL_FILE = 'TRAIN_000054_happier-inside.jpg'   # img65, 唯一在 VAL 的 EXIF 图

os.makedirs(OUT, exist_ok=True)

# ---------- 1. 加载 GT ----------
coco = COCO(GT_PATH)
cat_info = {c['id']: c['name'] for c in coco.dataset['categories']}
cats = sorted(cat_info.keys())
name2id = {v: k for k, v in cat_info.items()}
imgs = {im['id']: im for im in coco.dataset['images']}
gt_by_img = {}
for a in coco.dataset['annotations']:
    gt_by_img.setdefault(a['image_id'], []).append(a)

def find_preds(arm, tag):
    hits = (glob.glob(f'{BASE}/experiments/ablation_data_{arm}/metrics/val_eval_{tag}/bbox.json') +
            glob.glob(f'{BASE}/experiments/ablation_data_{arm}/metrics/val_eval_{tag}.json/bbox.json'))
    if not hits:
        raise FileNotFoundError(f'无预测: ablation_data_{arm}/metrics/val_eval_{tag}/bbox.json')
    return json.load(open(hits[0]))

def iou_wh(bb1, bb2):
    x1, y1, w1, h1 = bb1; x2, y2, w2, h2 = bb2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    iw, ih = max(0, xi2-xi1), max(0, yi2-yi1)
    inter = iw*ih
    return inter / (w1*h1 + w2*h2 - inter + 1e-9)

def analyze(preds):
    """返回 (per_class dict, overall dict, img_errors dict)"""
    coco_dt = coco.loadRes(preds)
    ev = COCOeval(coco, coco_dt, 'bbox'); ev.evaluate(); ev.accumulate()
    prec = ev.eval['precision']  # [T,R,K,A,M]
    rec = ev.eval['recall']      # [T,K,A,M]
    ap_all, ap50, ar100 = {}, {}, {}
    for i, cid in enumerate(cats):
        p = prec[:, :, i, 0, 2]; p = p[p > -1]
        ap_all[cid] = float(p.mean()) if len(p) else 0.0
        p50 = prec[0, :, i, 0, 2]; p50 = p50[p50 > -1]
        ap50[cid] = float(p50.mean()) if len(p50) else 0.0
        r = rec[:, i, 0, 2]; r = r[r > -1]
        ar100[cid] = float(r.mean()) if len(r) else 0.0
    mAP = float(np.mean([ap_all[c] for c in cats]))
    mAP50 = float(np.mean([ap50[c] for c in cats]))
    # conf=0.5 / IoU=0.5 greedy 匹配
    tp = {c: 0 for c in cats}; fp = {c: 0 for c in cats}; fn = {c: 0 for c in cats}
    img_err = {}
    for imid, gt_anns in gt_by_img.items():
        det = sorted([p for p in preds if p['image_id'] == imid], key=lambda d: -d['score'])
        det = [d for d in det if d['score'] >= CONF_TH]
        matched = set()
        per = {'image_id': imid, 'file': imgs[imid]['file_name'], 'n_gt': len(gt_anns),
               'n_det': len(det), 'tp': 0, 'fp': 0, 'fn': len(gt_anns), 'det': []}
        for d in det:
            dc = d['category_id']
            bi, biou, bgc = -1, 0.0, None
            for gi, a in enumerate(gt_anns):
                if gi in matched: continue
                iou = iou_wh(a['bbox'], d['bbox'])
                if iou > biou: bi, biou, bgc = gi, iou, a['category_id']
            if bi >= 0 and biou >= IOU_TH:
                matched.add(bi)
                per['tp'] += 1; per['fn'] -= 1
                per['det'].append({'gt': bgc, 'pred': dc, 'iou': round(biou, 3), 'score': round(d['score'], 3)})
                if dc == bgc: tp[dc] += 1
                else: fn[bgc] += 1; fp[dc] += 1
            else:
                fp[dc] += 1; per['fp'] += 1
                per['det'].append({'gt': None, 'pred': dc, 'iou': round(biou, 3), 'score': round(d['score'], 3)})
        for gi, a in enumerate(gt_anns):
            if gi not in matched:
                fn[a['category_id']] += 1
                per['det'].append({'gt': a['category_id'], 'pred': None, 'iou': None, 'score': None})
        img_err[imid] = per
    P = {c: tp[c]/(tp[c]+fp[c]) if tp[c]+fp[c] else 0.0 for c in cats}
    R = {c: tp[c]/(tp[c]+fn[c]) if tp[c]+fn[c] else 0.0 for c in cats}
    F1 = {c: 2*P[c]*R[c]/(P[c]+R[c]) if P[c]+R[c] else 0.0 for c in cats}
    tp_all = sum(tp.values()); fp_all = sum(fp.values()); fn_all = sum(fn.values())
    overall = dict(mAP=mAP, mAP50=mAP50, P=tp_all/(tp_all+fp_all) if tp_all+fp_all else 0,
                   R=tp_all/(tp_all+fn_all) if tp_all+fn_all else 0,
                   F1=2*tp_all/(2*tp_all+fp_all+fn_all) if tp_all+fp_all+fn_all else 0,
                   TP=tp_all, FP=fp_all, FN=fn_all)
    return dict(ap_all=ap_all, ap50=ap50, ar100=ar100, P=P, R=R, F1=F1, overall=overall, img_err=img_err)

# ---------- 2. 运行两臂 ----------
res = {}
for arm in ARMS:
    for tag in TAGS:
        preds = find_preds(arm, tag)
        key = f'{arm}_{tag}'
        print(f'[分析] {key}: {len(preds)} detections')
        res[key] = analyze(preds)

# ---------- 3. per-class 汇总 CSV ----------
rows = []
for cid in cats:
    row = {'cat': cid, 'class': cat_info[cid], 'gt': len([a for a in coco.dataset['annotations'] if a['category_id'] == cid])}
    for arm in ARMS:
        for tag in TAGS:
            k = f'{arm}_{tag}'
            row[f'{arm}_{tag}_AP5095'] = res[k]['ap_all'][cid]
            row[f'{arm}_{tag}_AP50'] = res[k]['ap50'][cid]
            row[f'{arm}_{tag}_P'] = res[k]['P'][cid]
            row[f'{arm}_{tag}_R'] = res[k]['R'][cid]
            row[f'{arm}_{tag}_F1'] = res[k]['F1'][cid]
    # 关键差异: exiffix_best vs clean_best
    d = res['exiffix_best']['ap_all'][cid] - res['clean_best']['ap_all'][cid]
    row['delta_best_AP5095'] = d
    rows.append(row)
import csv
with open(f'{OUT}/ablation_exif_perclass.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f'[写] {OUT}/ablation_exif_perclass.csv')

# ---------- 4. img65 (EXIF 图) recall 对比 ----------
def exif_img_analysis(key):
    for imid, per in res[key]['img_err'].items():
        if per['file'] == EXIF_VAL_FILE:
            gt_tp = sum(1 for d in per['det'] if d['gt'] is not None and d['pred'] == d['gt'])
            gt_fn = sum(1 for d in per['det'] if d['pred'] is None)
            gt_conf = sum(1 for d in per['det'] if d['gt'] is not None and d['pred'] is not None and d['pred'] != d['gt'])
            return {'n_gt': per['n_gt'], 'n_det': per['n_det'], 'recall': gt_tp / per['n_gt'] if per['n_gt'] else 0,
                    'tp': gt_tp, 'fn': gt_fn, 'confused': gt_conf, 'det_list': per['det']}
    return None

img65 = {}
for arm in ARMS:
    img65[arm] = {tag: exif_img_analysis(f'{arm}_{tag}') for tag in TAGS}
with open(f'{OUT}/ablation_exif_img65.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['arm', 'tag', 'n_gt', 'n_det', 'recall', 'tp', 'fn', 'confused'])
    for arm in ARMS:
        for tag in TAGS:
            r = img65[arm][tag]
            w.writerow([arm, tag, r['n_gt'], r['n_det'], round(r['recall'], 3), r['tp'], r['fn'], r['confused']])
print(f'[写] {OUT}/ablation_exif_img65.csv')

# ---------- 5. 图表 ----------
fig, ax = plt.subplots(figsize=(11, 6))
cls_names = [cat_info[c][:28] for c in cats]
x = np.arange(len(cats)); wdt = 0.38
ax.bar(x - wdt/2, [res['clean_best']['ap_all'][c] for c in cats], wdt, label='Arm A (clean)', color='#8da0cb')
ax.bar(x + wdt/2, [res['exiffix_best']['ap_all'][c] for c in cats], wdt, label='Arm B (exiffix)', color='#66c2a5')
ax.set_xticks(x); ax.set_xticklabels(cls_names, rotation=45, ha='right', fontsize=7)
ax.set_ylabel('AP@0.5:0.95 (VAL)'); ax.set_title('Ablation-Data: Arm A (clean) vs Arm B (EXIF-fix) — per-class AP')
ax.legend(); ax.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig(f'{OUT}/ablation_perclass_ap.png', dpi=130); plt.close()

# ---------- 6. Markdown 报告 ----------
def fmt_overall(k):
    o = res[k]['overall']
    return f"mAP@0.5:0.95={o['mAP']:.3f}  mAP@0.5={o['mAP50']:.3f}  P={o['P']:.3f}  R={o['R']:.3f}  F1={o['F1']:.3f}  (TP={o['TP']} FP={o['FP']} FN={o['FN']})"

delta = res['exiffix_best']['overall']['mAP'] - res['clean_best']['overall']['mAP']
delta_final = res['exiffix_final']['overall']['mAP'] - res['clean_final']['overall']['mAP']
wins = sum(1 for r in rows if r['delta_best_AP5095'] > 0)
losses = sum(1 for r in rows if r['delta_best_AP5095'] < 0)

md = []
md.append('# ABLATION_VAL_SUMMARY — Arm A (clean) vs Arm B (EXIF-fix), VAL 对比')
md.append('')
md.append('- 日期: 2026-08-14; 仅 VAL 分析; TEST 全程封闭 (未触碰)')
md.append('- 唯一自变量: 10 张 train/val EXIF 图像素烘焙修复; 模型/配置/epoch=100/bs=16/lr=0.002/seed=0 完全一致')
md.append('')
md.append('## 1. 总体指标 (VAL, COCO metric)')
md.append('')
md.append('| checkpoint | 臂 | mAP@0.5:0.95 | mAP@0.5 | P (conf0.5) | R (conf0.5) | F1 | TP/FP/FN |')
md.append('|---|---|---|---|---|---|---|---|')
for arm in ARMS:
    for tag in TAGS:
        o = res[f'{arm}_{tag}']['overall']
        md.append(f"| {tag} | {'A' if arm=='clean' else 'B'} | {o['mAP']:.3f} | {o['mAP50']:.3f} | {o['P']:.3f} | {o['R']:.3f} | {o['F1']:.3f} | {o['TP']}/{o['FP']}/{o['FN']} |")
md.append('')
md.append(f'- **mAP@0.5:0.95 差异 (best)**: Arm B - Arm A = **{delta:+.4f}**  |  差异 (final): {delta_final:+.4f}')
md.append(f'- per-class AP50:95 赢/输/平 (best): **{wins} 胜 / {losses} 负 / {13-wins-losses} 平**')
md.append('')
md.append('## 2. EXIF 图 (img65 = TRAIN_000054) recall 变化')
md.append('')
md.append('> 注: 11 张 EXIF 图中仅 img65 位于 VAL (可直接测 recall); 9 张在 train (影响训练), 1 张在 test (封闭不动)。')
md.append('')
md.append('| 臂 | tag | GT | 检测数 | recall | TP | FN | 定位对类错 |')
md.append('|---|---|---|---|---|---|---|---|')
for arm in ARMS:
    for tag in TAGS:
        r = img65[arm][tag]
        md.append(f"| {'A' if arm=='clean' else 'B'} | {tag} | {r['n_gt']} | {r['n_det']} | {r['recall']:.3f} | {r['tp']} | {r['fn']} | {r['confused']} |")
md.append('')
md.append('## 3. per-class AP50:95 差异 (Arm B - Arm A, best_model)')
md.append('')
md.append('| 类 | GT | ArmA | ArmB | Δ |')
md.append('|---|---|---|---|---|')
for r in sorted(rows, key=lambda r: -abs(r['delta_best_AP5095'])):
    md.append(f"| {r['class']} | {r['gt']} | {r['clean_best_AP5095']:.3f} | {r['exiffix_best_AP5095']:.3f} | {r['delta_best_AP5095']:+.3f} |")
md.append('')
md.append('## 4. 结论与建议')
delta_win = delta > 0.005
if delta_win:
    md.append(f"- **结论: EXIF 修复有效** (VAL mAP@0.5:0.95 提升 {delta:+.4f} > 0.005) → **建议将 EXIF 修复作为后续实验的数据基础**。")
elif delta > 0:
    md.append(f"- **结论: EXIF 修复略有提升但幅度小** ({delta:+.4f} ≤ 0.005) → 方向1 属正确性修复而非主性能瓶颈; 建议作为数据管线标配 (消除错位图), 但主要提升需靠方向2/3。")
else:
    md.append(f"- **结论: EXIF 修复未见提升** ({delta:+.4f}) → 方向1 不是主要性能瓶颈, 如实记录, 不建议据此改数据管线。")
md.append('- **TEST 仍封闭**: 未用任何 VAL 结果查看/选择 TEST; 是否上 TEST 由后续决策决定。')
md.append('')
md.append('可视化: `ablation_perclass_ap.png`; 明细: `ablation_exif_perclass.csv`, `ablation_exif_img65.csv`')

with open(f'{OUT}/ABLATION_VAL_SUMMARY.md', 'w') as f:
    f.write('\n'.join(md))
print(f'[写] {OUT}/ABLATION_VAL_SUMMARY.md')

print('完成。')
