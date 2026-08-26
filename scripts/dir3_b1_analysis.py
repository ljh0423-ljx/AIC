#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dir3_b1_analysis.py — 方向3 B1 (class-weighted VFL) A0 vs B1 VAL 分析
仅 VAL; TEST 封闭。A0 = 方向2 EXIF-Fix best checkpoint (同一评估脚本重算)。
生成:
  A0_B1_metrics.csv              总体指标 (mAP@.5/.5:.95, P/R/F1@conf.5, best+final)
  per_class_B1_comparison.csv    13 类 AP + Recall 对比 (A0 vs B1, Δ)
  class_weight_effect.csv        每类: w_c × ΔAP × ΔRecall (权重-效果关联)
  strong_class_regression.csv    强类回退检查
  *.png                          可视化
"""
import json, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/root/autodl-tmp/AIC2026_AgriVision'
EVAL = f'{BASE}/experiments/ablation_dir3_B1/val_eval'
GT_PATH = f'{BASE}/dataset/processed_detection_exiffix/annotations/val.json'
WT_PATH = f'{BASE}/experiments/direction3/class_weight_train.json'
OUT = f'{BASE}/experiments/ablation_dir3_B1'
CONF_TH = 0.5
IOU_TH = 0.5
ARMS = ['A0', 'B1']
TAGS = ['best', 'final']

os.makedirs(OUT, exist_ok=True)

coco = json.load(open(GT_PATH))
cats = {c['id']: c['name'] for c in coco['categories']}
name2id = {v: k for k, v in cats.items()}
CAT_NAMES = [cats[i] for i in range(1, 14)]
imgs = {im['id']: im for im in coco['images']}
gt_by_img = {}
for a in coco['annotations']:
    gt_by_img.setdefault(a['image_id'], []).append(a)
gt_totals = {c['id']: 0 for c in coco['categories']}
for a in coco['annotations']:
    gt_totals[a['category_id']] += 1

# 强类回退监控表 (A0 最强 8 类; 阈值 ΔAP < -0.02 即告警)
STRONG = [  # (name, A0_AP 基准, 风险说明)
    ('Apple leaf', 0.811, 'w≈1.0, 低风险'),
    ('grape leaf', 0.738, 'w=1.09, 低'),
    ('Apple rust leaf', 0.650, 'w=1.27, 低'),
    ('Apple Scab Leaf', 0.484, 'w=1.51, 中(上调少样本过拟合)'),
    ('Tomato Septoria leaf spot', 0.461, 'w=0.56, 高风险(被下调的强类)'),
    ('grape leaf black rot', 0.403, 'w=1.87, 中(大幅上调 N=101)'),
    ('Tomato leaf late blight', 0.379, 'w=1.19, 低'),
    ('Tomato leaf bacterial spot', 0.284, 'w=0.83, 低'),
]
WEAK_MISS = [  # 诊断点名的弱势/漏检类
    'Tomato leaf yellow virus', 'Tomato leaf mosaic virus', 'Tomato mold leaf',
    'Tomato Septoria leaf spot', 'Tomato leaf bacterial spot',
]

def iou_wh(bb1, bb2):
    x1, y1, w1, h1 = bb1; x2, y2, w2, h2 = bb2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    iw, ih = max(0, xi2-xi1), max(0, yi2-yi1)
    inter = iw*ih
    return inter / (w1*h1 + w2*h2 - inter + 1e-9)

def parse_log(path):
    txt = open(path).read()
    def grab(pattern):
        m = re.search(pattern, txt)
        return float(m.group(1)) if m else None
    ap50_95 = grab(r'IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|\s*maxDets=\s*100\s*\] = ([\d.]+)')
    ap50 = grab(r'IoU=0\.50\s*\|\s*area=\s*all\s*\|\s*maxDets=\s*100\s*\] = ([\d.]+)')
    per_class = {}
    start = txt.find('Per-category of bbox AP:')
    if start >= 0:
        seg = txt[start:]
        seg = seg[:seg.find('\n[')] if '\n[' in seg else seg
        for line in seg.split('\n'):
            if not line.strip().startswith('|'):
                continue
            cells = [c.strip() for c in line.split('|')[1:-1]]
            for i in range(0, len(cells) - 1, 2):
                name, ap = cells[i], cells[i + 1]
                if name in name2id and ap != 'None':
                    per_class[name] = float(ap)
    return ap50_95, ap50, per_class

def analyze_preds(arm, tag):
    for cand in (f'{EVAL}/{arm}_{tag}.json/bbox.json', f'{EVAL}/{arm}_{tag}/bbox.json'):
        if os.path.exists(cand):
            bbox_path = cand
            break
    else:
        raise FileNotFoundError(f'bbox.json missing for {arm}_{tag}')
    log_path = f'{EVAL}/{arm}_{tag}.log'
    preds = json.load(open(bbox_path))
    det_by_img = {}
    for p in preds:
        det_by_img.setdefault(p['image_id'], []).append(p)
    tp = fp = loc_ok = cls_ok = 0
    cls_tp = {c['id']: 0 for c in coco['categories']}   # 每类定位正确数
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
                tp += 1; loc_ok += 1
                cls_tp[bgc] += 1
                if dc == bgc: cls_ok += 1
            else:
                fp += 1
    fn = len(coco['annotations']) - loc_ok
    ap50_95, ap50, per_class = parse_log(log_path)
    p, r = tp / (tp + fp), tp / (tp + fn)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    cls_recall = {c['id']: cls_tp[c['id']] / gt_totals[c['id']]
                  for c in coco['categories']}
    return {'arm': arm, 'tag': tag, 'ap50_95': ap50_95, 'ap50': ap50,
            'tp': tp, 'fp': fp, 'fn': fn, 'loc_ok': loc_ok, 'cls_ok': cls_ok,
            'loc_but_wrong_cls': loc_ok - cls_ok,
            'precision': p, 'recall': r, 'f1': f1,
            'per_class': per_class, 'cls_recall': cls_recall}

results = {}
for arm in ARMS:
    for tag in TAGS:
        results[f'{arm}_{tag}'] = analyze_preds(arm, tag)
        r = results[f'{arm}_{tag}']
        print(f"[{arm} {tag}] mAP@.5:.95={r['ap50_95']} mAP@.5={r['ap50']} "
              f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
              f"loc错类={r['loc_but_wrong_cls']}")

# ---------- CSV 1: 总体指标 ----------
rows = []
for arm in ARMS:
    for tag in TAGS:
        r = results[f'{arm}_{tag}']
        rows.append({'arm': arm, 'checkpoint': tag, 'mAP@0.5:0.95': r['ap50_95'],
                     'mAP@0.5': r['ap50'], 'Precision@conf0.5': r['precision'],
                     'Recall@conf0.5': r['recall'], 'F1@conf0.5': r['f1'],
                     'loc_ok': r['loc_ok'], 'cls_ok': r['cls_ok'],
                     'loc_but_wrong_cls': r['loc_but_wrong_cls'],
                     'TP': r['tp'], 'FP': r['fp'], 'FN': r['fn']})
df_metrics = pd.DataFrame(rows)
df_metrics.to_csv(f'{OUT}/A0_B1_metrics.csv', index=False)

# ---------- CSV 2: 13 类 AP + Recall 对比 (best) ----------
pc = {arm: results[f'{arm}_best']['per_class'] for arm in ARMS}
rc = {arm: results[f'{arm}_best']['cls_recall'] for arm in ARMS}
rows = []
for i, name in enumerate(CAT_NAMES):
    cid = i + 1
    rows.append({'category': name, 'A0_AP': pc['A0'].get(name, np.nan),
                 'B1_AP': pc['B1'].get(name, np.nan),
                 'dAP': round(pc['B1'].get(name, 0) - pc['A0'].get(name, 0), 3),
                 'A0_Recall': rc['A0'][cid], 'B1_Recall': rc['B1'][cid],
                 'dRecall': round(rc['B1'][cid] - rc['A0'][cid], 4)})
df_pc = pd.DataFrame(rows)
df_pc.to_csv(f'{OUT}/per_class_B1_comparison.csv', index=False)

# ---------- CSV 3: class weight × ΔAP × ΔRecall (class_weight_effect.csv) ----------
wt = json.load(open(WT_PATH))
w_arr = {str(c): wt['weight'][str(c)] for c in range(1, 14)}
rows = []
for i, name in enumerate(CAT_NAMES):
    cid = i + 1
    rows.append({'category': name, 'class_weight': w_arr[str(cid)],
                 'A0_AP': pc['A0'].get(name, np.nan), 'B1_AP': pc['B1'].get(name, np.nan),
                 'dAP': round(pc['B1'].get(name, 0) - pc['A0'].get(name, 0), 3),
                 'A0_Recall': rc['A0'][cid], 'B1_Recall': rc['B1'][cid],
                 'dRecall': round(rc['B1'][cid] - rc['A0'][cid], 4),
                 'direction': 'up' if w_arr[str(cid)] > 1.0 else 'down'})
df_eff = pd.DataFrame(rows)
df_eff.to_csv(f'{OUT}/class_weight_effect.csv', index=False)

# ---------- CSV 4: 强类回退检查 ----------
rows = []
for name, base_ap, risk in STRONG:
    dap = pc['B1'].get(name, 0) - pc['A0'].get(name, 0)
    rows.append({'category': name, 'A0_AP': base_ap, 'B1_AP': pc['B1'].get(name, np.nan),
                 'dAP': round(dap, 3), 'risk': risk,
                 'ALERT': 'YES' if dap < -0.02 else ''})
df_str = pd.DataFrame(rows)
df_str.to_csv(f'{OUT}/strong_class_regression.csv', index=False)
print('\n--- 强类回退检查 ---')
for _, row in df_str.iterrows():
    print(f"  {row['category']:32s} dAP={row['dAP']:+.3f}  {row['risk']}  {row['ALERT']}")

# ---------- 弱势类 Recall 汇总 ----------
print('\n--- 点名弱势/漏检类 Recall@conf0.5 ---')
for name in WEAK_MISS:
    cid = name2id[name]
    print(f"  {name:32s} A0={rc['A0'][cid]:.3f}  B1={rc['B1'][cid]:.3f}  "
          f"Δ={rc['B1'][cid]-rc['A0'][cid]:+.3f}")

# ---------- 可视化 (英文标签) ----------
x = np.arange(13); w = 0.38
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - w/2, [pc['A0'].get(n, 0) for n in CAT_NAMES], w, label='A0 (EXIF-Fix VFL)')
ax.bar(x + w/2, [pc['B1'].get(n, 0) for n in CAT_NAMES], w, label='B1 (class-weighted VFL)')
ax.set_xticks(x); ax.set_xticklabels([f'{i+1}: {n.split()[-1]}' for i, n in enumerate(CAT_NAMES)], rotation=90, fontsize=6)
ax.set_ylabel('AP@0.5:0.95'); ax.legend(); ax.set_title('Per-class AP: A0 vs B1 (VAL best)')
fig.tight_layout(); fig.savefig(f'{OUT}/per_class_ap_A0_vs_B1.png', dpi=120); plt.close(fig)

fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - w/2, [rc['A0'][i+1] for i in range(13)], w, label='A0')
ax.bar(x + w/2, [rc['B1'][i+1] for i in range(13)], w, label='B1')
ax.set_xticks(x); ax.set_xticklabels([f'{i+1}: {n.split()[-1]}' for i, n in enumerate(CAT_NAMES)], rotation=90, fontsize=6)
ax.set_ylabel('Recall@conf0.5'); ax.legend(); ax.set_title('Per-class Recall@conf0.5: A0 vs B1 (VAL best)')
fig.tight_layout(); fig.savefig(f'{OUT}/per_class_recall_A0_vs_B1.png', dpi=120); plt.close(fig)

# 权重-效果散点: x=w_c, y=ΔAP, 颜色=ΔRecall
wv = np.array([w_arr[str(i+1)] for i in range(13)])
dap = np.array([pc['B1'].get(n, 0) - pc['A0'].get(n, 0) for n in CAT_NAMES])
drec = np.array([rc['B1'][i+1] - rc['A0'][i+1] for i in range(13)])
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(wv, dap, c=drec, cmap='RdYlGn', s=90)
for i, n in enumerate(CAT_NAMES):
    ax.annotate(str(i+1), (wv[i], dap[i]), fontsize=7)
ax.axhline(0, color='k', lw=0.6); ax.axvline(1.0, color='k', lw=0.6, ls='--')
ax.set_xlabel('class weight w_c (TRAIN inverse frequency)')
ax.set_ylabel('B1 - A0 AP@0.5:0.95')
ax.set_title('Class-weight vs effect (color = dRecall, VAL best)')
plt.colorbar(sc, label='dRecall@conf0.5')
fig.tight_layout(); fig.savefig(f'{OUT}/class_weight_effect.png', dpi=120); plt.close(fig)

# ---------- 成功判据判定 ----------
A0 = results['A0_best']; B1 = results['B1_best']
dm = B1['ap50_95'] - A0['ap50_95']
print('\n======== 成功判据判定 (best) ========')
print(f"主判据: B1-A0 mAP@0.5:0.95 = {dm:+.4f} (需 ≥ +0.005)  "
      f"{'PASS' if dm >= 0.005 else 'FAIL'}")
r_up = B1['recall'] - A0['recall']; f1_up = B1['f1'] - A0['f1']
print(f"副判据: dRecall={r_up:+.4f}  dF1={f1_up:+.4f} (需任一 >0)  "
      f"{'PASS' if (r_up > 0 or f1_up > 0) else 'FAIL'}")
alerts = df_str[df_str['ALERT'] == 'YES']
print(f"负向红线: 强类回退告警数 = {len(alerts)}  "
      f"{'PASS(无)' if len(alerts) == 0 else 'FAIL(有)'}")
for _, a in alerts.iterrows():
    print(f"    ⚠ {a['category']} dAP={a['dAP']:+.3f}")
print('\n完成: OUT 目录 (CSVs + PNGs)')
