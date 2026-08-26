#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dir2_ablation_analysis.py — 方向2 消融 A0/A1/A2 统一 VAL 分析
仅 VAL; TEST 封闭。输入三臂 best/final 的 eval 输出, 生成:
  A0_A1_A2_metrics.csv       三臂总体指标 (mAP@.5/.5:.95, P/R/F1@conf.5, loc错类数)
  per_class_comparison.csv   13 类 AP 对比 (A0 vs A1 vs A2, Δ)
  confusion_pair_comparison.csv  命名混淆对双向混淆率对比 (含 Δ 与方向)
  confusion_matrix_{arm}.csv 各臂 VAL 混淆矩阵 (conf=.5, IoU=.5)
  *.png                      可视化
  DIRECTION2_ABLATION_REPORT.md  消融报告
"""
import json, os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/root/autodl-tmp/AIC2026_AgriVision'
EVAL = f'{BASE}/experiments/direction2/val_eval'
GT_PATH = f'{BASE}/dataset/processed_detection_exiffix/annotations/val.json'
OUT = f'{BASE}/experiments/direction2'
CONF_TH = 0.5
IOU_TH = 0.5
ARMS = ['A0', 'A1', 'A2']
TAGS = ['best', 'final']

os.makedirs(OUT, exist_ok=True)

# ---------- 加载 GT ----------
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

NAMED_PAIRS = [  # (gt_name, pred_name, 标签)
    ('Tomato leaf bacterial spot', 'Tomato Septoria leaf spot', 'bact->Septoria'),
    ('Tomato Septoria leaf spot', 'Tomato leaf bacterial spot', 'Septoria->bact'),
    ('Tomato leaf mosaic virus', 'Tomato leaf yellow virus', 'mosaic->yellow'),
    ('Tomato leaf yellow virus', 'Tomato leaf mosaic virus', 'yellow->mosaic'),
    ('Tomato Early blight leaf', 'Tomato leaf bacterial spot', 'EB->bact'),
    ('Tomato leaf bacterial spot', 'Tomato Early blight leaf', 'bact->EB'),
    ('Tomato mold leaf', 'Tomato Early blight leaf', 'mold->EB'),
    ('Tomato Early blight leaf', 'Tomato mold leaf', 'EB->mold'),
]
PAIR_GID = [(name2id[a], name2id[b], lbl) for a, b, lbl in NAMED_PAIRS]

def iou_wh(bb1, bb2):
    x1, y1, w1, h1 = bb1; x2, y2, w2, h2 = bb2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    iw, ih = max(0, xi2-xi1), max(0, yi2-yi1)
    inter = iw*ih
    return inter / (w1*h1 + w2*h2 - inter + 1e-9)

# ---------- 解析 COCO eval 日志 ----------
def parse_log(path):
    txt = open(path).read()
    def grab(pattern):
        m = re.search(pattern, txt)
        return float(m.group(1)) if m else None
    ap50_95 = grab(r'IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|\s*maxDets=\s*100\s*\] = ([\d.]+)')
    ap50 = grab(r'IoU=0\.50\s*\|\s*area=\s*all\s*\|\s*maxDets=\s*100\s*\] = ([\d.]+)')
    # per-category 表 (3 列, 按行拆分 | )
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

# ---------- 逐臂: 混淆匹配 (conf=.5, IoU=.5, greedy) ----------
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
    conf_mat = np.zeros((13, 13), dtype=int)
    tp = fp = loc_ok = cls_ok = 0
    pair_count = {lbl: 0 for _, _, lbl in PAIR_GID}
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
                if dc == bgc:
                    cls_ok += 1
                conf_mat[bgc - 1][dc - 1] += 1
                for g, p, lbl in PAIR_GID:
                    if bgc == g and dc == p:
                        pair_count[lbl] += 1
            else:
                fp += 1
    fn = len(coco['annotations']) - loc_ok
    ap50_95, ap50, per_class = parse_log(log_path)
    p, r = tp / (tp + fp), tp / (tp + fn)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {
        'arm': arm, 'tag': tag,
        'ap50_95': ap50_95, 'ap50': ap50,
        'tp': tp, 'fp': fp, 'fn': fn,
        'loc_ok': loc_ok, 'cls_ok': cls_ok,
        'loc_but_wrong_cls': loc_ok - cls_ok,
        'precision': p, 'recall': r, 'f1': f1,
        'conf_mat': conf_mat, 'per_class': per_class, 'pair_count': pair_count,
    }

results = {}
for arm in ARMS:
    for tag in TAGS:
        results[f'{arm}_{tag}'] = analyze_preds(arm, tag)
        r = results[f'{arm}_{tag}']
        print(f"[{arm} {tag}] mAP@.5:.95={r['ap50_95']} mAP@.5={r['ap50']} "
              f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
              f"loc错类={r['loc_but_wrong_cls']}")

# ---------- CSV 1: 三臂总体指标 (best + final) ----------
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
df_metrics.to_csv(f'{OUT}/A0_A1_A2_metrics.csv', index=False)

# ---------- CSV 2: 13 类 AP 对比 (best) ----------
pc = {arm: results[f'{arm}_best']['per_class'] for arm in ARMS}
rows = []
for name in CAT_NAMES:
    rows.append({'category': name, 'A0_AP': pc['A0'].get(name, np.nan),
                 'A1_AP': pc['A1'].get(name, np.nan), 'A2_AP': pc['A2'].get(name, np.nan),
                 'A1-A0': round(pc['A1'].get(name, 0) - pc['A0'].get(name, 0), 3),
                 'A2-A0': round(pc['A2'].get(name, 0) - pc['A0'].get(name, 0), 3)})
df_pc = pd.DataFrame(rows)
df_pc.to_csv(f'{OUT}/per_class_comparison.csv', index=False)

# ---------- CSV 3: 混淆对双向混淆率对比 (best) ----------
rows = []
for g, p, lbl in PAIR_GID:
    gt_total = gt_totals[g]
    rec = {'pair': lbl, 'gt_total': gt_total,
           'A0_count': results['A0_best']['pair_count'][lbl],
           'A1_count': results['A1_best']['pair_count'][lbl],
           'A2_count': results['A2_best']['pair_count'][lbl],
           'A0_rate': results['A0_best']['pair_count'][lbl] / gt_total,
           'A1_rate': results['A1_best']['pair_count'][lbl] / gt_total,
           'A2_rate': results['A2_best']['pair_count'][lbl] / gt_total}
    rec['A1-A0'] = round(rec['A1_rate'] - rec['A0_rate'], 4)
    rec['A2-A0'] = round(rec['A2_rate'] - rec['A0_rate'], 4)
    rows.append(rec)
df_pair = pd.DataFrame(rows)
df_pair.to_csv(f'{OUT}/confusion_pair_comparison.csv', index=False)

# ---------- 混淆矩阵 CSV ----------
for arm in ARMS:
    cm = results[f'{arm}_best']['conf_mat']
    with open(f'{OUT}/confusion_matrix_{arm}.csv', 'w') as f:
        f.write('gt\\\\pred,' + ','.join(CAT_NAMES) + '\n')
        for g in range(13):
            f.write(CAT_NAMES[g] + ',' + ','.join(str(cm[g][p]) for p in range(13)) + '\n')

# ---------- 可视化 (英文标签, 避免 CJK 字体缺失) ----------
def heat(ax, cm, title):
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(13)); ax.set_yticks(range(13))
    ax.set_xticklabels([f'{i+1}' for i in range(13)], fontsize=7)
    ax.set_yticklabels([f'{i+1}' for i in range(13)], fontsize=7)
    for i in range(13):
        for j in range(13):
            if cm[i][j]:
                ax.text(j, i, str(cm[i][j]), ha='center', va='center', fontsize=4)
    ax.set_xlabel('pred (1-13, see CSV for names)', fontsize=7)
    ax.set_ylabel('gt', fontsize=7)
    return im

fig, axes = plt.subplots(1, 3, figsize=(21, 7))
for k, arm in enumerate(ARMS):
    heat(axes[k], results[f'{arm}_best']['conf_mat'], f'{arm} (best) VAL confusion matrix')
fig.tight_layout()
fig.savefig(f'{OUT}/confusion_matrix_A0_A1_A2.png', dpi=120)
plt.close(fig)

# 每类 AP 对比柱状图
x = np.arange(13); w = 0.27
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x - w, [pc['A0'].get(n, 0) for n in CAT_NAMES], w, label='A0')
ax.bar(x, [pc['A1'].get(n, 0) for n in CAT_NAMES], w, label='A1')
ax.bar(x + w, [pc['A2'].get(n, 0) for n in CAT_NAMES], w, label='A2')
ax.set_xticks(x); ax.set_xticklabels([f'{i+1}: {n.split()[-1]}' for i, n in enumerate(CAT_NAMES)], rotation=90, fontsize=6)
ax.set_ylabel('AP@0.5:0.95'); ax.legend(); ax.set_title('Per-class AP comparison (VAL best)')
fig.tight_layout(); fig.savefig(f'{OUT}/per_class_ap_comparison.png', dpi=120); plt.close(fig)

# 混淆对混淆率变化
fig, ax = plt.subplots(figsize=(12, 5))
labels = df_pair['pair'].tolist()
xa = np.arange(len(labels))
ax.plot(xa, df_pair['A0_rate'] * 100, 'o-', label='A0')
ax.plot(xa, df_pair['A1_rate'] * 100, 's-', label='A1')
ax.plot(xa, df_pair['A2_rate'] * 100, '^-', label='A2')
ax.set_xticks(xa); ax.set_xticklabels(labels, rotation=30, fontsize=8)
ax.set_ylabel('Confusion rate (% of GT)'); ax.legend(); ax.set_title('Named confusion-pair rates (VAL best)')
fig.tight_layout(); fig.savefig(f'{OUT}/confusion_pair_rates.png', dpi=120); plt.close(fig)

print(f'\n完成: {OUT}/ (CSVs + PNGs)')
