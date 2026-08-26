#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
direction4_error_cases.py — 方向4 密集场景典型错误案例图 (A0 best, VAL only)
按 5 类: lowconf_miss / nms_suspect / class_confusion / small_target / heavy_overlap
每图标注: GT(实线框+类id)、预测框 conf>=0.1(虚线, 类id+score)。
TEST 封闭; 原始数据不修改。输出 experiments/direction4/error_cases/<case_id>_<class>.jpg
"""
import json, os
from collections import defaultdict, Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BASE = '/root/autodl-tmp/AIC2026_AgriVision'
GT_PATH = f'{BASE}/dataset/processed_detection_exiffix/annotations/val.json'
BBOX = f'{BASE}/experiments/ablation_dir3_B1/val_eval/A0_best.json/bbox.json'
IMG_ROOT = f'{BASE}/dataset/processed_detection_exiffix/images/val'
OUT = f'{BASE}/experiments/direction4/error_cases'
os.makedirs(OUT, exist_ok=True)
CONF_TH, IOU_TH = 0.5, 0.5
IMG_CAT = ['lowconf_miss', 'nms_suspect', 'class_confusion', 'small_target', 'heavy_overlap']

coco = json.load(open(GT_PATH))
cats = {c['id']: c['name'] for c in coco['categories']}
name2id = {v: k for k, v in cats.items()}
imgs = {im['id']: im for im in coco['images']}
gt_by = defaultdict(list)
for a in coco['annotations']: gt_by[a['image_id']].append(a)
det_by = defaultdict(list)
for p in json.load(open(BBOX)): det_by[p['image_id']].append(p)

def iou(bb1, bb2):
    x1, y1, w1, h1 = bb1; x2, y2, w2, h2 = bb2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    iw, ih = max(0, xi2-xi1), max(0, yi2-yi1)
    inter = iw*ih
    return inter / (w1*h1 + w2*h2 - inter + 1e-9)

def size_of(a, im):
    r = a['bbox'][2]*a['bbox'][3]/(im['width']*im['height'])
    return 'small' if r < 0.005 else ('medium' if r < 0.02 else 'large')

# ---- 每图: GT 状态 + 原因标签 ----
img_cases = defaultdict(list)   # imid -> list of (gt, tag)
for imid, anns in gt_by.items():
    dens = len(anns)
    dets = det_by.get(imid, [])
    used = set()
    for a in anns:  # conf0.5 匹配
        best, bi = -1, -1
        for di, d in enumerate(dets):
            if di in used or d['score'] < CONF_TH: continue
            io = iou(a['bbox'], d['bbox'])
            if io > best: best, bi = io, di
        if best >= IOU_TH: used.add(bi)
    for a in anns:
        cover, bs = None, 0.0
        for d in dets:
            io = iou(a['bbox'], d['bbox'])
            if io >= IOU_TH and d['score'] > bs: bs, cover = d['score'], d
        matched = (cover is not None and bs >= CONF_TH)
        tags = []
        if not matched:
            if cover is None: tags.append('nocover')
            elif cover['category_id'] == a['category_id']: tags.append('lowconf_miss')
            else: tags.append('class_confusion')
        # 重叠: 与其他 GT IoU>0.3
        ovl = any(iou(a['bbox'], b['bbox']) >= 0.3 for b in anns if b is not a)
        if ovl: tags.append('heavy_overlap')
        if size_of(a, imgs[imid]) == 'small': tags.append('small_target')
        # NMS疑似: 漏检且有 >=0.4 高置信检测覆盖 或 与另一GT >=0.4 重叠
        if not matched:
            b_overlap = any(iou(a['bbox'], d['bbox']) >= 0.4 and d['score'] >= 0.5 for d in dets)
            a_overlap = any(iou(a['bbox'], b['bbox']) >= 0.4 for b in anns if b is not a)
            if b_overlap or a_overlap: tags.append('nms_suspect')
        for t in tags:
            img_cases[imid].append((a, t))

# ---- 类别证据数排序选择 ----
def pick(n_need, cat):
    """优先多证据图, 返回 (imid, 证据GT列表) 列表, 每图最多一个主要GT做标注重点"""
    cand = sorted(img_cases.items(), key=lambda kv: -sum(1 for _, t in kv[1] if t == cat))
    out = []
    for imid, lst in cand:
        ev = [(a, t) for a, t in lst if t == cat]
        if ev:
            out.append((imid, ev))
        if len(out) >= n_need: break
    return out

sel = {}
sel['lowconf_miss'] = pick(8, 'lowconf_miss')
sel['class_confusion'] = pick(7, 'class_confusion')
sel['small_target'] = pick(2, 'small_target')
sel['heavy_overlap'] = pick(2, 'heavy_overlap')
sel['nms_suspect'] = pick(3, 'nms_suspect')
# nms 太稀, 补拥挤(中心距<0.25diag)密集图
if len(sel['nms_suspect']) < 3:
    extra = []
    for imid, anns in gt_by.items():
        if len(anns) < 5: continue
        W, H = imgs[imid]['width'], imgs[imid]['height']
        diag = np.hypot(W, H)
        near = 0
        for i in range(len(anns)):
            for j in range(i+1, len(anns)):
                a, b = anns[i], anns[j]
                ca = (a['bbox'][0]+a['bbox'][2]/2, a['bbox'][1]+a['bbox'][3]/2)
                cb = (b['bbox'][0]+b['bbox'][2]/2, b['bbox'][1]+b['bbox'][3]/2)
                if np.hypot(ca[0]-cb[0], ca[1]-cb[1])/diag < 0.22: near += 1
        if near >= 2:
            extra.append((imid, near))
    extra.sort(key=lambda x: -x[1])
    used_ids = {imid for imid, _ in sel['nms_suspect']}
    for imid, n in extra:
        if imid in used_ids: continue
        sel['nms_suspect'].append((imid, [(a, 'nms_suspect') for a in gt_by[imid][:3]]))
        if len(sel['nms_suspect']) >= 3: break

# ---- 绘图 ----
cat_cn = {'lowconf_miss': 'LOWCONF_MISS', 'nms_suspect': 'NMS_SUSPECT/CROWDING',
          'class_confusion': 'CLASS_CONFUSION', 'small_target': 'SMALL_TARGET',
          'heavy_overlap': 'HEAVY_OVERLAP'}
cmap = plt.get_cmap('tab20', 13)
count = 0
for cat, cases in sel.items():
    for imid, ev in cases:
        im = imgs[imid]
        fn = os.path.join(IMG_ROOT, im['file_name'])
        if not os.path.exists(fn): continue
        img = plt.imread(fn)
        fig, ax = plt.subplots(figsize=(13, 10))
        ax.imshow(img)
        dets = [d for d in det_by.get(imid, []) if d['score'] >= 0.1]
        # 预测框(虚线, 类色)
        for d in dets:
            x, y, w, h = d['bbox']
            rect = Rectangle((x, y), w, h, fill=False, ls='--', lw=1.2,
                             edgecolor=cmap((d['category_id']-1)/13))
            ax.add_patch(rect)
            ax.text(x, y-2, f"{d['category_id']}:{d['score']:.2f}", fontsize=6.5,
                    color=cmap((d['category_id']-1)/13), va='bottom')
        # GT 框(绿实线 + 原因高亮)
        ev_gt_ids = {id(a): a for a, _ in ev}
        for a in gt_by[imid]:
            x, y, w, h = a['bbox']
            is_ev = id(a) in ev_gt_ids
            ec = '#ff2222' if is_ev else '#00aa00'
            lw = 2.6 if is_ev else 1.4
            rect = Rectangle((x, y), w, h, fill=False, edgecolor=ec, lw=lw)
            ax.add_patch(rect)
            tag = next((t for t_ in ev if t_[0] is a for t in [t_[1]]), '')
            ax.text(x, y+h+4, f'GT{a["category_id"]}' + (f'[{tag}]' if is_ev else ''),
                    fontsize=7, color=ec, va='top')
        ax.set_title(f'{cat_cn[cat]} | {im["file_name"]} targets={len(gt_by[imid])} '
                     f'| green=GT red=[reason] dashed=pred conf>=0.1', fontsize=10)
        ax.axis('off')
        fig.tight_layout()
        count += 1
        fig.savefig(f'{OUT}/{count:02d}_{cat}_img{imid}.jpg', dpi=110, bbox_inches='tight')
        plt.close(fig)
print(f'生成 {count} 张错误案例图 -> {OUT}')
# 每类计数
for cat, cases in sel.items():
    print(f'  {cat_cn[cat]:14s}: {len(cases)} 图')
