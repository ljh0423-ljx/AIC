#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
direction4_diagnosis.py — 方向4 密集场景深度诊断 (A0/EXIF-Fix best, VAL only)
按用户 11 项要求统计, 输出 experiments/direction4/d4_diagnostic.json + 控制台汇总。
TEST 封闭; 原始数据集不修改; 不做任何训练/推理修改。
"""
import json, os
import numpy as np
from collections import Counter, defaultdict

BASE = '/root/autodl-tmp/AIC2026_AgriVision'
GT_PATH = f'{BASE}/dataset/processed_detection_exiffix/annotations/val.json'
BBOX = f'{BASE}/experiments/ablation_dir3_B1/val_eval/A0_best.json/bbox.json'
OUT = f'{BASE}/experiments/direction4'
os.makedirs(OUT, exist_ok=True)
CONF_TH = 0.5
IOU_TH = 0.5

coco = json.load(open(GT_PATH))
cats = {c['id']: c['name'] for c in coco['categories']}
CAT_NAMES = [cats[i] for i in range(1, 14)]
imgs = {im['id']: im for im in coco['images']}
gt_by_img = defaultdict(list)
for a in coco['annotations']:
    gt_by_img[a['image_id']].append(a)
gt_totals = Counter(a['category_id'] for a in coco['annotations'])
preds = json.load(open(BBOX))
det_by_img = defaultdict(list)
for p in preds:
    det_by_img[p['image_id']].append(p)

def iou_wh(bb1, bb2):
    x1, y1, w1, h1 = bb1; x2, y2, w2, h2 = bb2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    iw, ih = max(0, xi2-xi1), max(0, yi2-yi1)
    inter = iw*ih
    return inter / (w1*h1 + w2*h2 - inter + 1e-9)

def area_ratio(a, im):
    return a['bbox'][2] * a['bbox'][3] / (im['width'] * im['height'])

def size(a, im):
    r = area_ratio(a, im)
    if r < 0.005: return 'small'
    if r < 0.02: return 'medium'
    return 'large'

# ---------- 图像密度 ----------
img_density = {imid: len(v) for imid, v in gt_by_img.items()}
DENSITY_BUCKETS = [(1, '1'), (2, '2'), (3, '4', '3-4'), (5, 9, '5-9'), (10, 9999, '10+')]

def bucket_of(n):
    if n == 1: return '1'
    if n == 2: return '2'
    if n <= 4: return '3-4'
    if n <= 9: return '5-9'
    return '10+'

# 候选框 conf 带
def band(s):
    if s < 0.1: return 'lt0.1'
    if s < 0.3: return '0.1-0.3'
    if s < 0.5: return '0.3-0.5'
    return 'ge0.5'
BANDS = ['lt0.1', '0.1-0.3', '0.3-0.5', 'ge0.5']

# 全量候选按带+类统计
cand_band_class = {b: Counter() for b in BANDS}
cand_band = {b: 0 for b in BANDS}
for p in preds:
    b = band(p['score'])
    cand_band[b] += 1
    cand_band_class[b][p['category_id']] += 1

# ---------- 匹配辅助 ----------
def match_gt(imid, conf=CONF_TH):
    """返回每 GT: (matched, matched_pred_score, best_iou_any, best_score_any)"""
    dets = sorted([d for d in det_by_img.get(imid, []) if d['score'] >= conf],
                  key=lambda d: -d['score'])
    used = set()
    res = []
    for gi, a in enumerate(gt_by_img[imid]):
        best_iou, best_sc, best_cat = 0.0, 0.0, None
        bi = -1
        for di, d in enumerate(dets):
            if di in used: continue
            iou = iou_wh(a['bbox'], d['bbox'])
            if iou > best_iou: best_iou, best_sc, best_cat, bi = iou, d['score'], d['category_id'], di
        if best_iou >= IOU_TH:
            used.add(bi)
            res.append((True, best_sc, best_iou, best_cat))
        else:
            res.append((False, best_sc, best_iou, best_cat))
    return res

# 每图统计
per_img = {}
for imid, anns in gt_by_img.items():
    dens = len(anns)
    res = match_gt(imid)
    per_img[imid] = {'density': dens, 'n_miss05': sum(1 for m, *_ in res if not m),
                     'gt_list': anns, 'match_res': res}

tot_miss05 = sum(v['n_miss05'] for v in per_img.values())
tot_gt = len(coco['annotations'])

# ---------- 1) 每图目标数 vs 漏检率 ----------
dens_miss = Counter()
dens_gt = Counter()
for v in per_img.values():
    b = bucket_of(v['density'])
    dens_gt[b] += v['density']
    dens_miss[b] += v['n_miss05']
item1 = {b: {'gt': dens_gt[b], 'miss': dens_miss[b],
             'miss_rate': round(dens_miss[b]/dens_gt[b], 4)} for b in dens_gt}

# ---------- 2) 密度区间 Recall/Precision/F1 ----------
# 用整体匹配算 FP: 对每个 conf, 每图按分数降序贪心匹配, 统计 TP/FP/FN
def prec_recall_f1(conf):
    tp = fp = 0
    for imid, anns in gt_by_img.items():
        dets = sorted([d for d in det_by_img.get(imid, []) if d['score'] >= conf],
                      key=lambda d: -d['score'])
        used = set()
        for d in dets:
            bi, biou = -1, 0.0
            for gi, a in enumerate(anns):
                if gi in used: continue
                iou = iou_wh(a['bbox'], d['bbox'])
                if iou > biou: bi, biou = gi, iou
            if bi >= 0 and biou >= IOU_TH:
                used.add(bi); tp += 1
            else:
                fp += 1
    fn = tot_gt - tp
    p = tp/(tp+fp) if tp+fp else 0
    r = tp/(tp+fn) if tp+fn else 0
    f1 = 2*p*r/(p+r) if p+r else 0
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': round(p,4), 'recall': round(r,4), 'f1': round(f1,4)}

item2 = {}
for conf in [0.1, 0.3, 0.5]:
    # 每密度桶
    bucket_stats = {}
    for imid, anns in gt_by_img.items():
        b = bucket_of(len(anns))
        dets = sorted([d for d in det_by_img.get(imid, []) if d['score'] >= conf], key=lambda d: -d['score'])
        used = set(); t = 0; f = 0
        for d in dets:
            bi, biou = -1, 0.0
            for gi, a in enumerate(anns):
                if gi in used: continue
                iou = iou_wh(a['bbox'], d['bbox'])
                if iou > biou: bi, biou = gi, iou
            if bi >= 0 and biou >= IOU_TH: used.add(bi); t += 1
            else: f += 1
        bs = bucket_stats.setdefault(b, {'gt': 0, 'tp': 0, 'fp': 0})
        bs['gt'] += len(anns); bs['tp'] += t; bs['fp'] += f
    item2[str(conf)] = {'overall': prec_recall_f1(conf), 'per_density': {}}
    for b, s in bucket_stats.items():
        fn = s['gt'] - s['tp']
        p = s['tp']/(s['tp']+s['fp']) if (s['tp']+s['fp']) else 0
        r = s['tp']/(s['tp']+fn) if (s['tp']+fn) else 0
        f1 = 2*p*r/(p+r) if (p+r) else 0
        item2[str(conf)]['per_density'][b] = {'gt': s['gt'], 'tp': s['tp'], 'fp': s['fp'], 'fn': fn,
                                              'recall': round(r,4), 'precision': round(p,4), 'f1': round(f1,4)}

# ---------- 3) 密度下 bbox 面积分布 + 小目标比例 ----------
item3 = {}
for imid, anns in gt_by_img.items():
    b = bucket_of(len(anns))
    s = item3.setdefault(b, {'n': 0, 'size': Counter(), 'area_all': []})
    for a in anns:
        s['n'] += 1; s['size'][size(a, imgs[imid])] += 1
        s['area_all'].append(area_ratio(a, imgs[imid]))
for b, s in item3.items():
    s['small_ratio'] = round(s['size']['small']/s['n'], 4)
    s['area_median'] = round(float(np.median(s['area_all'])), 5)
    s['size_dist'] = dict(s['size'])
    del s['area_all']

# ---------- 4) 目标间空间拥挤 (同一图 GT 两两) ----------
pair_iou_all, pair_center_all, crowded_pairs = [], [], []
pair_per_img = {}
for imid, anns in gt_by_img.items():
    W, H = imgs[imid]['width'], imgs[imid]['height']
    diag = np.hypot(W, H)
    pis = {'pairs': 0, 'n_overlap01': 0, 'n_overlap03': 0}
    for i in range(len(anns)):
        for j in range(i+1, len(anns)):
            a, b = anns[i], anns[j]
            iou = iou_wh(a['bbox'], b['bbox'])
            c1 = (a['bbox'][0]+a['bbox'][2]/2, a['bbox'][1]+a['bbox'][3]/2)
            c2 = (b['bbox'][0]+b['bbox'][2]/2, b['bbox'][1]+b['bbox'][3]/2)
            cd = np.hypot(c1[0]-c2[0], c1[1]-c2[1])/diag
            pair_iou_all.append(iou); pair_center_all.append(cd)
            pis['pairs'] += 1
            if iou > 0.1: pis['n_overlap01'] += 1
            if iou > 0.3: pis['n_overlap03'] += 1
    pair_per_img[imid] = pis
item4 = {
    'pair_iou_mean': round(float(np.mean(pair_iou_all)), 4),
    'pair_iou_median': round(float(np.median(pair_iou_all)), 4),
    'pair_iou_ge0.1_ratio': round(sum(1 for x in pair_iou_all if x > 0.1)/len(pair_iou_all), 4),
    'pair_iou_ge0.3_ratio': round(sum(1 for x in pair_iou_all if x > 0.3)/len(pair_iou_all), 4),
    'center_dist_mean': round(float(np.mean(pair_center_all)), 4),
    'center_dist_lt0.2_ratio': round(sum(1 for x in pair_center_all if x < 0.2)/len(pair_center_all), 4),
}

# ---------- 5) 同类密集 vs 异类密集 ----------
# 实测: 所有 VAL 密集图(>=5)几乎为单一病害团簇(ratio=1.00)。按主导类再分桶。
def dom_class_ratio(anns):
    c = Counter(a['category_id'] for a in anns)
    top, n = c.most_common(1)[0]
    return top, n/len(anns)
same_dense = {'img': 0, 'gt': 0, 'miss': 0}
mix_dense = {'img': 0, 'gt': 0, 'miss': 0}
per_topclass_dense = {}   # 主导类 → {img, gt, miss}
ratio_list = []
for imid, anns in gt_by_img.items():
    if len(anns) < 5: continue
    top, ratio = dom_class_ratio(anns)
    ratio_list.append(round(ratio, 3))
    m = per_img[imid]['n_miss05']
    t = per_topclass_dense.setdefault(top, {'img': 0, 'gt': 0, 'miss': 0})
    t['img'] += 1; t['gt'] += len(anns); t['miss'] += m
    if ratio >= 0.6:
        same_dense['img'] += 1; same_dense['gt'] += len(anns); same_dense['miss'] += m
    else:
        mix_dense['img'] += 1; mix_dense['gt'] += len(anns); mix_dense['miss'] += m
item5 = {
    'finding': 'all_dense_are_single_disease_cluster',
    'min_dom_ratio': min(ratio_list), 'n_dense_img': len(ratio_list),
    'same_class_dense': {'img': same_dense['img'], 'gt': same_dense['gt'], 'miss': same_dense['miss'],
                         'miss_rate': round(same_dense['miss']/same_dense['gt'], 4)} if same_dense['gt'] else None,
    'mixed_dense': {'img': mix_dense['img'], 'gt': mix_dense['gt'], 'miss': mix_dense['miss'],
                    'miss_rate': round(mix_dense['miss']/mix_dense['gt'], 4)} if mix_dense['gt'] else None,
    'per_dominant_class': {cats[c]: {'img': v['img'], 'gt': v['gt'], 'miss': v['miss'],
                                     'miss_rate': round(v['miss']/v['gt'], 4)}
                           for c, v in sorted(per_topclass_dense.items(),
                                              key=lambda kv: -(kv[1]['miss']/kv[1]['gt']))
                           if v['gt'] > 0},
}

# ---------- 6) 漏检目标置信度分布 (覆盖检测的最高分数) ----------
# 对每个 conf0.5 漏检 GT: 找 IoU>=0.5 的检测中 score 最高者
miss_score = []
miss_best_iou_any = []
per_cls_miss = defaultdict(lambda: {'gt': 0, 'miss': 0, 'score_band': Counter()})
for imid, anns in gt_by_img.items():
    dets_all = det_by_img.get(imid, [])
    for a in anns:
        cid = a['category_id']
        best_sc = 0.0
        for d in dets_all:
            if iou_wh(a['bbox'], d['bbox']) >= IOU_TH and d['score'] > best_sc:
                best_sc = d['score']
        per_cls_miss[cid]['gt'] += 1
        if best_sc < CONF_TH:
            per_cls_miss[cid]['miss'] += 1
            per_cls_miss[cid]['score_band'][band(best_sc)] += 1
            miss_score.append(best_sc)
item6 = {'miss_score_mean': round(float(np.mean(miss_score)), 4),
         'miss_score_band': {b: sum(1 for x in miss_score if band(x) == b) for b in BANDS}}

# ---------- 7) conf 各区段候选框数量及类别分布 ----------
item7 = {'candidate_total': len(preds), 'per_band': {}}
for b in BANDS:
    item7['per_band'][b] = {'count': cand_band[b],
                            'class_top': [(cats[c], n) for c, n in cand_band_class[b].most_common(5)]}

# ---------- 8) 降阈值 Recall 恢复最多的类别 ----------
# recall per class at conf 0.5 vs 0.1
def recall_per_class(conf):
    r = Counter()
    for imid, anns in gt_by_img.items():
        dets = sorted([d for d in det_by_img.get(imid, []) if d['score'] >= conf], key=lambda d: -d['score'])
        used = set()
        for d in dets:
            bi, biou, bgc = -1, 0.0, None
            for gi, a in enumerate(anns):
                if gi in used: continue
                iou = iou_wh(a['bbox'], d['bbox'])
                if iou > biou: bi, biou, bgc = gi, iou, a['category_id']
            if bi >= 0 and biou >= IOU_TH: used.add(bi); r[bgc] += 1
    return r
rec50 = recall_per_class(0.5)
rec30 = recall_per_class(0.3)
rec10 = recall_per_class(0.1)
item8 = []
for cid in range(1, 14):
    item8.append({'class': cats[cid], 'gt': gt_totals[cid],
                  'recall@0.5': round(rec50[cid]/gt_totals[cid], 4),
                  'recall@0.3': round(rec30[cid]/gt_totals[cid], 4),
                  'recall@0.1': round(rec10[cid]/gt_totals[cid], 4),
                  'recover_0.3-0.5': round((rec30[cid]-rec50[cid])/gt_totals[cid], 4),
                  'recover_0.1-0.5': round((rec10[cid]-rec50[cid])/gt_totals[cid], 4)})
item8.sort(key=lambda x: -x['recover_0.1-0.5'])

# ---------- 9) 降阈值 FP 增加 ----------
# FP 集合: 匹配后未命中的检测, 按类/所在图密度统计; 比较 conf0.5 vs conf0.1
def fp_stats(conf):
    fp_class = Counter(); fp_density = Counter()
    for imid, anns in gt_by_img.items():
        dets = sorted([d for d in det_by_img.get(imid, []) if d['score'] >= conf], key=lambda d: -d['score'])
        used = set()
        for d in dets:
            bi, biou = -1, 0.0
            for gi, a in enumerate(anns):
                if gi in used: continue
                iou = iou_wh(a['bbox'], d['bbox'])
                if iou > biou: bi, biou = gi, iou
            if bi >= 0 and biou >= IOU_TH: used.add(bi)
            else:
                fp_class[d['category_id']] += 1
                fp_density[bucket_of(len(anns))] += 1
    return fp_class, fp_density
fp50, fpden50 = fp_stats(0.5)
fp10, fpden10 = fp_stats(0.1)
item9 = {
    'fp@0.5_total': sum(fp50.values()), 'fp@0.1_total': sum(fp10.values()),
    'fp_gain_0.1-0.5': sum(fp10.values()) - sum(fp50.values()),
    'fp_gain_by_class_top': [(cats[c], n) for c, n in (fp10 - fp50).most_common(8)],
    'fp_gain_by_density': {b: fpden10[b] - fpden50[b] for b in set(fpden10) | set(fpden50)},
}

# ---------- 10) NMS 疑似误删 ----------
# 判据A: GT 漏检且存在与之 IoU>=0.4 的另一 GT (两个目标挤在同一区域, NMS 通常只保留一个)
# 判据B: GT 漏检且存在与之 IoU>=0.4 且 score>=0.5 的检测 (高置信邻居被保留, 该目标框被压)
nms_suspect = []
nms_suspect_class = Counter()
nms_overlap_gt = 0
for imid, anns in gt_by_img.items():
    for a in anns:
        cid = a['category_id']
        best50, best50_sc = 0.0, 0.0
        dets50 = [d for d in det_by_img.get(imid, []) if d['score'] >= CONF_TH]
        for d in dets50:
            iou = iou_wh(a['bbox'], d['bbox'])
            if iou > best50: best50, best50_sc = iou, d['score']
        if best50 >= IOU_TH: continue  # 未漏检
        # 判据B: 高置信重叠检测
        b_overlap = any(iou_wh(a['bbox'], d['bbox']) >= 0.4 and d['score'] >= 0.5 for d in dets50)
        # 判据A: 与另一 GT 高重叠
        a_overlap = any(iou_wh(a['bbox'], b['bbox']) >= 0.4 for b in anns if b is not a)
        if a_overlap: nms_overlap_gt += 1
        if b_overlap or a_overlap:
            nms_suspect.append(imid)
            nms_suspect_class[cid] += 1
item10 = {'nms_suspect_gt': len(nms_suspect),
          'nms_suspect_rate': round(len(nms_suspect)/tot_gt, 4),
          'gt_overlap04_both': nms_overlap_gt,
          'nms_suspect_by_class_top': [(cats[c], n) for c, n in nms_suspect_class.most_common(6)]}

# ---------- 11) 重点类密集 vs 稀疏 ----------
FOCUS = ['Tomato leaf yellow virus', 'Tomato leaf mosaic virus',
         'Tomato mold leaf', 'Tomato leaf bacterial spot', 'Tomato Septoria leaf spot']
name2id = {v: k for k, v in cats.items()}
item11 = {}
for name in FOCUS:
    cid = name2id[name]
    dense = {'gt': 0, 'miss': 0, 'recall@0.5': None, 'recall@0.1': None}
    sparse = {'gt': 0, 'miss': 0}
    dense_imgs = []
    for imid, anns in gt_by_img.items():
        for a in anns:
            if a['category_id'] != cid: continue
            dens = len(anns)
            m = 1 if per_img[imid]['match_res'][gt_by_img[imid].index(a)][0] == False else 0
            if dens >= 5:
                dense['gt'] += 1; dense['miss'] += m; dense_imgs.append(imid)
            else:
                sparse['gt'] += 1; sparse['miss'] += m
    item11[name] = {
        'dense_gt': dense['gt'], 'dense_miss': dense['miss'],
        'dense_miss_rate': round(dense['miss']/dense['gt'], 4) if dense['gt'] else None,
        'sparse_gt': sparse['gt'], 'sparse_miss': sparse['miss'],
        'sparse_miss_rate': round(sparse['miss']/sparse['gt'], 4) if sparse['gt'] else None,
        'n_dense_imgs': len(set(dense_imgs))}

# ---------- 保存 ----------
out = {
    'totals': {'gt': tot_gt, 'miss@0.5': tot_miss05, 'miss_rate': round(tot_miss05/tot_gt, 4),
               'pred_total': len(preds), 'img_total': len(imgs)},
    '1_density_vs_miss': item1,
    '2_density_bucket_prf': item2,
    '3_density_area_small': item3,
    '4_spatial_crowding': item4,
    '5_same_vs_mixed_dense': item5,
    '6_miss_conf_dist': item6,
    '7_candidate_bands': item7,
    '8_recall_recover_by_class': item8,
    '9_fp_gain': item9,
    '10_nms_suspect': item10,
    '11_focus_dense': item11,
    'per_class_miss': {str(c): per_cls_miss[c] for c in per_cls_miss},
}
json.dump(out, open(f'{OUT}/d4_diagnostic.json', 'w'), ensure_ascii=False, indent=1)

# ---------- 控制台 ----------
print('===== 方向4 密集场景诊断 (A0 best, VAL) =====')
print(f"总GT={tot_gt} 漏检@0.5={tot_miss05} ({tot_miss05/tot_gt:.1%})  候选框={len(preds)}")
print('\n[1] 密度 vs 漏检率')
for b in ['1','2','3-4','5-9','10+']:
    d = item1.get(b, {})
    print(f"  {b:>4s} 目标: GT={d.get('gt',0):3d} 漏={d.get('miss',0):3d} ({d.get('miss_rate',0):.1%})")
print('\n[2] 密度桶 P/R/F1 @conf0.5 (及 overall @0.1/0.3/0.5)')
for conf in ['0.5','0.3','0.1']:
    o = item2[conf]['overall']
    print(f"  conf={conf}: R={o['recall']:.3f} P={o['precision']:.3f} F1={o['f1']:.3f} (TP={o['tp']} FP={o['fp']})")
print('  @0.5 各桶 Recall:',
      {b: round(item2['0.5']['per_density'][b]['recall'],3) for b in item2['0.5']['per_density']})
print('\n[3] 密度下小目标比例:')
for b, s in item3.items():
    print(f"  {b:>4s}: small_ratio={s['small_ratio']:.2%} 中位面积={s['area_median']:.5f} {s['size_dist']}")
print('\n[4] 空间拥挤: pair_IoU均值={} 中位={}  IoU>0.1占比={} 中心距<0.2占比={}'.format(
    item4['pair_iou_mean'], item4['pair_iou_median'], item4['pair_iou_ge0.1_ratio'], item4['center_dist_lt0.2_ratio']))
print('\n[5] 同类 vs 异类密集:')
print(f"  同类密集: {item5['same_class_dense']}")
print(f"  异类密集: {item5['mixed_dense']}")
print('\n[6] 漏检目标覆盖分数分布:', item6)
print('\n[7] 候选框 conf 带:')
for b in BANDS:
    it = item7['per_band'][b]
    top = ', '.join(f'{n}({c})' for c, n in it['class_top'][:3])
    print(f"  {b:>8s}: {it['count']:5d}   top类: {top}")
print('\n[8] 降阈值 Recall 恢复最多 (前6):')
for x in item8[:6]:
    print(f"  {x['class']:30s} R@0.5={x['recall@0.5']:.2f} R@0.1={x['recall@0.1']:.2f} 恢复0.1-0.5={x['recover_0.1-0.5']:+.3f}")
print('\n[9] 降阈值 FP 增益:', item9['fp_gain_0.1-0.5'], '| 按密度:', item9['fp_gain_by_density'])
print('  FP 增益 top 类:', item9['fp_gain_by_class_top'][:5])
print('\n[10] NMS 疑似误删:', item10)
print('\n[11] 重点类密集 vs 稀疏:')
for name, d in item11.items():
    print(f"  {name:30s} 密集漏检率={d['dense_miss_rate']} ({d['dense_miss']}/{d['dense_gt']}, {d['n_dense_imgs']}图)  稀疏={d['sparse_miss_rate']} ({d['sparse_miss']}/{d['sparse_gt']})")
print(f'\n保存: {OUT}/d4_diagnostic.json')
