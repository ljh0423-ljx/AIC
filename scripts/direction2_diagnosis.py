#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
direction2_diagnosis.py — 方向2 组内判别增强: 基于 Arm B(best) VAL 预测的深度诊断
仅 VAL; TEST 封闭; 只读。重点: Tomato 内部细粒度混淆(定位对但类错)的每个细节。

输入:
  GT   = dataset/processed_detection_exiffix/annotations/val.json
  pred = experiments/ablation_data_exiffix/metrics/val_eval_best.json/bbox.json
输出 (experiments/direction2/):
  d2_confusion_matrix.csv       全 13×13 GT→Pred 混淆(conf=0.5,IoU=0.5)
  d2_named_pairs.json           bacterial spot↔Septoria / mosaic↔yellow / EB↔bact / mold↔EB 逐例证据
  d2_error_cases.json           误检/漏检/低置信/定位错误案例(按类分组, 含置信度/IoU)
  d2_feature_gap.json           特征表达差距量化 (空间重叠/置信分布/定位质量)
  DIRECTION2_DIAGNOSTIC.md      深度诊断报告
"""
import json, os
import numpy as np

BASE = '/root/autodl-tmp/AIC2026_AgriVision'
GT_PATH = f'{BASE}/dataset/processed_detection_exiffix/annotations/val.json'
PRED_PATH = f'{BASE}/experiments/ablation_data_exiffix/metrics/val_eval_best.json/bbox.json'
OUT = f'{BASE}/experiments/direction2'
CONF_TH = 0.5
IOU_TH = 0.5

os.makedirs(OUT, exist_ok=True)

# ---------- 加载 ----------
coco = json.load(open(GT_PATH))
cats = {c['id']: c['name'] for c in coco['categories']}
name2id = {v: k for k, v in cats.items()}
imgs = {im['id']: im for im in coco['images']}
gt_by_img = {}
for a in coco['annotations']:
    gt_by_img.setdefault(a['image_id'], []).append(a)
preds = json.load(open(PRED_PATH))

TOMATO_IDS = [i for i, n in cats.items() if n.startswith('Tomato')]
NAMED_PAIRS = [  # (gt_id, pred_id, 标签名)
    (name2id['Tomato leaf bacterial spot'], name2id['Tomato Septoria leaf spot'], 'bact->Septoria'),
    (name2id['Tomato Septoria leaf spot'], name2id['Tomato leaf bacterial spot'], 'Septoria->bact'),
    (name2id['Tomato leaf mosaic virus'], name2id['Tomato leaf yellow virus'], 'mosaic->yellow'),
    (name2id['Tomato leaf yellow virus'], name2id['Tomato leaf mosaic virus'], 'yellow->mosaic'),
    (name2id['Tomato Early blight leaf'], name2id['Tomato leaf bacterial spot'], 'EB->bact'),
    (name2id['Tomato leaf bacterial spot'], name2id['Tomato Early blight leaf'], 'bact->EB'),
    (name2id['Tomato mold leaf'], name2id['Tomato Early blight leaf'], 'mold->EB'),
    (name2id['Tomato Early blight leaf'], name2id['Tomato mold leaf'], 'EB->mold'),
    (name2id['Tomato leaf'], name2id['Tomato leaf late blight'], 'leaf->lateblight'),
    (name2id['Tomato leaf late blight'], name2id['Tomato leaf'], 'lateblight->leaf'),
]

def iou_wh(bb1, bb2):
    x1, y1, w1, h1 = bb1; x2, y2, w2, h2 = bb2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    iw, ih = max(0, xi2-xi1), max(0, yi2-yi1)
    inter = iw*ih
    return inter / (w1*h1 + w2*h2 - inter + 1e-9)

def center(bb):
    return (bb[0]+bb[2]/2, bb[1]+bb[3]/2)

# ---------- 1. 匹配 (conf=0.5, IoU=0.5, greedy 分数降序) ----------
det_by_img = {}
for p in preds:
    det_by_img.setdefault(p['image_id'], []).append(p)

per_img = {}   # image_id -> {'gt':[...], 'match':[...]}
conf_mat = np.zeros((13, 13), dtype=int)   # [gt][pred]
img_err = {}
for imid, gt_anns in gt_by_img.items():
    dets = sorted(det_by_img.get(imid, []), key=lambda d: -d['score'])
    dets = [d for d in dets if d['score'] >= CONF_TH]
    matched = set()
    rec = {'tp': 0, 'fp': 0, 'fn': len(gt_anns), 'loc_ok': 0, 'cls_ok': 0,
           'gt': [], 'det': []}
    for d in dets:
        dc = d['category_id']
        bi, biou, bgc = -1, 0.0, None
        for gi, a in enumerate(gt_anns):
            if gi in matched: continue
            iou = iou_wh(a['bbox'], d['bbox'])
            if iou > biou: bi, biou, bgc = gi, iou, a['category_id']
        e = {'gt': bgc, 'pred': dc, 'iou': round(biou, 3), 'score': round(d['score'], 3),
             'box': [round(v,1) for v in d['bbox']], 'file': imgs[imid]['file_name']}
        if bi >= 0 and biou >= IOU_TH:
            matched.add(bi)
            rec['tp'] += 1; rec['fn'] -= 1
            rec['loc_ok'] += 1
            if dc == bgc: rec['cls_ok'] += 1
            conf_mat[bgc-1][dc-1] += 1
            rec['det'].append({**e, 'match': 'TP'})
        else:
            rec['fp'] += 1
            conf_mat[0][dc-1] += 0  # 背景 FP 不进矩阵, 单独计数
            rec['det'].append({**e, 'match': 'FP'})
    for gi, a in enumerate(gt_anns):
        if gi not in matched:
            rec['gt'].append({'id': a['category_id'], 'box': [round(v,1) for v in a['bbox']]})
    img_err[imid] = rec

# 汇总
tot = {'tp': sum(v['tp'] for v in img_err.values()),
       'fp': sum(v['fp'] for v in img_err.values()),
       'fn': sum(v['fn'] for v in img_err.values()),
       'loc_ok': sum(v['loc_ok'] for v in img_err.values()),
       'cls_ok': sum(v['cls_ok'] for v in img_err.values())}
tot['loc_but_wrong_cls'] = tot['loc_ok'] - tot['cls_ok']

# ---------- 2. 混淆矩阵 CSV ----------
with open(f'{OUT}/d2_confusion_matrix.csv', 'w') as f:
    f.write('gt\\\\pred,' + ','.join(cats[i] for i in range(1, 14)) + '\n')
    for g in range(1, 14):
        f.write(cats[g] + ',' + ','.join(str(conf_mat[g-1][p-1]) for p in range(1, 14)) + '\n')

# ---------- 3. 命名混淆对逐例证据 ----------
named = {}
for gt_id, pred_id, label in NAMED_PAIRS:
    cases = []
    for imid, rec in img_err.items():
        for e in rec['det']:
            if e['match'] == 'TP' and e['gt'] == gt_id and e['pred'] == pred_id:
                cases.append({'file': e['file'], 'iou': e['iou'], 'score': e['score'], 'box': e['box']})
    named[label] = {'gt_id': gt_id, 'pred_id': pred_id, 'count': len(cases),
                    'gt_total': sum(1 for a in coco['annotations'] if a['category_id'] == gt_id),
                    'pct_gt': round(len(cases) / sum(1 for a in coco['annotations'] if a['category_id'] == gt_id), 3),
                    'cases': cases}

# 组内 vs 跨作物混淆
within_tomato = np.sum(conf_mat[np.ix_([c-1 for c in TOMATO_IDS], [c-1 for c in TOMATO_IDS])]) - sum(np.diag(conf_mat)[c-1] for c in TOMATO_IDS)
diag_tomato = sum(np.diag(conf_mat)[c-1] for c in TOMATO_IDS)
cross_crop = 0
tomato_set = set(TOMATO_IDS)
for g in range(1, 14):
    for p in range(1, 14):
        if g == p: continue
        if (g in tomato_set) != (p in tomato_set):
            cross_crop += conf_mat[g-1][p-1]

# ---------- 4. 误检/漏检/低置信/定位错误案例 ----------
weak_classes = ['Tomato leaf mosaic virus', 'Tomato leaf yellow virus', 'Tomato mold leaf',
                'Tomato leaf bacterial spot', 'Tomato Early blight leaf']
case_info = {'missed': {}, 'low_conf_correct': {}, 'weak_loc': {}, 'high_conf_wrong': {}}

for cname in weak_classes + ['Tomato leaf', 'Tomato Septoria leaf spot', 'Tomato leaf late blight']:
    cid = name2id[cname]
    # 漏检: 该 GT 完全无框覆盖
    miss_imgs = []
    for imid, rec in img_err.items():
        for g in rec['gt']:
            if g['id'] == cid:
                miss_imgs.append(imgs[imid]['file_name'])
    case_info['missed'][cname] = {'n_fn': len(miss_imgs), 'imgs': miss_imgs}

# 低置信正确检测: TP 且 score<0.6 (阈值操作点边缘)
low_conf = []
for imid, rec in img_err.items():
    for e in rec['det']:
        if e['match'] == 'TP' and e['pred'] == e['gt'] and e['score'] < 0.6:
            low_conf.append({'file': e['file'], 'gt': cats[e['gt']], 'iou': e['iou'], 'score': e['score']})
case_info['low_conf_correct'] = {'n': len(low_conf), 'cases': low_conf}

# 高置信但类错: TP 定位但类错, score>=0.7 (分类器"自信地错")
high_conf_wrong = []
for imid, rec in img_err.items():
    for e in rec['det']:
        if e['match'] == 'TP' and e['gt'] != e['pred'] and e['score'] >= 0.7:
            high_conf_wrong.append({'file': e['file'], 'gt': cats[e['gt']], 'pred': cats[e['pred']],
                                    'iou': e['iou'], 'score': e['score']})
case_info['high_conf_wrong'] = {'n': len(high_conf_wrong), 'cases': high_conf_wrong}

# 弱定位: 类对但 IoU 0.5-0.75
weak_loc = []
for imid, rec in img_err.items():
    for e in rec['det']:
        if e['match'] == 'TP' and e['gt'] == e['pred'] and 0.5 <= e['iou'] < 0.75:
            weak_loc.append({'file': e['file'], 'gt': cats[e['gt']], 'iou': e['iou'], 'score': e['score']})
case_info['weak_loc'] = {'n': len(weak_loc), 'cases': weak_loc}

# ---------- 5. 特征表达差距量化 ----------
gap = {}
# 5.1 命名混淆对的空间重叠: 两类的 GT box 是否出现在同一图/相邻区域
for label, info in named.items():
    pass  # 在 5.3 处理
# 5.2 各混淆对的类间特征可分性代理: TP 正确 score 分布 vs 混淆(类错) score 分布
def scores_of(pred_id, correct):
    s = []
    for imid, rec in img_err.items():
        for e in rec['det']:
            if e['match'] == 'TP' and e['pred'] == pred_id and (e['pred'] == e['gt']) == correct:
                s.append(e['score'])
    return s

gap['score_overlap'] = {}
for gt_id, pred_id, label in NAMED_PAIRS:
    corr = scores_of(gt_id, True)
    wrong = scores_of(pred_id, False)  # pred_id 被用于错误标记
    # wrong here: detections of class pred_id that were matched to GT pred_id (correct) — 需按正确类组织
    s_correct = scores_of(pred_id, True)
    s_wrong = []
    for imid, rec in img_err.items():
        for e in rec['det']:
            if e['match'] == 'TP' and e['pred'] == pred_id and e['gt'] != pred_id:
                s_wrong.append(e['score'])
    if s_correct and s_wrong:
        gap['score_overlap'][label] = {
            'correct_mean': round(np.mean(s_correct), 3), 'correct_n': len(s_correct),
            'wrong_mean': round(np.mean(s_wrong), 3), 'wrong_n': len(s_wrong),
            'overlap': round(min(1.0, max(0.0, (min(max(s_correct), max(s_wrong)) - max(min(s_correct), min(s_wrong))) /
                                          (max(max(s_correct), max(s_wrong)) - min(min(s_correct), min(s_wrong)) + 1e-9))), 3)}

# 5.3 空间共现: 命名对两类的 GT 出现在同一图的比例 + 类间 box 重叠
def cls_imgs(cid):
    s = set()
    for a in coco['annotations']:
        if a['category_id'] == cid:
            s.add(a['image_id'])
    return s

gap['cooccurrence'] = {}
for gt_id, pred_id, label in NAMED_PAIRS[:6]:
    ig, ip = cls_imgs(gt_id), cls_imgs(pred_id)
    inter = len(ig & ip)
    gap['cooccurrence'][label] = {
        'img_gt': len(ig), 'img_pred': len(ip), 'same_img': inter,
        'cond_same_given_gt': round(inter / len(ig), 3) if ig else 0}

# 5.4 定位质量: 类对 TP 的 IoU 桶
iou_buckets = {k: 0 for k in ['0.50-0.60', '0.60-0.75', '0.75-0.90', '0.90-1.00']}
for imid, rec in img_err.items():
    for e in rec['det']:
        if e['match'] == 'TP' and e['gt'] == e['pred']:
            i = e['iou']
            if i < 0.6: iou_buckets['0.50-0.60'] += 1
            elif i < 0.75: iou_buckets['0.60-0.75'] += 1
            elif i < 0.9: iou_buckets['0.75-0.90'] += 1
            else: iou_buckets['0.90-1.00'] += 1
gap['iou_buckets'] = iou_buckets
gap['tot'] = tot

# ---------- 6. 保存 JSON ----------
json.dump({'named_pairs': named, 'conf_mat': conf_mat.tolist(), 'case_info': case_info,
           'gap': gap, 'within_tomato': int(within_tomato), 'diag_tomato': int(diag_tomato),
           'cross_crop': int(cross_crop), 'tomato_ids': TOMATO_IDS,
           'img_err': img_err},
          open(f'{OUT}/d2_error_cases.json', 'w'), ensure_ascii=False, indent=1)

# ---------- 7. 控制台摘要 ----------
print('==== 方向2 深度诊断摘要 (Arm B best, VAL) ====')
print(f"总体: TP={tot['tp']} FP={tot['fp']} FN={tot['fn']} loc_ok={tot['loc_ok']} cls_ok={tot['cls_ok']} 定位对但类错={tot['loc_but_wrong_cls']}")
print(f"组内(番茄)混淆={within_tomato} 番茄正确同检(diag)={diag_tomato} 跨作物混淆={cross_crop}")
print('--- 命名混淆对 ---')
for label, info in named.items():
    if info['count']:
        print(f"  {label}: {info['count']}/{info['gt_total']} ({info['pct_gt']})")
print('--- 高置信类错 ---')
print(f"  {case_info['high_conf_wrong']['n']} 例 (score>=0.7 定位对但类错)")
for c in case_info['high_conf_wrong']['cases'][:8]:
    print(f"    {c['file']}  GT={c['gt']}  pred={c['pred']}  iou={c['iou']} score={c['score']}")
print('--- 低置信正确检测 ---')
print(f"  {case_info['low_conf_correct']['n']} 例 (score<0.6 的 TP)")
print('--- 弱定位 (类对, IoU 0.5-0.75) ---')
print(f"  {case_info['weak_loc']['n']} 例")
print('--- 各弱类漏检数 ---')
for cname in weak_classes:
    print(f"  {cname}: {case_info['missed'][cname]['n_fn']}")
print('--- 混淆对置信分布 (正确score均值 vs 类错score均值) ---')
for label, v in gap['score_overlap'].items():
    print(f"  {label}: 对={v['correct_mean']}(n={v['correct_n']}) 错={v['wrong_mean']}(n={v['wrong_n']}) 重叠={v['overlap']}")
print('--- 混淆对空间共现 (同一图) ---')
for label, v in gap['cooccurrence'].items():
    print(f"  {label}: GT图={v['img_gt']} Pred图={v['img_pred']} 同图={v['same_img']} (given GT={v['cond_same_given_gt']})")
print('--- 类对 TP 的 IoU 桶 ---')
for k, v in iou_buckets.items():
    print(f"  {k}: {v}")
print(f'完成: {OUT}/')
