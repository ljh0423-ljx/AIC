#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
d2_operator.py — D2 推理侧后处理算子(簇投票重打分 + 保守重标)
按 DIRECTION4_DESIGN.md §2.2 定义实现;不修改模型/权重/数据集。
TRAIN 锁定参数后冻结;VAL 仅做最终评估。
"""
import numpy as np
from collections import defaultdict

BAND_LO, BAND_HI = 0.3, 0.5   # 校准带(诊断: 184 漏检的覆盖全部落在 0.05-0.49, 主体 0.3-0.5)
SCORE_FLOOR = 0.1             # 主导类/邻居统计的"证据底线"(排除 <0.1 海量噪声池)


def apply_operator(preds, img_meta, alpha=0.6, straggler=0.3,
                   use_rescore=False, tau=0.2, lam=0.1, min_agg=0.8):
    """
    preds:      list[dict{image_id, category_id, bbox, score}]
    img_meta:   {image_id: (width, height)}
    返回: (new_preds, stats)
      stats: {relabeled: int, rescored: int, images_with_dom: int}
    不修改输入 preds(输出全新 dict 列表)。
    """
    # 按图分组
    by_img = defaultdict(list)
    for p in preds:
        by_img[p['image_id']].append(p)
    new_preds = []
    stats = {'relabeled': 0, 'rescored': 0, 'images_with_dom': 0}

    for imid, dets in by_img.items():
        W, H = img_meta[imid]
        diag = np.hypot(W, H)
        # --- 图像级检测主导类(score>=0.1, 按分数加权) ---
        agg = defaultdict(float)          # class -> 总分(score>=0.1)
        for d in dets:
            if d['score'] >= SCORE_FLOOR:
                agg[d['category_id']] += d['score']
        total = sum(agg.values())
        domc, domw = None, 0.0
        if total > 0:
            domc, domw = max(agg.items(), key=lambda kv: kv[1])
            if domw / total < alpha:      # 主导不显著 → 该图不重标
                domc = None
        if domc is not None:
            stats['images_with_dom'] += 1

        # 重标 + 重打分(仅在 0.3-0.5 带内)
        out = []
        for d in dets:
            nd = dict(d)
            sc = nd['score']
            if BAND_LO <= sc < BAND_HI:
                # --- 保守重标 ---
                if domc is not None and nd['category_id'] != domc:
                    origw = agg.get(nd['category_id'], 0.0)
                    if origw < straggler * domw:   # 原类是"少数游离"
                        nd['category_id'] = domc
                        stats['relabeled'] += 1
                # --- 簇投票重打分(仅当其通过 TRAIN 验证才启用) ---
                if use_rescore:
                    cx = nd['bbox'][0] + nd['bbox'][2] / 2
                    cy = nd['bbox'][1] + nd['bbox'][3] / 2
                    agg_nb = 0.0
                    for e in dets:
                        if e is d or e['score'] < SCORE_FLOOR:
                            continue
                        if e['category_id'] != nd['category_id']:
                            continue
                        ex = e['bbox'][0] + e['bbox'][2] / 2
                        ey = e['bbox'][1] + e['bbox'][3] / 2
                        if np.hypot(ex - cx, ey - cy) < tau * diag:
                            agg_nb += e['score']
                    if agg_nb >= min_agg:
                        nd['score'] = min(sc + lam, 0.55)
                        stats['rescored'] += 1
            out.append(nd)
        new_preds.extend(out)

    return new_preds, stats


def load_json(path):
    import json
    return json.load(open(path))
