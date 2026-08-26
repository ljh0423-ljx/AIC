#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_class_weight.py — B1 class-balanced 固定权重生成 (仅 TRAIN 统计)
方案: 归一化逆频率  w_c = (1/N_c) / mean_{c'}(1/N_{c'})
  - 13 类权重均值 = 1.0, Σw = 13 → 全局 loss 尺度不变, 只改变类间相对强调
  - 背景类 w_bg = 1.0
  - 只读取 train.json; 严禁使用 VAL/TEST 计算权重。输出冻结 JSON。
用法: python scripts/gen_class_weight.py
"""
import json
from collections import Counter

BASE = '/root/autodl-tmp/AIC2026_AgriVision'
TR = f'{BASE}/dataset/processed_detection_exiffix/annotations/train.json'
OUT = f'{BASE}/experiments/direction3/class_weight_train.json'

tr = json.load(open(TR))
tc = Counter(a['category_id'] for a in tr['annotations'])
cats = {c['id']: c['name'] for c in tr['categories']}
total = sum(tc.values())
C = 13
inv = [1.0 / tc[c] for c in range(1, C + 1)]
mean_inv = sum(inv) / C
w = {str(c): round((1.0 / tc[c]) / mean_inv, 6) for c in range(1, C + 1)}
w['bg'] = 1.0
mean_w = sum(w[str(c)] for c in range(1, C + 1)) / C

out = {
    'meta': {
        'source': 'TRAIN only', 'total_instances': total, 'num_classes': C,
        'scheme': 'normalized inverse frequency: w_c = (1/N_c)/mean_{c}(1/N_c), w_bg=1',
        'mean_weight': round(mean_w, 6), 'sum_weight': round(mean_w * C, 4),
        'generated': '2026-08-15', 'frozen': True,
        'warning': '仅训练时使用; 推理不改变; 若 VAL/TEST 用于权重计算即为作弊',
    },
    'category_name': {str(c): cats[c] for c in range(1, C + 1)},
    'class_count': {str(c): tc[c] for c in range(1, C + 1)},
    'weight': w,
}
json.dump(out, open(OUT, 'w'), ensure_ascii=False, indent=1)
print(f'Σw={out["meta"]["sum_weight"]}  均值={out["meta"]["mean_weight"]}')
for c in range(1, C + 1):
    print(f'{c:2d} {cats[c]:30s} N={tc[c]:4d}  w={w[str(c)]:.3f}')
print(f'已保存 {OUT}')
