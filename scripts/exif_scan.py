#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXIF Orientation 扫描 — 找出所有 Orientation != 1 的图片
只读, 不修改任何数据。
输出: metrics/exif_anomaly_report.json
"""
import json, os
from PIL import Image, ExifTags
import cv2

BASE = '/root/autodl-tmp/AIC2026_AgriVision'
DATA = f'{BASE}/dataset/processed_detection_clean'

# Orientation tag id
ORIENT = 274

def get_orientation(path):
    try:
        im = Image.open(path)
        exif = im._getexif()
        if exif and ORIENT in exif:
            return int(exif[ORIENT])
    except Exception as e:
        return None
    return 1

# 加载 COCO 标注尺寸
coco_meta = {}
for split in ['train', 'val', 'test']:
    d = json.load(open(f'{DATA}/annotations/{split}.json'))
    for im in d['images']:
        coco_meta[im['id']] = {'split': split, 'file': im['file_name'],
                               'w': im['width'], 'h': im['height']}
    # bbox 数 per image
for split in ['train', 'val', 'test']:
    d = json.load(open(f'{DATA}/annotations/{split}.json'))
    anns = {}
    for a in d['annotations']:
        anns.setdefault(a['image_id'], []).append(a['category_id'])
    for im in d['images']:
        coco_meta[im['id']]['n_bbox'] = len(anns.get(im['id'], []))
        coco_meta[im['id']]['cats'] = sorted(set(anns.get(im['id'], [])))

cat_names = {c['id']: c['name'] for c in json.load(open(f'{DATA}/annotations/train.json'))['categories']}

anomalies = []
total = 0
for split in ['train', 'val', 'test']:
    for fname in sorted(os.listdir(f'{DATA}/images/{split}')):
        path = f'{DATA}/images/{split}/{fname}'
        total += 1
        ori = get_orientation(path)
        if ori is None:
            anomalies.append({'split': split, 'file': fname, 'orientation': 'NO_EXIF_OR_READ_ERR'})
            continue
        if ori == 1:
            continue
        im = cv2.imread(path)
        h, w = im.shape[:2]
        # 找 COCO 记录
        rec = next((v for v in coco_meta.values() if v['file'] == fname), None)
        if rec is None:
            rec = {'split': split, 'w': None, 'h': None, 'n_bbox': None, 'cats': []}
        anomalies.append({
            'split': rec['split'], 'file': fname,
            'cv2_px_hw': [h, w], 'anno_hw': [rec['h'], rec['w']],
            'size_swapped': (h == rec['w'] and w == rec['h']),
            'orientation': ori,
            'n_bbox': rec['n_bbox'],
            'cats': [cat_names[c] for c in rec['cats']],
        })

print(f"扫描图片总数: {total}")
print(f"EXIF Orientation != 1 的图片: {len(anomalies)}")
print("=" * 100)
for i, a in enumerate(anomalies, 1):
    swapped = '是' if a.get('size_swapped') else '否'
    print(f"[{i:2d}] {a['split']:5s} {a['file'][:55]:55s} "
          f"px={a.get('cv2_px_hw')} 标注={a.get('anno_hw')} 互换={swapped} "
          f"Orientation={a.get('orientation')} bbox={a.get('n_bbox')} "
          f"类={','.join(a.get('cats', []))[:40]}")

json.dump(anomalies, open(f'{BASE}/experiments/baseline_v1/metrics/exif_anomaly_report.json', 'w'),
          ensure_ascii=False, indent=1)
print("\n报告已存: metrics/exif_anomaly_report.json")
