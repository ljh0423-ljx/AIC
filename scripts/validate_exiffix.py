#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_exiffix.py — EXIF 修复副本自动校验 (设计: EXIF_FIX_PLAN.md §4)
对 processed_detection_exiffix 执行 7 项校验, 全部 PASS 才允许训练 Arm B。
任一 FAIL -> 立即停止 (exit 1), 不得强行修复。
只读, 不修改任何数据。
输出: experiments/baseline_v1/metrics/VALIDATION_REPORT.txt
"""
import os, sys, json, hashlib, glob
import numpy as np
import cv2
from PIL import Image

BASE = '/root/autodl-tmp/AIC2026_AgriVision'
CLEAN = f'{BASE}/dataset/processed_detection_clean'
COPY  = f'{BASE}/dataset/processed_detection_exiffix'
OUT   = f'{BASE}/experiments/baseline_v1/metrics'
ORIENT = 274

results = []  # (check, status, detail)
def record(check, ok, detail):
    results.append((check, 'PASS' if ok else 'FAIL', detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {check}: {detail}")

# 待烘焙清单
TARGETS = [
    ('train', 'TEST_000185_20130610_110514.jpg', 6),
    ('train', 'TEST_000266_Shoemaker_7068.JPG.jpg', 6),
    ('train', 'TRAIN_000172_20130802_111648.jpg', 6),
    ('train', 'TRAIN_000198_20130610_110525.jpg', 6),
    ('train', 'TRAIN_000225_early-blight-in-high-tunnel-tomatoes-19tm74h.jpg', 6),
    ('train', 'TRAIN_000325_tomato-septoria-3.jpg', 6),
    ('train', 'TRAIN_000329_tomato-septoria-5.jpg', 8),
    ('train', 'TRAIN_000437_flies.jpg', 6),
    ('train', 'TRAIN_000494_IMG_2348.jpg', 6),
    ('val',  'TRAIN_000054_happier-inside.jpg', 8),
]
BAKED = {(s, f) for s, f, _ in TARGETS}

def load_coco(split):
    d = json.load(open(f'{COPY}/annotations/{split}.json'))
    im_by_file = {im['file_name']: im for im in d['images']}
    anns = d['annotations']
    return im_by_file, anns

def get_orientation(path):
    try:
        im = Image.open(path)
        exif = im._getexif()
        return int(exif[ORIENT]) if (exif and ORIENT in exif) else 1
    except Exception:
        return None

def img_files(split):
    return sorted(glob.glob(f'{COPY}/images/{split}/*'))

# ---------------- 加载各分片 ----------------
coco = {s: load_coco(s) for s in ['train', 'val', 'test']}
# 期待的目标数
EXP_OBJ = {'train': 3079, 'val': 350, 'test': 423}
# 期待(clean)EXIF 错位图 = 11 张
MISMATCH_EXPECTED_CLEAN = {(s, f) for s, f, _ in TARGETS} | {('test', 'TRAIN_000064_photo-1.jpg')}

# =============== 检查 1: 图像-标注尺寸一致 ===============
print("--- 检查1: 图像-标注尺寸一致 ---")
clean_mismatch, copy_mismatch = [], []
for split in ['train', 'val', 'test']:
    im_by_file, _ = coco[split]
    for p in img_files(split):
        fn = os.path.basename(p)
        jw, jh = im_by_file[fn]['width'], im_by_file[fn]['height']
        # clean 侧
        im_c = cv2.imread(f'{CLEAN}/images/{split}/{fn}')
        hc, wc = im_c.shape[:2] if im_c is not None else (None, None)
        if (wc, hc) != (jw, jh):
            clean_mismatch.append((split, fn))
        # 副本侧
        im = cv2.imread(p)
        h, w = im.shape[:2] if im is not None else (None, None)
        if (w, h) != (jw, jh):
            copy_mismatch.append((split, fn))
# clean 应恰好 11 张错位 (含 test 1 张)
if set(clean_mismatch) != MISMATCH_EXPECTED_CLEAN:
    record('1a. clean 错位清单', False, f'期望 {len(MISMATCH_EXPECTED_CLEAN)} 张, 实际 {len(set(clean_mismatch))} 张: {sorted(set(clean_mismatch))[:15]}')
else:
    record('1a. clean 错位清单 (11 张基线)', True, '11 张 EXIF 错位图全部定位 (train 9 / val 1 / test 1)')
# 副本: train+val 应 0 错位; test 应恰好 1 (TRAIN_000064, 未烘焙)
copy_tv = [(s, f) for s, f in copy_mismatch if s != 'test']
copy_te = [(s, f) for s, f in copy_mismatch if s == 'test']
if copy_tv:
    record('1b. 副本 train+val 尺寸一致', False, f'{len(copy_tv)} 张仍错位: {copy_tv[:10]}')
else:
    record('1b. 副本 train+val 尺寸一致', True, 'train+val 全部 cv2 尺寸 == JSON (0 错位)')
if set(copy_te) == {('test', 'TRAIN_000064_photo-1.jpg')}:
    record('1c. 副本 test 尺寸 (预期 1 张错位)', True, '仅 TRAIN_000064 保持原始错位 (test 封闭, 未烘焙)')
else:
    record('1c. 副本 test 尺寸 (预期 1 张错位)', False, f'异常错位: {copy_te}')

# =============== 检查 2: bbox 合法性 ===============
print("--- 检查2: bbox 合法性 (x2>x1, y2>y1, 界内) ---")
bad_boxes = []
for split in ['train', 'val', 'test']:
    im_by_file, anns = coco[split]
    im_by_id = {im['id']: im for im in im_by_file.values()}
    for a in anns:
        x, y, w, h = a['bbox']
        if w <= 0 or h <= 0:
            bad_boxes.append((split, a['id'], '非正宽高'))
            continue
        im = im_by_id.get(a['image_id'])
        if im is None:
            bad_boxes.append((split, a['id'], 'image_id 缺失'))
            continue
        W, H = im['width'], im['height']
        if x < 0 or y < 0 or x + w > W + 1e-6 or y + h > H + 1e-6:
            bad_boxes.append((split, a['id'], f'({x},{y},{w},{h}) vs {W}x{H}'))
if bad_boxes:
    record('2. bbox 合法性', False, f'{len(bad_boxes)} 个非法: {bad_boxes[:5]}')
else:
    record('2. bbox 合法性', True, '全部 bbox 正宽高且位于图像内 (0 非法)')

# =============== 检查 3: EXIF Orientation 清零 ===============
print("--- 检查3: EXIF Orientation 处理 ---")
orient_issues = []
for split in ['train', 'val', 'test']:
    for p in img_files(split):
        fn = os.path.basename(p)
        o = get_orientation(p)
        is_baked = (split, fn) in BAKED
        if is_baked:
            if o not in (None, 1):
                orient_issues.append((split, fn, f'baked 后仍 O={o}'))
        else:
            # 未烘焙图 (含 O=0 的 2 张标签异常图、test TRAIN_000064) 应与 clean 完全一致
            o_clean = get_orientation(f'{CLEAN}/images/{split}/{fn}')
            if o != o_clean:
                orient_issues.append((split, fn, f'O={o} != clean O={o_clean}'))
if orient_issues:
    record('3. EXIF Orientation', False, f'{orient_issues[:5]}')
else:
    o64 = get_orientation(f'{COPY}/images/test/TRAIN_000064_photo-1.jpg')
    record('3. EXIF Orientation', True, f'10 张烘焙后 Orientation 已清除; 其余图(含 O=0 两图、TRAIN_000064 O={o64})与 clean 一致')

# =============== 检查 4: 烘焙正确性 (逆变换 vs clean 原始) ===============
print("--- 检查4: 烘焙正确性 (逆旋转 vs clean, 平均绝对差<3) ---")
INV = {6: cv2.ROTATE_90_COUNTERCLOCKWISE, 8: cv2.ROTATE_90_CLOCKWISE}
max_mae = 0.0
for split, fname, ori in TARGETS:
    baked = cv2.imread(f'{COPY}/images/{split}/{fname}')
    orig  = cv2.imread(f'{CLEAN}/images/{split}/{fname}')
    back  = cv2.rotate(baked, INV[ori])
    mae = float(np.mean(np.abs(back.astype(np.int16) - orig.astype(np.int16))))
    max_mae = max(max_mae, mae)
if max_mae < 3.0:
    record('4. 烘焙正确性', True, f'10 张逆变换最大 MAE={max_mae:.3f} (<3, 仅 JPEG 重压缩噪声)')
else:
    record('4. 烘焙正确性', False, f'最大 MAE={max_mae:.3f} >= 3, 内容可能错位')

# =============== 检查 5: YOLO↔COCO 交叉比对 ===============
print("--- 检查5: YOLO↔COCO bbox 交叉比对 (≤1px) ---")
mismatch_cnt = 0
for split in ['train', 'val', 'test']:
    im_by_file, anns = coco[split]
    ann_by_img = {}
    for a in anns:
        ann_by_img.setdefault(a['image_id'], []).append(a)
    img_id_of = {im['file_name']: im['id'] for im in im_by_file.values()}
    for fn, im in im_by_file.items():
        txt = f'{COPY}/labels/{split}/{os.path.splitext(fn)[0]}.txt'
        if not os.path.isfile(txt):
            mismatch_cnt += 1; continue
        W, H = im['width'], im['height']
        yolo = []
        for line in open(txt):
            parts = line.split()
            if len(parts) < 5: continue
            cx, cy, bw, bh = map(float, parts[1:5])
            yolo.append((int(float(parts[0])) + 1,  # YOLO 0-based -> COCO 1-based
                         round((cx - bw/2) * W, 2), round((cy - bh/2) * H, 2),
                         round(bw * W, 2), round(bh * H, 2)))
        gt = sorted((a['category_id'], *a['bbox']) for a in ann_by_img.get(img_id_of[fn], []))
        yolo_s = sorted(yolo)
        if len(yolo_s) != len(gt):
            mismatch_cnt += 1
            continue
        for y, g in zip(yolo_s, gt):
            if any(abs(y[i+1] - g[i+1]) > 1.0 for i in range(4)):
                mismatch_cnt += 1
if mismatch_cnt == 0:
    record('5. YOLO↔COCO 交叉比对', True, '全部分片 txt→bbox 与 COCO 逐条一致 (≤1px), 0 差异')
else:
    record('5. YOLO↔COCO 交叉比对', False, f'{mismatch_cnt} 处差异')

# =============== 检查 6: 数量守恒 + JSON 一致性 ===============
print("--- 检查6: 数量守恒 + JSON 逐字节一致 ---")
ok6 = True
for split in ['train', 'val', 'test']:
    n_im = len(img_files(split))
    n_txt = len(glob.glob(f'{COPY}/labels/{split}/*.txt'))
    n_obj = len(coco[split][1])
    exp_im = {'train': 915, 'val': 113, 'test': 114}[split]
    if n_im != exp_im or n_txt != n_im or n_obj != EXP_OBJ[split]:
        ok6 = False
        record('6. 数量守恒', False, f'{split}: im={n_im} txt={n_txt} obj={n_obj} (期望 im={exp_im} obj={EXP_OBJ[split]})')
# JSON 逐字节一致 (clean vs 副本)
json_md5_ok = True
for split in ['train', 'val', 'test']:
    a = hashlib.md5(open(f'{CLEAN}/annotations/{split}.json', 'rb').read()).hexdigest()
    b = hashlib.md5(open(f'{COPY}/annotations/{split}.json', 'rb').read()).hexdigest()
    if a != b:
        json_md5_ok = False
        record('6. JSON 一致性', False, f'{split}.json MD5 不一致')
if ok6 and json_md5_ok:
    record('6. 数量守恒 + JSON 一致', True, '915/113/114 图=txt, 3079/350/423 对象; train/val/test JSON 与 clean 逐字节一致')

# =============== 检查 7: clean 未被触碰 ===============
print("--- 检查7: clean 未被触碰 (10 张仍为原始错位) ---")
clean_untouched = True
for split, fname, ori in TARGETS:
    im = cv2.imread(f'{CLEAN}/images/{split}/{fname}')
    jw, jh = coco[split][0][fname]['width'], coco[split][0][fname]['height']
    # clean 应为原始未烘焙状态: 原始像素 = 与 JSON 交换后的尺寸 (错位仍在)
    if im is None or (im.shape[1], im.shape[0]) != (jh, jw):
        clean_untouched = False
        record('7. clean 未被触碰', False, f'{split}/{fname} 原始错位消失 (clean 可能被改动)')
if clean_untouched:
    record('7. clean 未被触碰', True, 'clean 的 10 张 EXIF 图仍为原始错位尺寸 (内容未改, 烘焙仅作用于副本)')

# ---------------- 汇总 ----------------
print("\n" + "=" * 70)
all_pass = all(s == 'PASS' for _, s, _ in results)
os.makedirs(OUT, exist_ok=True)
with open(f'{OUT}/VALIDATION_REPORT.txt', 'w') as f:
    f.write('EXIF 修复副本校验报告 — validate_exiffix.py\n')
    f.write('=' * 60 + '\n')
    for check, status, detail in results:
        f.write(f"[{status}] {check}: {detail}\n")
    f.write('=' * 60 + '\n')
    f.write(f"总结果: {'ALL PASS' if all_pass else 'FAILED'}\n")
print(f"总结果: {'ALL PASS' if all_pass else 'FAILED'}")
print(f"报告: {OUT}/VALIDATION_REPORT.txt")
sys.exit(0 if all_pass else 1)
