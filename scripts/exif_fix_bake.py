#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXIF 旋转像素烘焙 — 在 processed_detection_exiffix 副本上执行 (设计: EXIF_FIX_PLAN.md §3)

规则:
  1. 只烘焙 10 张 train/val EXIF 异常图 (O=6 -> 90°CW, O=8 -> 90°CCW);
  2. test 分片完全不动 (TRAIN_000064_photo-1.jpg 保持原样);
  3. 绝不修改原始 processed_detection / processed_detection_clean;
  4. 写入使用 tempfile + os.replace (生成新 inode), 避免截断与 clean 共享的硬链接。
"""
import os, sys, json, tempfile
import cv2

BASE = '/root/autodl-tmp/AIC2026_AgriVision'
CLEAN = f'{BASE}/dataset/processed_detection_clean'
COPY  = f'{BASE}/dataset/processed_detection_exiffix'

# 待烘焙清单 (split, filename, EXIF Orientation) —— 来自 exif_anomaly_report.json
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
ROT = {6: cv2.ROTATE_90_CLOCKWISE, 8: cv2.ROTATE_90_COUNTERCLOCKWISE}

# 读取 COCO JSON 尺寸 (供烘焙后即时自检)
def load_anno_dims():
    m = {}
    for split in ['train', 'val', 'test']:
        d = json.load(open(f'{CLEAN}/annotations/{split}.json'))
        for im in d['images']:
            m[(split, im['file_name'])] = (im['width'], im['height'])
    return m

def main():
    if os.path.isdir(COPY):
        print(f'[存在] {COPY} 已存在, 跳过复制')
    else:
        assert os.path.isdir(CLEAN), f'clean 副本缺失: {CLEAN}'
        print(f'[创建] cp -al {CLEAN} -> {COPY}')
        ret = os.system(f'cp -al "{CLEAN}" "{COPY}"')
        assert ret == 0, f'cp -al 失败 (ret={ret})'
        print('  ok.')

    # 预检副本结构完整性
    for split in ['train', 'val', 'test']:
        n_im = len([f for f in os.listdir(f'{COPY}/images/{split}') if f.endswith(('.jpg', '.JPG', '.png', '.PNG'))])
        n_txt = len(os.listdir(f'{COPY}/labels/{split}'))
        print(f'  副本 images/{split}={n_im}  labels/{split}={n_txt}')

    dims = load_anno_dims()
    baked = []
    for split, fname, ori in TARGETS:
        src = f'{COPY}/images/{split}/{fname}'
        assert os.path.isfile(src), f'[ERROR] 副本中不存在: {src}'
        assert os.path.isfile(f'{CLEAN}/images/{split}/{fname}'), f'[ERROR] clean 不存在: {CLEAN}/images/{split}/{fname}'
        img = cv2.imread(src)
        assert img is not None, f'[ERROR] cv2 无法读取: {src}'
        h, w = img.shape[:2]
        rotated = cv2.rotate(img, ROT[ori])
        jw, jh = dims[(split, fname)]  # JSON width, height
        assert (rotated.shape[1], rotated.shape[0]) == (jw, jh), \
            f'[ERROR] 烘焙尺寸不符 {fname}: {rotated.shape[1]}x{rotated.shape[0]} != JSON {jw}x{jh}'
        # temp + os.replace -> 新 inode, 不触碰 clean
        fd, tmp = tempfile.mkstemp(suffix='.jpg', dir=os.path.dirname(src))
        ok = cv2.imwrite(tmp, rotated, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        os.close(fd)
        assert ok, f'[ERROR] 写临时 JPEG 失败: {tmp}'
        os.chmod(tmp, 0o644)
        os.replace(tmp, src)
        baked.append((split, fname, ori))
        print(f'[烘焙] {split:5s} {fname[:58]:58s} O={ori} {w}x{h} -> {rotated.shape[1]}x{rotated.shape[0]} == JSON ✓')

    # 强制确认 test 未被触碰
    t = f'{COPY}/images/test/TRAIN_000064_photo-1.jpg'
    assert os.path.isfile(t), 'test 图缺失'
    os.stat(t)  # 仅读取
    print(f'[test] TRAIN_000064_photo-1.jpg 保持原样 (未烘焙): {t}')
    # 记录烘焙清单
    json.dump(baked, open(f'{BASE}/experiments/baseline_v1/metrics/exif_baked_list.json', 'w'),
              ensure_ascii=False, indent=1)
    print(f'烘焙完成: {len(baked)} 张 (train {sum(1 for s,_,_ in baked if s=="train")}, val {sum(1 for s,_,_ in baked if s=="val")})')
    print('下一步: python scripts/validate_exiffix.py 必须 7 项全 PASS 才可训练 Arm B')

if __name__ == '__main__':
    main()
