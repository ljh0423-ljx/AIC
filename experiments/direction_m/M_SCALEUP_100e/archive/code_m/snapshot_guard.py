# -*- coding: utf-8 -*-
"""M 实验 DRIFT_GUARD 快照: 记录关键文件 md5, 训练后再复核, 证明条件零漂移。

记录对象:
  1. 代码: vram_probe.py / compat_check.py / analyze_val.py / snapshot_guard.py
  2. 配置: M 正式 / M smoke / A0 yml
  3. 数据索引: 检测数据集 train/val/test json (exiffix)
  4. 预训练: obj365 m 权重
  5. A0 参照: best_model / model_final (冻结, 防被改)
  6. M 检查点 (训练后记录): best_model / model_final
用法: python scripts/m/snapshot_guard.py [--json 路径]
"""
from __future__ import print_function
import os, sys, hashlib, json, argparse

_PROJ = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
D = os.path.join(_PROJ, 'experiments', 'direction_m', 'M_SCALEUP_100e')


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=os.path.join(D, 'DRIFT_GUARD.json'))
    args = ap.parse_args()

    files = {
        'code/vram_probe.py': os.path.join(_PROJ, 'scripts', 'm', 'vram_probe.py'),
        'code/compat_check.py': os.path.join(_PROJ, 'scripts', 'm', 'compat_check.py'),
        'code/snapshot_guard.py': os.path.join(_PROJ, 'scripts', 'm', 'snapshot_guard.py'),
        'code/analyze_val.py': os.path.join(_PROJ, 'scripts', 'analyze_val.py'),
        'config/m_formal.yml': os.path.join(_PROJ, 'configs', 'ppyoloe_plus_crn_m_100e_agrivision.yml'),
        'config/m_smoke.yml': os.path.join(_PROJ, 'configs', 'ppyoloe_plus_crn_m_100e_agrivision_smoke.yml'),
        'config/a0_agrivision.yml': os.path.join(_PROJ, 'configs', 'ppyoloe_plus_crn_s_100e_agrivision.yml'),
        'data/det_train.json': os.path.join(_PROJ, 'dataset', 'processed_detection_exiffix', 'annotations', 'train.json'),
        'data/det_val.json': os.path.join(_PROJ, 'dataset', 'processed_detection_exiffix', 'annotations', 'val.json'),
        'data/det_test.json': os.path.join(_PROJ, 'dataset', 'processed_detection_exiffix', 'annotations', 'test.json'),
        'pretrain/obj365_m.pdparams': os.path.join(os.path.expanduser('~'), '.cache', 'paddle', 'weights', 'ppyoloe_crn_m_obj365_pretrained.pdparams'),
    }
    # A0 参照
    a0 = os.path.join(_PROJ, 'experiments', 'baseline_v1', 'checkpoints')
    for name, fn in [('a0/best_model.pdparams', 'best_model.pdparams'),
                     ('a0/model_final.pdparams', 'model_final.pdparams')]:
        p = os.path.join(a0, fn)
        if os.path.isfile(p):
            files[name] = p
    # M 检查点 (训练后存在则记录)
    ck = os.path.join(D, 'checkpoints')
    for fn in ['best_model.pdparams', 'model_final.pdparams']:
        p = os.path.join(ck, fn)
        if os.path.isfile(p):
            files['checkpoints/' + fn] = p

    snap = {'timestamp': None, 'phase': None, 'files': {}}
    missing = []
    for name, p in sorted(files.items()):
        if os.path.isfile(p):
            snap['files'][name] = {'md5': md5(p), 'size': os.path.getsize(p), 'path': p}
        else:
            missing.append(name)
    with open(args.json, 'w') as f:
        json.dump(snap, f, indent=2, ensure_ascii=False)
    print('快照写入:', args.json)
    print('已记录 {} 文件'.format(len(snap['files'])))
    if missing:
        print('缺失(未记录):', missing)


if __name__ == '__main__':
    main()
