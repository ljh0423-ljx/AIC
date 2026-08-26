# -*- coding: utf-8 -*-
"""B1 DRIFT_GUARD 快照: 记录关键文件 md5, 训练后再复核, 证明条件零漂移。

记录对象:
  1. 代码: pv_cls_pretrain.py / merge_pv_backbone.py / analyze_val.py
  2. 配置: A0 / B1 / B1_smoke yml
  3. 数据索引: pv_cls_index.csv (分类), 检测数据集 train/val/test json
  4. 预训练产物: best/final backbone + full checkpoints
  5. 合并权重: B1_pretrain_combined.pdparams
  6. A0 参照: best_model / model_final (冻结, 防被改)
用法: python scripts/b1/snapshot_guard.py [--json 路径]
"""
from __future__ import print_function
import os, sys, hashlib, json, argparse

_PROJ = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
D = os.path.join(_PROJ, 'experiments', 'direction5', 'B1_PVPRETRAIN_100e')


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
        'code/pv_cls_pretrain.py': os.path.join(_PROJ, 'scripts', 'b1', 'pv_cls_pretrain.py'),
        'code/merge_pv_backbone.py': os.path.join(_PROJ, 'scripts', 'b1', 'merge_pv_backbone.py'),
        'code/analyze_val.py': os.path.join(_PROJ, 'scripts', 'analyze_val.py'),
        'code/compat_check.py': os.path.join(_PROJ, 'scripts', 'b1', 'compat_check.py'),
        'config/a0_agrivision.yml': os.path.join(_PROJ, 'configs', 'ppyoloe_plus_crn_s_100e_agrivision.yml'),
        'config/b1.yml': os.path.join(_PROJ, 'configs', 'ppyoloe_plus_crn_s_100e_b1.yml'),
        'config/b1_smoke.yml': os.path.join(_PROJ, 'configs', 'ppyoloe_plus_crn_s_100e_b1_smoke.yml'),
        'data/pv_cls_index.csv': os.path.join(_PROJ, 'experiments', 'direction5', 'pv_cls_index.csv'),
        'data/det_train.json': os.path.join(_PROJ, 'dataset', 'processed_detection_exiffix', 'annotations', 'train.json'),
        'data/det_val.json': os.path.join(_PROJ, 'dataset', 'processed_detection_exiffix', 'annotations', 'val.json'),
        'data/det_test.json': os.path.join(_PROJ, 'dataset', 'processed_detection_exiffix', 'annotations', 'test.json'),
    }
    # 预训练产物 (若存在)
    pret = os.path.join(D, 'pretrain')
    for f in ['best_backbone.pdparams', 'best_full.pdparams',
              'final_backbone.pdparams', 'final_full.pdparams',
              'cfg.json', 'train.csv', 'val.csv']:
        p = os.path.join(pret, f)
        if os.path.isfile(p):
            files['pretrain/' + f] = p
    # 合并权重 (若存在)
    merge = os.path.join(D, 'merge', 'B1_pretrain_combined.pdparams')
    if os.path.isfile(merge):
        files['merge/B1_pretrain_combined.pdparams'] = merge
    # B1 检测检查点 (若存在)
    ck = os.path.join(D, 'checkpoints')
    for f in ['best_model.pdparams', 'model_final.pdparams']:
        p = os.path.join(ck, f)
        if os.path.isfile(p):
            files['checkpoints/' + f] = p
    # A0 参照 (baseline_v1 checkpoints 顶层)
    a0 = os.path.join(_PROJ, 'experiments', 'baseline_v1', 'checkpoints')
    cand = [('a0/best_model.pdparams', os.path.join(a0, 'best_model.pdparams')),
            ('a0/model_final.pdparams', os.path.join(a0, 'model_final.pdparams'))]
    for name, p in cand:
        if os.path.isfile(p):
            files[name] = p

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
