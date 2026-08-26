# -*- coding: utf-8 -*-
"""B1 兼容性检查:PV 分类 backbone 与 A0 PP-YOLOE+-s backbone 层名/shape 映射。

核验三个状态字典的 backbone.* 段:
  1) obj365 官方预训练 (A0 的 pretrain_weights, 含 backbone+neck+head)
  2) 检测模型 (A0 config create('YOLOv3'))
  3) 分类 backbone (create('CSPResNet') 同配置, 用于 PV_CORE 分类预训练)

要求: 三者 backbone.* 键名+shape 完全一致 → 训练完成的 PV backbone 可直接
替换 obj365 的 backbone.* 段, neck/head 保持 obj365 初始化(A0 一致)。

输出: 映射报告 (matched / name_mismatch_shape / missing / extra / 前N个键样例)
用法: python scripts/b1/compat_check.py [--out experiments/direction5/B1_PVPRETRAIN_100e/compat_report.json]
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, json, argparse
import paddle

_PROJ = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, os.path.join(_PROJ, 'PaddleDetection'))
os.chdir(os.path.join(_PROJ, 'PaddleDetection'))

from ppdet.core.workspace import load_config, create
from ppdet.modeling.backbones import CSPResNet
from ppdet.utils.checkpoint import is_url, get_weights_path

# PP-YOLOE+-s backbone 精确配置 (与 configs/ppyoloe/_base_/ppyoloe_plus_crn.yml + A0 顶层 depth/width 一致)
BB_KWARGS = dict(
    layers=[3, 6, 6, 3], channels=[64, 128, 256, 512, 1024], act='swish',
    return_idx=[1, 2, 3], use_large_stem=True, use_alpha=True,
    width_mult=0.50, depth_mult=0.33)
OBJ365 = os.path.join(os.path.expanduser('~'), '.cache', 'paddle', 'weights',
                      'ppyoloe_crn_s_obj365_pretrained.pdparams')
A0_CFG = os.path.join(_PROJ, 'configs', 'ppyoloe_plus_crn_s_100e_agrivision.yml')


def backbone_keys(sd, prefix=None):
    out = {}
    for k, v in sd.items():
        name = k
        if prefix and k.startswith(prefix):
            name = k[len(prefix):]
        out[name] = list(v.shape)
    return out


def compare(a_name, a, b_name, b):
    """a vs b 逐键比对, 返回统计。"""
    ka, kb = set(a), set(b)
    common = ka & kb
    shape_mismatch = [k for k in common if a[k] != b[k]]
    missing = sorted(ka - kb)   # a 有 b 无
    extra = sorted(kb - ka)     # b 有 a 无
    return {
        'a': a_name, 'b': b_name,
        'a_keys': len(a), 'b_keys': len(b),
        'matched': len(common) - len(shape_mismatch),
        'name_match_shape_mismatch': shape_mismatch,
        'missing_in_b': missing,
        'extra_in_b': extra,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(_PROJ, 'experiments',
                    'direction5', 'B1_PVPRETRAIN_100e', 'compat_report.json'))
    args = ap.parse_args()

    # 1) obj365 预训练
    obj = paddle.load(OBJ365)
    obj_bb = backbone_keys(obj, prefix='backbone.')
    print('[obj365] 总键 {} | backbone 键 {}'.format(len(obj), len(obj_bb)))

    # 2) 检测模型 (A0 config)
    cfg = load_config(A0_CFG)
    det = create(cfg.architecture)
    det_bb = backbone_keys(det.state_dict(), prefix='backbone.')
    print('[det-model] 总键 {} | backbone 键 {}'.format(len(det.state_dict()),
                                                      len(det_bb)))

    # 3) 分类 backbone (直接构造, 与 PP-YOLOE+-s 同配置)
    bb = CSPResNet(**BB_KWARGS)
    cls_bb = backbone_keys(bb.state_dict())
    print('[cls-backbone] 总键 {} (纯 backbone)'.format(len(cls_bb)))

    r1 = compare('obj365', obj_bb, 'det_model', det_bb)
    r2 = compare('cls_backbone', cls_bb, 'det_model', det_bb)
    r3 = compare('cls_backbone', cls_bb, 'obj365', obj_bb)
    rep = {'obj365_path': OBJ365, 'a0_cfg': A0_CFG,
           'obj365_total_keys': len(obj),
           'obj365_top_prefix': sorted({k.split('.')[0] for k in obj}),
           'classifier_p5_channels': None,
           'cls_vs_det': r2, 'cls_vs_obj365': r3, 'obj365_vs_det': r1}

    # 分类器 head 输入通道 = 检测 backbone 最深特征通道 (CSPResNet._out_channels[-1])
    try:
        rep['classifier_p5_channels'] = int(bb._out_channels[-1])
    except Exception as e:
        rep['classifier_p5_channels_note'] = str(e)

    # 采样展示匹配键
    sample = sorted(set(obj_bb) & set(cls_bb))
    rep['sample_keys'] = sample[:12]
    rep['n_common_backbone_keys'] = len(sample)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)

    print('\n=== 结论 ===')
    for name, r in [('cls_backbone vs det_model', r2),
                    ('cls_backbone vs obj365', r3),
                    ('obj365 vs det_model', r1)]:
        print('{}: 键 {}/{} 匹配, shape 不匹配 {}, missing {}, extra {}'.format(
            name, r['matched'], r['b_keys'], len(r['name_match_shape_mismatch']),
            len(r['missing_in_b']), len(r['extra_in_b'])))
        if r['name_match_shape_mismatch']:
            print('   shape 不匹配:', r['name_match_shape_mismatch'][:5])
        if r['missing_in_b']:
            print('   missing:', r['missing_in_b'][:8])
        if r['extra_in_b']:
            print('   extra:', r['extra_in_b'][:8])
    print('\n报告写入:', args.out)


if __name__ == '__main__':
    main()
