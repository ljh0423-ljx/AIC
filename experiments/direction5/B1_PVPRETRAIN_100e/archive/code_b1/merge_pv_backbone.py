# -*- coding: utf-8 -*-
"""B1 阶段2 — 合并预训练权重: A0 的 neck/head (obj365 初始化) + PV 预训练 backbone。

背景: A0 的 pretrain_weights = ppyoloe_crn_s_obj365_pretrained.pdparams,
其 backbone.* 来自 Objects365。B1 唯一变量 = backbone 初始化来源,
因此生成合并文件:
  - neck.* / yolo_head.* 完全保持 obj365 原值 (A0 一致)
  - backbone.* 替换为 PV_CORE 分类预训练得到的 backbone 权重 (bare 键 -> 加 backbone. 前缀)

这样 B1 训练配置除 pretrain_weights 指向本合并文件外, 其余与 A0 完全相同。
load_pretrain_weight 的 match_state_dict 会对每个 model 键精确匹配 (a==b),
shape 不匹配自动跳过 (yolo_head.pred_cls 80->13 与 A0 行为一致)。

用法:
  python scripts/b1/merge_pv_backbone.py \
    --obj365 <obj365.pdparams> \
    --pv_backbone <best_backbone.pdparams> \
    --out experiments/direction5/B1_PVPRETRAIN_100e/merge/B1_pretrain_combined.pdparams
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, json, argparse
import paddle

_PROJ = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--obj365', required=True)
    ap.add_argument('--pv_backbone', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    sd_obj = paddle.load(args.obj365)
    sd_pv = paddle.load(args.pv_backbone)

    # 1) obj365 中保留非 backbone 段 (neck + yolo_head)
    keep = {k: v for k, v in sd_obj.items() if not k.startswith('backbone.')}
    print('[obj365] 总键 {} | 保留 non-backbone {} (neck+yolo_head)'.format(
        len(sd_obj), len(keep)))
    print('        obj365 顶层段: {}'.format(
        sorted({k.split('.')[0] for k in sd_obj})))

    # 2) PV backbone (bare 键 stem.*/stages.*) -> 加 backbone. 前缀
    pv_prefixed = {}
    for k, v in sd_pv.items():
        if k.startswith('backbone.'):
            pv_prefixed[k] = v
        else:
            pv_prefixed['backbone.' + k] = v
    print('[PV] 总键 {} -> backbone.* 键 {}'.format(len(sd_pv), len(pv_prefixed)))
    print('      PV backbone 顶层段: {}'.format(
        sorted({k.split('.')[1] for k in pv_prefixed})))

    # 3) 合并 + 一致性断言
    merged = dict(keep)
    merged.update(pv_prefixed)
    n_bb = sum(1 for k in merged if k.startswith('backbone.'))
    assert n_bb == len(pv_prefixed), 'backbone key count mismatch'
    # shape 一致性: PV backbone 与 obj365 backbone 对应键 shape 必须一致 (同一 backbone 结构)
    obj_bb = {k[len('backbone.'):]: v for k, v in sd_obj.items()
              if k.startswith('backbone.')}
    shape_mismatch = []
    for k, v in pv_prefixed.items():
        bare = k[len('backbone.'):]
        if bare in obj_bb and list(v.shape) != list(obj_bb[bare].shape):
            shape_mismatch.append((bare, list(v.shape), list(obj_bb[bare].shape)))
    if shape_mismatch:
        print('!! shape mismatch (PV vs obj365 backbone): {}'.format(
            shape_mismatch[:5]))
        sys.exit(1)
    print('[merged] 总键 {} (backbone={} neck+head={})'.format(
        len(merged), n_bb, len(merged) - n_bb))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    paddle.save(merged, args.out)
    print('合并权重已保存:', args.out)

    with open(args.out + '.json', 'w') as f:
        json.dump({
            'obj365': args.obj365, 'pv_backbone': args.pv_backbone,
            'merged': args.out,
            'n_total': len(merged), 'n_backbone': n_bb,
            'n_neck_head': len(merged) - n_bb,
            'shape_mismatch': shape_mismatch,
        }, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
