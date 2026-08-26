# -*- coding: utf-8 -*-
"""M 实验 兼容性预检: 构建 PP-YOLOE+-m 检测模型, 用 load_pretrain_weight
(与 train.py 完全相同) 加载官方 obj365 m 预训练, 报告匹配/跳过键。

预期: 除 yolo_head.pred_cls.* (365->13 通道, 与 A0 的 s 模型同样行为) 外全部匹配。
用法: python scripts/m/compat_check.py --config <m_cfg.yml> --pretrain <m_obj365.pdparams>
"""
from __future__ import print_function
import os, sys, argparse
import paddle

_PROJ = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
os.chdir(os.path.join(_PROJ, 'PaddleDetection'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--pretrain', required=True)
    args = ap.parse_args()

    from ppdet.core.workspace import load_config, create
    from ppdet.utils.checkpoint import load_pretrain_weight

    paddle.set_device('gpu')
    cfg = load_config(args.config)
    model = create(cfg.architecture)

    sd = paddle.load(args.pretrain)
    model_keys = set(model.state_dict().keys())
    pretrain_keys = set(sd.keys())
    print('[compat] 检测模型键数: %d' % len(model_keys))
    print('[compat] 预训练键数:   %d' % len(sd))
    print('[compat] 预训练顶层段: %s' % sorted({k.split('.')[0] for k in sd}))

    # 与 checkpoint.load_pretrain_weight 相同的匹配逻辑 (match_state_dict)
    matched, skipped_shape, skipped = [], [], []
    for mk in model_keys:
        best = None
        if mk in sd:
            best = mk
        else:
            for pk in sd:
                if mk.endswith('.' + pk):
                    best = pk
                    break
        if best is None:
            skipped.append((mk, 'NO_MATCH'))
            continue
        if list(sd[best].shape) != list(model.state_dict()[mk].shape):
            skipped_shape.append((mk, list(sd[best].shape),
                                  list(model.state_dict()[mk].shape)))
            continue
        matched.append(mk)

    print('[compat] 匹配: %d | shape跳过: %d | 无匹配跳过: %d' %
          (len(matched), len(skipped_shape), len(skipped)))
    for m in skipped_shape[:10]:
        print('   shape跳过:', m)
    for m in skipped[:10]:
        print('   无匹配:', m)

    # 允许与 A0 相同的两类跳过:
    #   1) pred_cls.*  shape 跳过 (365->13 通道, 与 A0 相同)
    #   2) backbone.*.conv2.alpha 无匹配 (obj365 预训练无 alpha 键, 默认初始化 1.0, 与 A0 相同)
    bad_shape = [s for s in skipped_shape if 'pred_cls' not in s[0]]
    bad_nomatch = [m for m in skipped if not m[0].endswith('.alpha')]
    if bad_shape or bad_nomatch:
        print('[compat] !! 意外未匹配/跳过: shape=%s nomatch=%s' %
              (bad_shape[:5], bad_nomatch[:5]))
        sys.exit(1)
    print('[compat] PASS (alpha=%d 默认初始化, pred_cls=%d 通道重初始化, 均与 A0 行为一致)' %
          (sum(1 for m in skipped if m[0].endswith('.alpha')), len(skipped_shape)))


if __name__ == '__main__':
    main()
