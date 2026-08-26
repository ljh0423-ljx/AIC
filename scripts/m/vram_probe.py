# -*- coding: utf-8 -*-
"""M 实验 VRAM 探针: 用与 trainer.py 完全相同的 DataLoader+model(data) 路径,
测量 PP-YOLOE+-m (depth=0.67/width=0.75) 在训练模式、最大输入 768x768 下的
峰值显存, 以决定 batch_size (优先保持与 A0 一致 bs=16/lr=0.002)。

用法:
  python scripts/m/vram_probe.py --config <m_cfg.yml> --bs 16 [--steps 12]
同时打印参数量。
"""
from __future__ import print_function
import os, sys, argparse, time
import paddle

_PROJ = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
os.chdir(os.path.join(_PROJ, 'PaddleDetection'))  # 与 train.py 相同 cwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--steps', type=int, default=12)
    args = ap.parse_args()

    from ppdet.core.workspace import load_config, create

    paddle.set_device('gpu')
    cfg = load_config(args.config)
    cfg['TrainReader']['batch_size'] = args.bs
    print('[probe] config=%s bs=%d' % (os.path.basename(args.config), args.bs))

    model = create(cfg.architecture)
    dataset = create('TrainDataset')()
    loader = create('TrainReader')(dataset, cfg.worker_num)

    # 强制训练最大尺寸 768x768 (BatchRandomResize 是第一个 batch transform)
    try:
        loader._batch_transforms.transforms_cls[0].target_size = [768]
        print('[probe] BatchRandomResize 强制 target_size=768')
    except Exception as e:
        print('[probe] 无法强制768, 用随机尺寸: %s' % e)

    n_params = sum(p.size if hasattr(p, 'size') else p.numpy().size
                   for p in model.parameters())
    print('[probe] 模型参数量: %d (%.2f M)' % (n_params, n_params / 1e6))

    model.train()
    peak = 0
    t0 = time.time()
    for i, data in enumerate(loader):
        if i >= args.steps:
            break
        if isinstance(data, dict):
            data['epoch_id'] = 0  # 与 trainer.py 相同: 注入 epoch_id
        outs = model(data)
        loss = outs['loss']
        loss.backward()
        model.clear_gradients()
        m = paddle.device.cuda.max_memory_allocated() / 1e9
        peak = max(peak, m)
        print('  step %2d loss=%.4f peak=%.2fGB' % (i, float(loss.numpy()), m))
    tot = paddle.device.cuda.get_device_properties(0).total_memory / 1e9
    print('[probe] DONE 峰值显存=%.2f GB / %.1f GB GPU (%.0f%%)' %
          (peak, tot, 100.0 * peak / tot))
    print('[probe] 耗时 %.1fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
