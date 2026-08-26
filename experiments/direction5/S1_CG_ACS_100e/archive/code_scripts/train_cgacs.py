# Copyright (c) 2026 AIC2026 AgriVision. All Rights Reserved.
#
# S1 CG-ACS — dual-stream training driver (stage-0 smoke + 100e formal).
#
# One optimizer step = 1 det batch (PlantDoc, det loss + lambda*aux cls loss)
#                     + 1 cls batch (PV_CORE, lambda*aux cls loss only)
# Gradient accumulation (both backwards before step) keeps the det-stream
# iteration count / LR schedule identical to A0.
#
# The aux classification head is controlled by the config: if `YOLOv3.
# aux_cls_head` is present the model carries CGACSAuxClsHead; otherwise the
# model is exactly A0 (aux_cls_head=None, det-only forward).
#
# Per-snapshot VAL eval (--eval) replicates A0's train-time evaluation
# semantics EXACTLY (verified against ppdet/engine/trainer.py + callbacks.py):
#   * eval runs only at snapshot epochs ((epoch+1) % snapshot_epoch == 0 or last)
#   * if use_ema: bias-corrected EMA weights (ema.apply()) are swapped into the
#     model BEFORE the eval; checkpoints save EMA as .pdparams and the raw
#     training weights as .pdema (same layout as A0's save_model)
#   * best_model is selected on VAL mAP@0.5:0.95 (COCOMetric 'bbox'[0])
#
# Usage (formal 100e, A0-parity bs16/lr0.002):
#   python -u scripts/train_cgacs.py -c configs/ppyoloe_plus_crn_s_100e_cgacs.yml \
#       --epoch 100 --det_batch 16 --cls_batch 16 --snapshot_epoch 5 --eval \
#       --save_dir experiments/direction5/S1_CG_ACS_100e/checkpoints \
#       --cls_index experiments/direction5/pv_cls_index.csv \
#       --log_file experiments/direction5/S1_CG_ACS_100e/train.log \
#       --vdl_log_dir experiments/direction5/S1_CG_ACS_100e/logs \
#       -o LearningRate.base_lr=0.002

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import math
import os
import sys

import yaml

import numpy as np
import paddle

# ---- project / PaddleDetection path setup -------------------------------
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, 'scripts'))
sys.path.insert(0, os.path.join(PROJECT, 'PaddleDetection'))

from ppdet.core.workspace import load_config, merge_config, create
from ppdet.engine import set_random_seed, init_parallel_env
from ppdet.optimizer import ModelEMA
from ppdet.utils.checkpoint import load_pretrain_weight
from ppdet.metrics import COCOMetric

from cgacs.pv_cls_dataset import PVClsDataset, collate_cls


def parse_args():
    parser = argparse.ArgumentParser(
        description='S1 CG-ACS dual-stream train (smoke / formal)')
    parser.add_argument('-c', '--config', required=True)
    parser.add_argument('--save_dir', default='experiments/direction5/checkpoints')
    parser.add_argument('--log_file', default='experiments/direction5/smoke_logs/train.log')
    parser.add_argument('--cls_index', default='experiments/direction5/pv_cls_index.csv')
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--det_batch', type=int, default=8)
    parser.add_argument('--cls_batch', type=int, default=8)
    parser.add_argument('--cls_stream', type=int, default=1,
                        help='enable PV_CORE cls stream (0 = A0-equivalent det-only)')
    parser.add_argument('--cls_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_steps', type=int, default=-1,
                        help='limit total optimizer steps (for quick checks)')
    parser.add_argument('--snapshot_epoch', type=int, default=1)
    parser.add_argument('--print_flops', type=int, default=0)
    parser.add_argument('--grad_check', type=int, default=0,
                        help='log param grad L1 norms at step 0 (gradient backprop check)')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='run VAL eval at each snapshot epoch + best model selection '
                             '(exact A0 train-time eval semantics)')
    parser.add_argument('--vdl_log_dir', default=None,
                        help='VisualDL log dir; if set, train scalars + eval metrics are logged')
    parser.add_argument('-o', '--opt', nargs='*', default=[])
    return parser.parse_args()


def parse_opt(opts):
    """Replicate ppdet.utils.cli.ArgsParser._parse_opt: -o 'k=v ...' -> dict."""
    config = {}
    if not opts:
        return config
    for s in opts:
        s = s.strip()
        k, v = s.split('=', 1)
        if '.' not in k:
            config[k] = yaml.load(v, Loader=yaml.Loader)
        else:
            keys = k.split('.')
            if keys[0] not in config:
                config[keys[0]] = {}
            cur = config[keys[0]]
            for idx, key in enumerate(keys[1:]):
                if idx == len(keys) - 2:
                    cur[key] = yaml.load(v, Loader=yaml.Loader)
                else:
                    cur[key] = {}
                    cur = cur[key]
    return config


def build_eval(FLAGS, cfg, logln):
    """Build VAL (EvalDataset + EvalReader + COCOMetric) exactly like
    trainer._eval_with_loader. Returns (eval_loader, metric)."""
    eval_dataset = create('EvalDataset')()
    eval_batch_sampler = paddle.io.BatchSampler(
        eval_dataset, batch_size=cfg.EvalReader['batch_size'])
    eval_loader = create('EvalReader')(
        eval_dataset, cfg.worker_num, batch_sampler=eval_batch_sampler)
    metric = COCOMetric(
        anno_file=eval_dataset.get_anno(),
        classwise=False,
        bias=0,
        IouType='bbox',
        save_prediction_only=False,
        save_threshold=0)
    logln('[eval] EvalDataset samples: {} | EvalReader bs: {} | anno: {}'.format(
        len(eval_dataset), cfg.EvalReader['batch_size'],
        eval_dataset.get_anno()))
    return eval_loader, metric


def run_val_eval(model, eval_loader, metric):
    """One VAL eval pass over EvalDataset. Returns (ap_all, ap50, ap75)."""
    model.eval()
    metric.reset()
    with paddle.no_grad():
        for step_id, data in enumerate(eval_loader):
            if paddle.base.core.is_compiled_with_cuda():
                for k, v in data.items():
                    if isinstance(v, paddle.Tensor):
                        data[k] = v.cuda()
            outs = model(data)
            metric.update(data, outs)
    metric.accumulate()
    metric.log()
    res = metric.get_results()
    if 'bbox' in res and len(res['bbox']) > 0:
        ap_all = float(res['bbox'][0])
        ap50 = float(res['bbox'][1]) if len(res['bbox']) > 1 else float('nan')
        ap75 = float(res['bbox'][2]) if len(res['bbox']) > 2 else float('nan')
    else:
        ap_all = ap50 = ap75 = float('nan')
    model.train()
    return ap_all, ap50, ap75


def main():
    args = parse_args()
    FLAGS = args

    # ---- working dir: MUST match A0 (run inside PaddleDetection so that
    # relative dataset_dir 'dataset/processed_detection' resolves to the
    # exiffix symlink, NOT the original dataset with corrupt files).
    # Resolve all user paths against PROJECT BEFORE chdir.
    PD = os.path.join(PROJECT, 'PaddleDetection')
    cfg_abs = FLAGS.config if os.path.isabs(FLAGS.config) else os.path.join(
        PROJECT, FLAGS.config)
    FLAGS.save_dir = FLAGS.save_dir if os.path.isabs(FLAGS.save_dir) else \
        os.path.join(PROJECT, FLAGS.save_dir)
    FLAGS.log_file = FLAGS.log_file if os.path.isabs(FLAGS.log_file) else \
        os.path.join(PROJECT, FLAGS.log_file)
    FLAGS.cls_index = FLAGS.cls_index if os.path.isabs(FLAGS.cls_index) else \
        os.path.join(PROJECT, FLAGS.cls_index)
    if FLAGS.vdl_log_dir:
        FLAGS.vdl_log_dir = FLAGS.vdl_log_dir if os.path.isabs(FLAGS.vdl_log_dir) else \
            os.path.join(PROJECT, FLAGS.vdl_log_dir)
    os.chdir(PD)
    FLAGS.config = cfg_abs

    # ---- config ----------------------------------------------------------
    cfg = load_config(FLAGS.config)
    merge_config(parse_opt(FLAGS.opt))
    cfg.epoch = FLAGS.epoch
    cfg.snapshot_epoch = FLAGS.snapshot_epoch
    cfg.save_dir = FLAGS.save_dir
    # 必须用 dict 式赋值: cfg['TrainReader'] 是 SchemaDict(dict 子类, 无 __setattr__),
    # `cfg.TrainReader.batch_size = x` 会写成实例属性而非 dict key, create('TrainReader')
    # 通过 `**cfg['TrainReader']` 读取仍取到基配置的 batch_size=8 (A0 用 -o TrainReader.batch_size
    # 是 merge_config dict 式赋值, 所以正确; 此处必须保持一致)。
    cfg['TrainReader']['batch_size'] = FLAGS.det_batch

    has_aux = 'aux_cls_head' in cfg['YOLOv3']
    if not has_aux and FLAGS.cls_stream:
        FLAGS.cls_stream = 0
        print('[INFO] config has no aux_cls_head -> cls stream disabled '
              '(A0-equivalent run)')

    set_random_seed(FLAGS.seed)
    init_parallel_env()

    os.makedirs(FLAGS.save_dir, exist_ok=True)
    if FLAGS.vdl_log_dir:
        os.makedirs(FLAGS.vdl_log_dir, exist_ok=True)
    log = open(FLAGS.log_file, 'w')

    def logln(msg=''):
        msg = str(msg)
        print(msg, flush=True)
        log.write(msg + '\n')
        log.flush()

    logln('=== S1 CG-ACS dual-stream train (formal 100e) ===')
    logln('config: {} | aux_cls_head: {} | cls_stream: {} | seed: {}'.format(
        FLAGS.config, has_aux, bool(FLAGS.cls_stream), FLAGS.seed))
    logln('det_batch: {} | cls_batch: {} | epoch: {} | snapshot_epoch: {}'.format(
        FLAGS.det_batch, FLAGS.cls_batch, FLAGS.epoch, FLAGS.snapshot_epoch))
    logln('base_lr: {} (A0-parity: config file default scaled per user decision '
          'bs16 -> 0.002)'.format(cfg['LearningRate']['base_lr']))
    logln('lambda_cls: {} | feat_idx: {} | aux_resize: 320'.format(
        cfg['CGACSAuxClsHead']['lambda_cls'],
        cfg['CGACSAuxClsHead']['feat_idx']))

    # ---- VisualDL ----------------------------------------------------------
    vdl = None
    if FLAGS.vdl_log_dir:
        try:
            from visualdl import LogWriter
            vdl = LogWriter(FLAGS.vdl_log_dir)
            logln('[vdl] logging to {}'.format(FLAGS.vdl_log_dir))
        except Exception as e:
            logln('[vdl] disabled: {}'.format(e))
            vdl = None

    # ---- model -----------------------------------------------------------
    model = create(cfg.architecture)
    model.load_meanstd(cfg['TestReader']['sample_transforms'])
    pretrain_weights = cfg.get('pretrain_weights', None)
    if pretrain_weights:
        load_pretrain_weight(model, pretrain_weights)
        logln('[model] loaded pretrain: {}'.format(pretrain_weights))
    n_params = sum(p.numel() for p in model.parameters())
    logln('[model] total trainable params: {}'.format(n_params))
    if has_aux:
        aux_n = sum(p.numel() for p in model.aux_cls_head.parameters())
        logln('[model] CGACSAuxClsHead params: {} (feat_idx={}, in_ch={})'.format(
            aux_n, model.aux_cls_head.feat_idx, model.aux_cls_head.in_channels))

    # ---- det loader (exact A0 path) --------------------------------------
    dataset = create('TrainDataset')()
    loader = create('TrainReader')(dataset, cfg.worker_num)
    steps_per_epoch = len(loader)
    actual_bs = loader._batch_sampler.batch_size
    if actual_bs != FLAGS.det_batch:
        raise RuntimeError(
            'det-loader actual batch_size {} != requested {} (cfg override '
            'failed)'.format(actual_bs, FLAGS.det_batch))
    logln('[det-loader] samples: {} | steps/epoch: {} | batch_size: {}'.format(
        len(dataset), steps_per_epoch, actual_bs))

    # ---- cls loader (PV_CORE) ----------------------------------------------
    cls_loader_iter = None
    if FLAGS.cls_stream:
        cls_ds = PVClsDataset(FLAGS.cls_index, num_classes=13, resize=320,
                              shuffle_seed=FLAGS.seed)
        cls_loader = paddle.io.DataLoader(
            cls_ds,
            batch_size=FLAGS.cls_batch,
            shuffle=True,
            drop_last=True,
            num_workers=FLAGS.cls_workers,
            collate_fn=collate_cls)
        logln('[cls-loader] PV_CORE samples: {} | cls batch: {}'.format(
            len(cls_ds), FLAGS.cls_batch))

    # ---- optimizer / lr / ema (A0 parity) --------------------------------
    lr = create('LearningRate')(steps_per_epoch)
    optimizer = create('OptimizerBuilder')(lr, model)

    use_ema = cfg.get('use_ema', False)
    ema = None
    if use_ema:
        ema = ModelEMA(
            model,
            decay=cfg.get('ema_decay', 0.9998),
            gamma=cfg.get('ema_gamma', 2000),
            ema_decay_type=cfg.get('ema_decay_type', 'threshold'),
            cycle_epoch=cfg.get('cycle_epoch', -1),
            ema_black_list=cfg.get('ema_black_list', None),
            ema_filter_no_grad=cfg.get('ema_filter_no_grad', False))
        logln('[ema] enabled decay={}'.format(cfg.get('ema_decay', 0.9998)))

    # ---- training loop -----------------------------------------------------
    model.train()
    total_steps = 0
    total_finite = 0
    log_iter = max(1, cfg.get('log_iter', 10))
    nan_occurred = False

    best_ap = -1.0
    best_epoch = -1
    final_ap = float('nan')
    final_ap50 = float('nan')
    final_ap75 = float('nan')
    eval_loader = None
    metric = None
    eval_step = 0
    vdl_train_step = 0

    def fetch_cls_batch():
        nonlocal cls_loader_iter
        if cls_loader_iter is None:
            cls_loader_iter = iter(cls_loader)
        try:
            return next(cls_loader_iter)
        except StopIteration:
            cls_loader_iter = iter(cls_loader)
            return next(cls_loader_iter)

    for epoch_id in range(FLAGS.epoch):
        logln('--- epoch {} start (steps/epoch={}) ---'.format(
            epoch_id, steps_per_epoch))
        for step_id, data in enumerate(loader):
            data['epoch_id'] = epoch_id
            if paddle.base.core.is_compiled_with_cuda():
                for k, v in data.items():
                    if isinstance(v, paddle.Tensor):
                        data[k] = v.cuda()
            det_out = model(data)
            det_loss = det_out['loss']
            det_loss.backward()

            cls_out = None
            if FLAGS.cls_stream:
                cls_data = fetch_cls_batch()
                if paddle.base.core.is_compiled_with_cuda():
                    for k, v in cls_data.items():
                        if isinstance(v, np.ndarray):
                            cls_data[k] = paddle.to_tensor(v).cuda()
                cls_out = model(cls_data)
                cls_out['loss'].backward()

            # gradient backprop verification (det+cls both reach aux & det heads)
            if FLAGS.grad_check and total_steps == 0:
                def _g1(pname):
                    p = model
                    for part in pname.split('.'):
                        p = getattr(p, part)
                    if p.grad is not None:
                        return float(paddle.abs(p.grad).sum().numpy())
                    return None
                g_aux = _g1('aux_cls_head.fc.weight') if has_aux else None
                g_det = _g1('yolo_head.proj_conv.weight')
                g_bn = None
                for n, p in model.named_parameters():
                    if n.endswith('backbone.early_conv.weight') or \
                            n.endswith('backbone.stem_conv.weight') or \
                            (g_bn is None and 'backbone' in n):
                        g_bn = _g1(n)
                        break
                logln('[grad_check] step0 | aux_fc.grad L1={} | '
                      'proj_conv.grad L1={} | backbone.grad L1={}'.format(
                          g_aux, g_det, g_bn))

            optimizer.step()
            curr_lr = optimizer.get_lr()
            lr.step()
            optimizer.clear_grad()
            if use_ema:
                ema.update()

            total_steps += 1
            all_fin = True
            for k, v in det_out.items():
                if isinstance(v, paddle.Tensor) and not bool(
                        paddle.isfinite(v).all()):
                    all_fin = False
            if cls_out:
                for k, v in cls_out.items():
                    if isinstance(v, paddle.Tensor) and not bool(
                            paddle.isfinite(v).all()):
                        all_fin = False
            total_finite += int(all_fin)
            if not all_fin:
                nan_occurred = True
                logln('[!!] NON-FINITE loss at step {}'.format(total_steps))

            if step_id % log_iter == 0 or step_id == steps_per_epoch - 1:
                det_loss_v = float(det_loss.numpy())
                det_items = ' | '.join(
                    '{}={:.5f}'.format(k, float(v.numpy()))
                    for k, v in det_out.items() if isinstance(v, paddle.Tensor))
                cls_items = ''
                if cls_out:
                    cls_items = ' | cls:' + ' | '.join(
                        '{}={:.5f}'.format(k, float(v.numpy()))
                        for k, v in cls_out.items()
                        if isinstance(v, paddle.Tensor))
                logln(
                    '[epoch {} step {}/{}] lr={:.6f} det:{} | det_total={:.5f}{} | finite={}'
                    .format(epoch_id, step_id, steps_per_epoch, curr_lr,
                            det_items, det_loss_v, cls_items, all_fin))

                if vdl is not None:
                    vdl.add_scalar('lr', float(curr_lr), vdl_train_step)
                    for k, v in det_out.items():
                        if isinstance(v, paddle.Tensor):
                            vdl.add_scalar(k, float(v.numpy()), vdl_train_step)
                    if cls_out:
                        for k, v in cls_out.items():
                            if isinstance(v, paddle.Tensor):
                                vdl.add_scalar('cls_' + k, float(v.numpy()),
                                               vdl_train_step)
                    vdl_train_step += 1

            if FLAGS.max_steps > 0 and total_steps >= FLAGS.max_steps:
                logln('[stop] reached --max_steps {}'.format(FLAGS.max_steps))
                break
        if FLAGS.max_steps > 0 and total_steps >= FLAGS.max_steps:
            break

        # ---- snapshot: A0-parity checkpoint + VAL eval --------------------
        is_snapshot = ((epoch_id + 1) % FLAGS.snapshot_epoch == 0 or
                       epoch_id == FLAGS.epoch - 1)
        if not is_snapshot:
            continue

        save_name = str(epoch_id) if epoch_id != FLAGS.epoch - 1 else 'model_final'
        if ema is not None:
            # A0 trainer.py: weight = deepcopy(model.state_dict()); model.set_dict(ema.apply())
            train_sd = copy.deepcopy(model.state_dict())
            ema_sd = ema.apply()
            model.set_dict(ema_sd)
            ckpt_weights = ema_sd       # -> .pdparams (bias-corrected EMA), A0 layout
            train_weights = train_sd    # -> .pdema  (raw training weights), A0 layout
        else:
            ckpt_weights = model.state_dict()
            train_weights = None

        save_path = os.path.join(FLAGS.save_dir, save_name)
        paddle.save(ckpt_weights, save_path + '.pdparams')
        if train_weights is not None:
            paddle.save(train_weights, save_path + '.pdema')
        opt_sd = optimizer.state_dict()
        opt_sd['last_epoch'] = epoch_id + 1
        paddle.save(opt_sd, save_path + '.pdopt')
        logln('[save] {} -> {}.pdparams/.pdema/.pdopt'.format(
            save_name, save_path))

        if FLAGS.eval:
            if eval_loader is None:
                eval_loader, metric = build_eval(FLAGS, cfg, logln)
            ap_all, ap50, ap75 = run_val_eval(model, eval_loader, metric)
            epoch_metric = {'metric': ap_all, 'epoch': epoch_id + 1}
            paddle.save(epoch_metric, save_path + '.pdstates')
            if vdl is not None:
                vdl.add_scalar('bbox-mAP', ap_all, eval_step)
                vdl.add_scalar('bbox-mAP0.5', ap50, eval_step)
                vdl.add_scalar('bbox-mAP0.75', ap75, eval_step)
                eval_step += 1

            if epoch_id == FLAGS.epoch - 1:
                final_ap, final_ap50, final_ap75 = ap_all, ap50, ap75

            if ap_all >= best_ap:
                best_ap = ap_all
                best_epoch = epoch_id + 1
                best_path = os.path.join(FLAGS.save_dir, 'best_model')
                paddle.save(ckpt_weights, best_path + '.pdparams')
                if train_weights is not None:
                    paddle.save(train_weights, best_path + '.pdema')
                opt_sd2 = optimizer.state_dict()
                opt_sd2['last_epoch'] = epoch_id + 1
                paddle.save(opt_sd2, best_path + '.pdopt')
                best_metric = {'metric': best_ap, 'epoch': best_epoch}
                paddle.save(best_metric, best_path + '.pdstates')
                logln('[best] VAL mAP@0.5:0.95 = {:.4f} @ epoch {} -> best_model'.format(
                    best_ap, best_epoch))
            else:
                logln('[val] epoch {} mAP@0.5:0.95={:.4f} | mAP50={:.4f} | '
                      'mAP75={:.4f} | best={:.4f}@epoch{}'.format(
                          epoch_id + 1, ap_all, ap50, ap75, best_ap, best_epoch))

        if ema is not None:
            # restore original training weights for continued training
            model.set_dict(train_sd)

    logln('=== done: total_steps={} | finite steps={} | nan_occurred={} ==='.format(
        total_steps, total_finite, nan_occurred))
    if FLAGS.eval:
        logln('=== best VAL mAP@0.5:0.95 = {:.4f} @ epoch {} | '
              'model_final VAL = {:.4f} (mAP50={:.4f}, mAP75={:.4f}) ==='.format(
                  best_ap, best_epoch, final_ap, final_ap50, final_ap75))
    log.close()

    rc = 0 if (not nan_occurred and total_finite == total_steps) else 1
    sys.exit(rc)


if __name__ == '__main__':
    main()
