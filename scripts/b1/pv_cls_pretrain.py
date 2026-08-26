# -*- coding: utf-8 -*-
"""B1 阶段1 — PV_CORE 13类图像分类预训练 (两阶段中的第一阶段)。

模型: PP-YOLOE+-s 同配置 backbone (CSPResNet depth_mult=0.33 width_mult=0.50
      use_large_stem use_alpha) + GAP + Linear(512, 13)。backbone 从零初始化
      (随机 init, 不加载任何预训练), 只在 PV_CORE 上训练。
目标: 训练完成提取 backbone.* 权重 → 作为 B1 检测微调的 pretrain_weights 来源。

监督: 单标签 13 类 CrossEntropy (label_smoothing=0.1), 与 PV_CORE 每图单一
      aic_class_id 一致。
输入: 320x320, /255 (BGR), 与检测管线 NormalizeImage(is_scale=True) 一致,
      保证预训练与微调输入分布一致。
数据: PV_CORE 19233 张 → 90/10 分层划分 (seed 固定) → 分类训练/验证。
产物: best_full / final_full / best_backbone.pdparams (裸键 stem.*/stages.*)
      train.log / val_history.json / cfg.json / train-val 索引 csv

用法:
  python scripts/b1/pv_cls_pretrain.py \
    --index_csv experiments/direction5/pv_cls_index.csv \
    --out_dir experiments/direction5/B1_PVPRETRAIN_100e/pretrain \
    --epochs 200 --batch 256 --lr 0.1 --warmup 5 --resize 320 --workers 8
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, json, time, math, argparse, random, threading
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

# 关键: 默认 cv2 线程数=128, 每次 resize 的小图也要付出线程调度开销 (~3ms),
# 设 1 线程后 resize 仅 ~0.3ms (10x). 增广线程池自身并行, 无需 cv2 内并行。
cv2.setNumThreads(1)

_PROJ = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, os.path.join(_PROJ, 'PaddleDetection'))
os.chdir(os.path.join(_PROJ, 'PaddleDetection'))

from ppdet.modeling.backbones import CSPResNet
from ppdet.modeling.ops import get_act_fn

# ---- 与 PP-YOLOE+-s 完全一致的 backbone 配置 (见 compat_check.BB_KWARGS) ----
BB_KWARGS = dict(
    layers=[3, 6, 6, 3], channels=[64, 128, 256, 512, 1024], act='swish',
    return_idx=[1, 2, 3], use_large_stem=True, use_alpha=True,
    width_mult=0.50, depth_mult=0.33)

CLASSES = [
    'Tomato Early blight leaf', 'Tomato Septoria leaf spot', 'Tomato leaf',
    'Tomato leaf bacterial spot', 'Tomato leaf late blight',
    'Tomato leaf mosaic virus', 'Tomato leaf yellow virus', 'Tomato mold leaf',
    'Apple Scab Leaf', 'Apple leaf', 'Apple rust leaf', 'grape leaf',
    'grape leaf black rot'
]


# ---------------- dataset ----------------
CACHE_MAXDIM = 384  # 解码缓存最大边 (uint8), 本机 RAM 充足, 全部缓存内存消除磁盘 I/O


def _decode_resize(args):
    """进程池 worker: 解码 + 等比例缩放至 max_dim, 返回 uint8 BGR。"""
    path, max_dim = args
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise IOError('cannot read {}'.format(path))
    h, w = im.shape[:2]
    m = max(h, w)
    if m > max_dim:
        k = max_dim / m
        im = cv2.resize(im, (max(1, int(w * k)), max(1, int(h * k))),
                        interpolation=cv2.INTER_AREA)
    return im


class PVClsPretrainDataset(paddle.io.Dataset):
    def __init__(self, index_csv, resize=320, train=True, seed=0,
                 cache_maxdim=CACHE_MAXDIM, decode_workers=16):
        self.resize = resize
        self.train = train
        self._rng = random.Random(seed)
        self.samples = []
        with open(index_csv) as f:
            assert f.readline().strip() == 'image_path,aic_class_id'
            for line in f:
                path, cid = line.rstrip('\n').rsplit(',', 1)
                self.samples.append((path, int(cid)))
        self._rng.shuffle(self.samples)
        # 全量解码缓存到内存 (多进程并行; 父进程持有, fork 后 worker COW 共享)
        t0 = time.time()
        paths = [p for p, _ in self.samples]
        if decode_workers > 1 and len(paths) > 1:
            with Pool(decode_workers) as pool:
                self._cache = pool.map(
                    _decode_resize, [(p, cache_maxdim) for p in paths])
        else:
            self._cache = [_decode_resize((p, cache_maxdim)) for p in paths]
        self._cache = [np.ascontiguousarray(im) for im in self._cache]
        print('[{}] decode cache: {} imgs @maxdim={} ({:.0f}s)'.format(
            'train' if train else 'val', len(self._cache), cache_maxdim,
            time.time() - t0), flush=True)

    def __len__(self):
        return len(self.samples)

    def _aug(self, im):
        h, w = im.shape[:2]
        if self.train:
            # RandomResizedCrop (scale 0.4~1.0, ratio 3/4~4/3)
            scale = self._rng.uniform(0.4, 1.0)
            area = h * w * scale
            ratio = math.exp(self._rng.uniform(math.log(0.75), math.log(4.0 / 3)))
            th = int(round(math.sqrt(area / ratio)))
            tw = int(round(math.sqrt(area * ratio)))
            th = min(max(th, 32), h)
            tw = min(max(tw, 32), w)
            y0 = self._rng.randint(0, h - th) if th < h else 0
            x0 = self._rng.randint(0, w - tw) if tw < w else 0
            im = im[y0:y0 + th, x0:x0 + tw]
            im = cv2.resize(im, (self.resize, self.resize),
                            interpolation=cv2.INTER_LINEAR)
            if self._rng.random() < 0.5:
                im = im[:, ::-1]
        else:
            im = cv2.resize(im, (self.resize, self.resize),
                            interpolation=cv2.INTER_CUBIC)
        return im

    def __getitem__(self, idx):
        cid = self.samples[idx][1]
        im = self._cache[idx]
        im = self._aug(im)  # uint8 HWC BGR
        return {'image': im, 'label': np.array(cid, dtype=np.int64)}


def collate(batch):
    # 返回 uint8 HWC (传输量小), GPU 转换由训练循环完成
    im = np.stack([b['image'] for b in batch])
    lb = np.stack([b['label'] for b in batch])
    return {'image': im, 'label': lb}


def to_gpu_batch(batch, place):
    im = paddle.to_tensor(batch['image'], place=place)  # [B,H,W,3] uint8
    im = im.transpose([0, 3, 1, 2]).astype(paddle.float32) / 255.0
    lb = paddle.to_tensor(batch['label'], place=place)
    return im, lb


class EpochPrefetcher:
    """按 epoch 后台预生成全部增强 batch (uint8), 与训练重叠。

    数据全在内存 (uint8 缓存), 增强在后台线程池并行完成, 输出为整 epoch
    的 batch 列表 (numpy uint8), 主循环仅做 GPU 转换+训练, 彻底绕开
    DataLoader 的逐 batch 传输/worker 冷启动开销。

    v2: 单一持久 ThreadPoolExecutor (不再每 epoch 新建/泄漏 executor);
        prefetch 将 67 个 chunk 全部提交, get 聚合结果。
    """

    def __init__(self, ds, batch, resize, workers=8):
        self.ds = ds
        self.batch = batch
        self.resize = resize
        self.workers = workers
        self.n = len(ds)
        self.n_batches = self.n // batch
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._next = None

    def _aug_one(self, im, seed):
        rng = np.random.default_rng(seed)
        h, w = im.shape[:2]
        scale = float(rng.uniform(0.4, 1.0))
        area = h * w * scale
        ratio = math.exp(float(rng.uniform(math.log(0.75), math.log(4.0 / 3))))
        th = int(round(math.sqrt(area / ratio)))
        tw = int(round(math.sqrt(area * ratio)))
        th = min(max(th, 32), h)
        tw = min(max(tw, 32), w)
        y0 = int(rng.integers(0, h - th + 1)) if h - th > 0 else 0
        x0 = int(rng.integers(0, w - tw + 1)) if w - tw > 0 else 0
        im = im[y0:y0 + th, x0:x0 + tw]
        im = cv2.resize(im, (self.resize, self.resize),
                        interpolation=cv2.INTER_LINEAR)
        if rng.random() < 0.5:
            im = im[:, ::-1]
        return np.ascontiguousarray(im)

    def _build_chunk(self, ids, seed_base):
        ims = np.empty((len(ids), self.resize, self.resize, 3), dtype=np.uint8)
        lbs = np.empty((len(ids),), dtype=np.int64)
        for j, i in enumerate(ids):
            ims[j] = self._aug_one(self.ds._cache[i], seed_base + j)
            lbs[j] = self.ds.samples[i][1]
        return {'image': ims, 'label': lbs}

    def prefetch(self, epoch):
        rng = np.random.default_rng(epoch)
        idx = rng.permutation(self.n)[:self.n_batches * self.batch]
        ids_list = [idx[s * self.batch:(s + 1) * self.batch]
                    for s in range(self.n_batches)]
        seed_base = epoch * 1000003
        self._next = [self._pool.submit(self._build_chunk, ids, seed_base)
                      for ids in ids_list]

    def get(self):
        batches = [f.result() for f in self._next]
        self._next = None
        return batches


# ---------------- model ----------------
class PVClsModel(nn.Layer):
    """backbone(PP-YOLOE+-s 同构) + GAP + FC(512,13)。"""
    def __init__(self, num_classes=13):
        super(PVClsModel, self).__init__()
        self.backbone = CSPResNet(**BB_KWARGS)
        p5 = int(self.backbone._out_channels[-1])
        self.pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Linear(p5, num_classes)

    def forward(self, image):
        outs = self.backbone({'image': image})
        f = outs[-1]
        g = self.pool(f).flatten(1)
        return self.fc(g)


# ---------------- split ----------------
def stratify_split(index_csv, out_dir, seed=0, val_frac=0.1):
    rows = []
    with open(index_csv) as f:
        assert f.readline().strip() == 'image_path,aic_class_id'
        for line in f:
            path, cid = line.rstrip('\n').rsplit(',', 1)
            rows.append((path, int(cid)))
    by_cls = {}
    for p, c in rows:
        by_cls.setdefault(c, []).append((p, c))
    rng = random.Random(seed)
    train_rows, val_rows = [], []
    for c, items in sorted(by_cls.items()):
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_frac)))
        val_rows.extend(items[:n_val])
        train_rows.extend(items[n_val:])
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    os.makedirs(out_dir, exist_ok=True)
    tr = os.path.join(out_dir, 'train.csv')
    va = os.path.join(out_dir, 'val.csv')
    with open(tr, 'w') as f:
        f.write('image_path,aic_class_id\n')
        for p, c in train_rows:
            f.write('{},{}\n'.format(p, c))
    with open(va, 'w') as f:
        f.write('image_path,aic_class_id\n')
        for p, c in val_rows:
            f.write('{},{}\n'.format(p, c))
    return tr, va, len(train_rows), len(val_rows)


# ---------------- train ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index_csv', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=0.1)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--resize', type=int, default=320)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--label_smooth', type=float, default=0.1)
    ap.add_argument('--val_every', type=int, default=1)
    args = ap.parse_args()
    args.index_csv = args.index_csv if os.path.isabs(args.index_csv) else \
        os.path.join(_PROJ, args.index_csv)
    args.out_dir = args.out_dir if os.path.isabs(args.out_dir) else \
        os.path.join(_PROJ, args.out_dir)

    paddle.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    paddle.set_device('gpu')  # 模型参数创建在 GPU 上

    tr_csv, va_csv, n_tr, n_va = stratify_split(
        args.index_csv, args.out_dir, seed=args.seed)
    print('[data] train={} val={} -> {} / {}'.format(n_tr, n_va, tr_csv, va_csv))

    ds_tr = PVClsPretrainDataset(tr_csv, resize=args.resize, train=True)
    ds_va = PVClsPretrainDataset(va_csv, resize=args.resize, train=False)
    dl_tr = paddle.io.DataLoader(
        ds_tr, batch_size=args.batch, shuffle=True, drop_last=True,
        num_workers=args.workers, collate_fn=collate,
        persistent_workers=(args.workers > 0))
    dl_va = paddle.io.DataLoader(
        ds_va, batch_size=args.batch, shuffle=False, drop_last=False,
        num_workers=args.workers, collate_fn=collate,
        persistent_workers=(args.workers > 0))

    model = PVClsModel(num_classes=13)
    model.train()
    # 收集总参数量 (仅记录)
    n_param = sum(np.prod(p.shape) for p in model.parameters())
    print('[model] params={} (backbone 同 PP-YOLOE+-s)'.format(n_param))

    steps_per_epoch = len(dl_tr)
    total_steps = args.epochs * steps_per_epoch
    warm_steps = args.warmup * steps_per_epoch

    place = paddle.CUDAPlace(0)

    lr_sched = paddle.optimizer.lr.LambdaDecay(
        learning_rate=args.lr,
        lr_lambda=lambda step: (
            step / max(warm_steps, 1)
            if step < warm_steps else
            0.5 * (1 + math.cos(math.pi * (step - warm_steps) /
                                max(total_steps - warm_steps, 1)))))
    opt = paddle.optimizer.Momentum(
        learning_rate=lr_sched, momentum=0.9, parameters=model.parameters(),
        weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)

    cfg_rec = {'index_csv': args.index_csv, 'out_dir': args.out_dir,
               'epochs': args.epochs, 'batch': args.batch, 'lr': args.lr,
               'warmup': args.warmup, 'resize': args.resize,
               'workers': args.workers, 'seed': args.seed,
               'label_smooth': args.label_smooth,
               'n_train': n_tr, 'n_val': n_va,
               'steps_per_epoch': steps_per_epoch, 'total_steps': total_steps,
               'model': 'CSPResNet PP-YOLOE+-s 同构 + GAP + FC(512,13)',
               'init': 'backbone 从零随机初始化, 仅 PV_CORE 训练',
               'loss': 'CrossEntropy label_smoothing=%.2f' % args.label_smooth,
               'classes': CLASSES}
    with open(os.path.join(args.out_dir, 'cfg.json'), 'w') as f:
        json.dump(cfg_rec, f, indent=2, ensure_ascii=False)

    log = open(os.path.join(args.out_dir, 'train.log'), 'w')
    def logln(s):
        print(s, flush=True)
        log.write(s + '\n')
        log.flush()

    best_acc, best_epoch = -1.0, -1
    hist = {'val_acc': [], 'val_loss': [], 'train_loss': []}
    global_step = 0
    t0 = time.time()
    logln('=== PV_CORE 分类预训练开始 ===')
    prefetcher = EpochPrefetcher(ds_tr, args.batch, args.resize, args.workers)
    prefetcher.prefetch(1)
    for ep in range(1, args.epochs + 1):
        ep_loss, ep_steps, ep_n = 0.0, 0, 0
        _t_ep = time.time()
        batches = prefetcher.get()
        if ep < args.epochs:
            prefetcher.prefetch(ep + 1)  # 后台构建下一 epoch, 与训练重叠
        for batch in batches:
            im, lb = to_gpu_batch(batch, place)
            logits = model(im)
            loss = crit(logits, lb)
            loss.backward()
            opt.step()
            opt.clear_grad()
            lr_sched.step()
            ep_loss += float(loss.item())
            ep_steps += 1
            ep_n += int(lb.shape[0])
            global_step += 1
        _t_tr = time.time() - _t_ep
        avg_tr = ep_loss / max(ep_steps, 1)
        hist['train_loss'].append(avg_tr)

        if ep % args.val_every == 0 or ep == args.epochs:
            model.eval()
            v_loss, v_n, v_cor = 0.0, 0, 0
            with paddle.no_grad():
                for batch in dl_va:
                    im, lb = to_gpu_batch(batch, place)
                    logits = model(im)
                    loss = crit(logits, lb)
                    v_loss += float(loss.item()) * int(lb.shape[0])
                    pred = logits.argmax(1)
                    v_cor += int((pred == lb).sum())
                    v_n += int(lb.shape[0])
            _t_va = time.time() - _t_ep
            logln('  [timing] train=%.1fs val=%.1fs' % (_t_tr, _t_va - _t_tr))
            vacc = v_cor / v_n
            vloss = v_loss / v_n
            hist['val_acc'].append(vacc)
            hist['val_loss'].append(vloss)
            model.train()
            best = ' *' if vacc > best_acc else ''
            if vacc > best_acc:
                best_acc, best_epoch = vacc, ep
                paddle.save(model.state_dict(),
                            os.path.join(args.out_dir, 'best_full.pdparams'))
                bb = {k: v for k, v in model.state_dict().items()
                      if k.startswith('backbone.')}
                paddle.save(
                    {k[len('backbone.'):]: v for k, v in bb.items()},
                    os.path.join(args.out_dir, 'best_backbone.pdparams'))
            logln('[%3d/%d] train_loss=%.4f | val_loss=%.4f val_acc=%.4f | '
                  'best=%.4f@%d%s | %.1fs | lr=%.5f' % (
                      ep, args.epochs, avg_tr, vloss, vacc,
                      best_acc, best_epoch, best,
                      time.time() - t0, float(lr_sched.get_lr())))

    paddle.save(model.state_dict(),
                os.path.join(args.out_dir, 'final_full.pdparams'))
    bb = {k: v for k, v in model.state_dict().items() if k.startswith('backbone.')}
    paddle.save({k[len('backbone.'):]: v for k, v in bb.items()},
                os.path.join(args.out_dir, 'final_backbone.pdparams'))
    logln('=== done: epochs={} best_val_acc={:.4f}@epoch{} total_time={:.0f}s '
          'steps={} ==='.format(args.epochs, best_acc, best_epoch,
                                time.time() - t0, global_step))
    with open(os.path.join(args.out_dir, 'val_history.json'), 'w') as f:
        json.dump(hist, f, indent=2)
    log.close()


if __name__ == '__main__':
    main()
