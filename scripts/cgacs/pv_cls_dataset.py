# Copyright (c) 2026 AIC2026 AgriVision. All Rights Reserved.
#
# PVClsDataset — image-level classification stream for S1 CG-ACS.
# - Read-only: references PV_CORE images by absolute path from an index CSV.
# - Output is aligned with the A0 det reader's NormalizeImage(is_scale=True):
#   float32 CHW image in [0, 1] (BGR channel order, same as PaddleDetection Decode).
# - No bbox is ever generated: image-level multi-label supervision only.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random

import cv2
import numpy as np
import paddle
import paddle.nn.functional as F


def build_pv_index(manifest_path, out_csv):
    """Generate index CSV (image_path, aic_class_id) from PV_CORE file_manifest.csv."""
    rows = []
    with open(manifest_path) as f:
        header = f.readline().strip().split(',')
        assert header[:7] == ['relative_path', 'pv_category', 'aic_class_id',
                              'aic_category_name', 'crop', 'size_bytes', 'md5'], header
        for line in f:
            parts = line.rstrip('\n').split(',')
            rel, cat, cid = parts[0], parts[1], parts[2]
            # absolute path under the PV_CORE root (dir of file_manifest.csv)
            base = os.path.dirname(os.path.abspath(manifest_path))
            rows.append((os.path.join(base, rel), int(cid)))
    rows.sort()
    with open(out_csv, 'w') as f:
        f.write('image_path,aic_class_id\n')
        for p, cid in rows:
            f.write('{},{}\n'.format(p, cid))
    return len(rows)


class PVClsDataset(paddle.io.Dataset):
    """Image-level classification dataset over PV_CORE.

    Args:
        index_csv: path to index CSV (image_path, aic_class_id).
        num_classes: 13 (AIC PlantDoc taxonomy, class_id 0-based).
        resize: int, cls stream input size (default 320).
        shuffle_seed: seed for internal per-epoch shuffle.
    """

    def __init__(self, index_csv, num_classes=13, resize=320, shuffle_seed=0):
        super(PVClsDataset, self).__init__()
        self.num_classes = num_classes
        self.resize = resize
        self._seed = shuffle_seed
        self._rng = random.Random(shuffle_seed)
        self.samples = []
        with open(index_csv) as f:
            assert f.readline().strip() == 'image_path,aic_class_id'
            for line in f:
                path, cid = line.rstrip('\n').rsplit(',', 1)
                self.samples.append((path, int(cid)))

    def __len__(self):
        return len(self.samples)

    def set_epoch(self, epoch):
        # reseed RNG deterministically per epoch (not used by loader, kept for parity)
        self._rng = random.Random(self._seed + epoch)

    def __getitem__(self, idx):
        path, cid = self.samples[idx]
        # BGR decode to match PaddleDetection Decode(); grey/rotated handled by cv2
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is None:
            raise IOError('PVClsDataset: cannot read {}'.format(path))
        im = cv2.resize(im, (self.resize, self.resize),
                        interpolation=cv2.INTER_CUBIC)
        im = im.astype(np.float32) / 255.0        # is_scale=True parity
        im = np.transpose(im, (2, 0, 1))          # HWC -> CHW

        target = np.zeros(self.num_classes, dtype=np.float32)
        target[cid] = 1.0

        return {
            'image': im,
            'aux_cls_target': target,
            'cls_label': np.array(cid, dtype=np.int64),
            'aux_mode': 'cls',
        }


def collate_cls(batch):
    """Collate a PVClsDataset batch into a model-ready inputs dict."""
    keys = list(batch[0].keys())
    out = {}
    for k in keys:
        v0 = batch[0][k]
        if isinstance(v0, str):
            out[k] = v0
        else:
            out[k] = np.stack([b[k] for b in batch])
    return out


def bce_with_logits_numpy(logits, targets):
    """Reference BCE-with-logits (mean over all elements) for smoke validation."""
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    maxes = np.maximum(logits, 0.0)
    loss = maxes - logits * targets + np.log1p(np.exp(-np.abs(logits)))
    return float(loss.mean())
