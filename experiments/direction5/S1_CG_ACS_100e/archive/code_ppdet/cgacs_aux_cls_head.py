# Copyright (c) 2026 AIC2026 AgriVision. All Rights Reserved.
#
# S1 CG-ACS (Classification-Grounded Auxiliary Classification Supervision):
# image-level 13-class auxiliary classification head on shared P3 neck features.
# Train-only. Default off. Does NOT modify the detection head (PPYOLOEHead).

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdet.core.workspace import register

__all__ = ['CGACSAuxClsHead']


@register
class CGACSAuxClsHead(nn.Layer):
    """
    Image-level multi-label auxiliary classification head.

    Input : neck_feats[feat_idx] (P3 by default, [B, C, H, W])
    Output: logits [B, num_classes]

    Loss: lambda_cls * BCE-with-logits, averaged over all [B, C] elements.
    Works for both one-hot (PV_CORE) and multi-hot (PlantDoc) targets.
    """
    __shared__ = ['num_classes']

    def __init__(self, in_channels=96, num_classes=13, lambda_cls=0.1,
                 feat_idx=0, input_shape=None):
        super(CGACSAuxClsHead, self).__init__()
        if input_shape is not None and in_channels <= 0:
            in_channels = input_shape[feat_idx].channels
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.lambda_cls = float(lambda_cls)
        self.feat_idx = feat_idx

        self.gap = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, feats):
        """feats: list of neck features, use feats[self.feat_idx] (P3 by default)."""
        x = feats[self.feat_idx]
        x = self.gap(x)            # [B, C, 1, 1]
        x = x.flatten(1)           # [B, C]
        logits = self.fc(x)        # [B, num_classes]
        return logits

    def loss(self, logits, targets):
        """BCE-with-logits, mean over [B, num_classes]. Scaled by lambda_cls."""
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='mean')
        return self.lambda_cls * loss

    @staticmethod
    def build_target_from_gt(gt_class, num_classes):
        """
        Multi-hot image-level target from detection gt_class.
        gt_class: list of [num_boxes] (0-based class ids) per image.
        Returns [B, num_classes] float32.
        """
        batch_size = len(gt_class)
        targets = np.zeros((batch_size, num_classes), dtype=np.float32)
        for b in range(batch_size):
            for c in np.asarray(gt_class[b]).reshape(-1):
                targets[b, int(c)] = 1.0
        return paddle.to_tensor(targets)
