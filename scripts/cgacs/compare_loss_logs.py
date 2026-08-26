# Copyright (c) 2026 AIC2026 AgriVision. All Rights Reserved.
#
# Compare per-step training losses between two logs (A0 reference vs S1-off),
# item-by-item on every logged key. Used for smoke verification (a):
# "A0 关闭 S1 时 loss 逐项与原始 A0 一致".
#
# Usage: python scripts/cgacs/compare_loss_logs.py LOG_A LOG_B

import re
import sys
from collections import defaultdict

STEP_RE = re.compile(
    r'Epoch: \[(\d+)\] \[(\d+)/\d+\] learning_rate: ([\d.eE+-]+)'
    r'(?:\s+(\w+): ([\d.eE+-]+))*')

# tools/train.py format: '... loss: 4.683079 loss_cls: ...'
TOOLS_RE = re.compile(
    r'Epoch: \[(\d+)\] \[ *(\d+)/\d+\] learning_rate: ([\d.eE+-]+)(.*)$')
KV_RE = re.compile(r'(\w+): ([\d.eE+-]+)')

# my-script format: '[epoch 0 step 10/114] lr=... det:loss=.. | loss_cls=.. | ...'
MINE_RE = re.compile(
    r'\[epoch (\d+) step (\d+)/\d+\] lr=([\d.eE+-]+) det:(.*?)\s*\| det_total=([\d.eE+-]+)(.*?)(?: \| finite=|\]|$)'
)


def parse_tools(path):
    """Return {(epoch, step): {key: value}} from tools/train.py log."""
    out = {}
    with open(path) as f:
        for line in f:
            m = TOOLS_RE.search(line)
            if not m:
                continue
            ep, st = int(m.group(1)), int(m.group(2))
            rec = {'learning_rate': float(m.group(3))}
            for k, v in KV_RE.findall(m.group(4)):
                rec[k] = float(v)
            out[(ep, st)] = rec
    return out


def parse_mine(path):
    """Return {(epoch, step): {key: value}} from train_cgacs.py log."""
    out = {}
    with open(path) as f:
        for line in f:
            m = MINE_RE.search(line)
            if not m:
                continue
            ep, st = int(m.group(1)), int(m.group(2))
            rec = {'learning_rate': float(m.group(3))}
            det_items = m.group(4)
            for seg in det_items.split('|'):
                seg = seg.strip()
                if not seg or '=' not in seg:
                    continue
                k, v = seg.split('=', 1)
                rec[k.strip()] = float(v.strip())
            # loss key alias: det_total is the same as 'loss'
            rec['loss'] = float(m.group(5))
            cls_part = m.group(6)
            if cls_part and 'loss_aux_cls' in cls_part:
                for seg in cls_part.split('|'):
                    seg = seg.strip()
                    if seg.startswith('loss_aux_cls='):
                        rec['loss_aux_cls'] = float(seg.split('=', 1)[1])
            out[(ep, st)] = rec
    return out


def compare(path_a, path_b, fmt_a, fmt_b):
    if fmt_a == 'tools':
        A = parse_tools(path_a)
    else:
        A = parse_mine(path_a)
    if fmt_b == 'tools':
        B = parse_tools(path_b)
    else:
        B = parse_mine(path_b)

    keys = ['loss', 'loss_cls', 'loss_iou', 'loss_dfl', 'loss_l1']
    common = sorted(set(A) & set(B))
    if not common:
        print('NO COMMON STEPS between logs')
        print('  A steps:', len(A), ' B steps:', len(B))
        return 1

    maxdiff = {k: 0.0 for k in keys}
    worst = {k: None for k in keys}
    ndiff = {k: 0 for k in keys}
    for (ep, st) in common:
        a, b = A[(ep, st)], B[(ep, st)]
        for k in keys:
            if k in a and k in b:
                d = abs(a[k] - b[k])
                if d > maxdiff[k]:
                    maxdiff[k] = d
                    worst[k] = (ep, st, a[k], b[k])
                if d > 1e-6:
                    ndiff[k] += 1

    print('common steps compared: {}  (A={}, B={})'.format(len(common), len(A), len(B)))
    print('{:<10} {:>14} {:>10} {:>10} {:>10}'.format('key', 'max|diff|', 'n_diff>1e-6', 'A@worst', 'B@worst'))
    ok = True
    for k in keys:
        tag = ''
        if worst[k]:
            tag = ' worst=({})'.format(worst[k])
        print('{:<10} {:>14.9f} {:>10} {:>10.6f} {:>10.6f} {}'.format(
            k, maxdiff[k], ndiff[k],
            worst[k][2] if worst[k] else float('nan'),
            worst[k][3] if worst[k] else float('nan'), tag))
        if maxdiff[k] > 1e-6:
            ok = False
    return 0 if ok else 1


if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('usage: compare_loss_logs.py A B fmtA fmtB  (fmt in tools|mine)')
        sys.exit(2)
    rc = compare(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    sys.exit(rc)
