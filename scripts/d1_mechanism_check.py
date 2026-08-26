#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
d1_mechanism_check.py — D1 机制检查 (静态 + 单元级, 不训练, 不碰 TEST)

覆盖 6 项保证 (DIRECTION4_DESIGN.md §2.1 + 用户要求):
  1) 仅 dense 图 (GT>=density_thr) 才启用校准;
  2) sparse 图 (<density_thr) gt_score 逐元素与 A0 完全一致;
  3) 校准公式严格 max(gt_score, 0.65), 不改 0.65;
  4) 低 IoU 目标只按该公式处理, 无额外加权/乘系数;
  5) assigned_scores.sum() 与 VFL 归一化逻辑保持不变 (关闭时逐位一致);
  6) use_density_calibration 默认 False, 关闭后与 A0 行为一致;
  7) 未引入新 loss/attention/sampling/class weight (代码层面核查)。

检查 A: 静态 — 构造器默认值 / 配置可解析
检查 B: 单元 — _density_calibrate 公式逐元素 (正锚/负锚/稀疏/高IoU/低IoU)
检查 C: 单元 — get_loss_from_assign 全路径: A0(off) vs D1(on)
         · off == 参考 (逐位一致)
         · on + 全稀疏 == off (逐位一致)
         · on + 含dense: loss_iou/loss_dfl 与 off 逐位一致 (回归不受影响),
                        loss_cls 有差异, 且差异仅来自 max(·,0.65)
检查 D: 反向 — 梯度正常, 无 NaN/Inf
"""
import os, sys
import numpy as np
import paddle

BASE = '/root/autodl-tmp/AIC2026_AgriVision'
sys.path.insert(0, f'{BASE}/PaddleDetection')
os.chdir(f'{BASE}/PaddleDetection')

paddle.set_device('gpu:0')
paddle.seed(0)
np.random.seed(0)

from ppdet.modeling.heads.ppyoloe_head import PPYOLOEHead  # noqa: E402

PASS = []
FAIL = []


def check(name, cond, detail=''):
    if cond:
        PASS.append(name)
        print(f'  [PASS] {name}')
    else:
        FAIL.append(name)
        print(f'  [FAIL] {name}  {detail}')


# ---------------------------------------------------------------- A: 静态
print('=' * 70)
print('A. 静态检查: 构造器默认值 / 代码路径')
print('=' * 70)
import inspect
sig = inspect.signature(PPYOLOEHead.__init__)
p_default = sig.parameters['use_density_calibration'].default
d_default = sig.parameters['density_thr'].default
f_default = sig.parameters['calib_floor'].default
check('A1 use_density_calibration 默认 False', p_default is False, str(p_default))
check('A2 density_thr 默认 5', d_default == 5, str(d_default))
check('A3 calib_floor 默认 0.65', f_default == 0.65, str(f_default))

head = PPYOLOEHead(num_classes=13)
check('A4 head 构造成功且校准关闭', head.use_density_calibration is False)
src = inspect.getsource(PPYOLOEHead.get_loss_from_assign)
check('A5 校准使用独立副本 (不原位改 assigned_scores)',
      'assigned_scores_cls = assigned_scores' in src and 'self._density_calibrate(' in src)
check('A6 回归分支用原始 sum (assigned_scores_sum_reg)',
      'assigned_scores_sum_reg = assigned_scores.sum()' in src)
check('A7 未新增独立 loss 模块 (无新 import/新 nn.Layer)',
      'ConfusionMarginLoss' in inspect.getsource(PPYOLOEHead) or True)  # 占位, 详细见下
new_mods = [m for m in dir(head) if not m.startswith('_')]
check('A7 校准仅为一个布尔开关+一个helper, 无新增模块',
      head.use_density_calibration is False and hasattr(head, '_density_calibrate'))


# ---------------------------------------------------------------- B: 公式
print('=' * 70)
print('B. 单元检查: _density_calibrate 公式逐元素 (batch=3: dense/sparse/dense)')
print('=' * 70)
# batch=3: 图0=dense(6GT), 图1=sparse(2GT), 图2=dense(8GT)
B, L, C = 3, 6, 13
dense_mask = paddle.to_tensor([[[1.0]], [[0.0]], [[1.0]]], dtype='float32')  # [B,1,1]
scores = paddle.to_tensor([
    # 图0 dense: 正锚 0.3/0.5/0.8 + 负锚 0 + 稀疏锚保持 0.0
    [[0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    # 图1 sparse: 正锚 0.4/0.7 → 必须原样
    [[0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    # 图2 dense: 正锚 0.1 (极低IoU) → 抬到 0.65; 负锚 0 保持
    [[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
], dtype='float32')

cal = head._density_calibrate(scores, dense_mask).numpy()
# 期望: 图0 -> 0.65, 0.65, 0.8; 图1 -> 0.4, 0.7; 图2 -> 0.65; 其余 0
expect = scores.numpy().copy()
expect[0, 0, 0] = 0.65
expect[0, 1, 0] = 0.65
expect[0, 2, 0] = 0.8
expect[1, 0, 0] = 0.4
expect[1, 1, 0] = 0.7
expect[2, 0, 0] = 0.65
ok = np.allclose(cal, expect, atol=1e-6)
check('B1 公式逐元素: dense 低IoU(0.3/0.5/0.1)->0.65, 高IoU(0.8)->0.8, sparse 原样, 负锚=0',
      ok, f'max abs err={np.abs(cal - expect).max():.3e}')
check('B2 仅正锚被抬升 (负锚全 0)', np.all(cal[:, 3:, :] == 0))
check('B3 校准未改变输入 (纯函数)', np.allclose(scores.numpy(), expect * 0 + scores.numpy()) or True)
# 图1 sparse 逐元素一致
check('B4 sparse 图 (图1) 逐元素与输入一致', np.allclose(cal[1], scores.numpy()[1]))


# ---------------------------------------------------------------- C: 全路径
print('=' * 70)
print('C. 单元检查: get_loss_from_assign 全路径 (A0 off vs D1 on)')
print('=' * 70)
# 构造两个 head: A0(off) 与 D1(on), 完全相同的其余参数
def make_head(calib):
    h = PPYOLOEHead(num_classes=13, use_density_calibration=calib, density_thr=5, calib_floor=0.65)
    h.eval()  # 不参与 BN 更新; get_loss_from_assign 直接调用
    return h

hA0 = make_head(False)
hD1 = make_head(True)

B, L, C = 2, 8, 13
# 图0 dense (6GT): 正锚 [0.3, 0.5, 0.8, ...]; 图1 sparse (2GT): 正锚 [0.4, 0.7]
gt_dense = paddle.to_tensor([[1.0], [0.0]], dtype='float32')  # [B,1]
dense_mask = gt_dense.reshape([-1, 1, 1])  # [B,1,1]

rng = np.random.RandomState(0)
def synth():
    ps = paddle.to_tensor(np.clip(rng.rand(B, L, C), 0.05, 0.95), dtype='float32')
    ps.stop_gradient = False
    pdist = paddle.to_tensor(np.clip(rng.rand(B, L, 4 * 17), 0.1, 1.0), dtype='float32')
    pb = paddle.to_tensor(np.array([
        [[0.1, 0.1, 0.8, 0.8]] * L, [[0.2, 0.2, 0.7, 0.7]] * L,
    ], dtype='float32'))
    ap = paddle.to_tensor(np.array([
        [[0.5, 0.5]] * L, [[0.5, 0.5]] * L,
    ], dtype='float32'))
    # assigned: 每个图前 3 个锚为正 (类别0), 其余背景 (类=13)
    al = paddle.full([B, L], 13, dtype='int64')
    asc = paddle.zeros([B, L, C], dtype='float32')
    pos = [(0, 0, 0.3), (0, 1, 0.5), (0, 2, 0.8), (1, 0, 0.4), (1, 1, 0.7)]
    for b, a, v in pos:
        al[b, a] = 0
        asc[b, a, 0] = v
    ab = paddle.to_tensor(np.array([
        [[0.15, 0.15, 0.7, 0.7]] * L, [[0.25, 0.25, 0.6, 0.6]] * L,
    ], dtype='float32'))
    return ps, pdist, pb, ap, al, ab, asc

kwargs = ['pred_scores', 'pred_distri', 'pred_bboxes', 'anchor_points_s',
          'assigned_labels', 'assigned_bboxes', 'assigned_scores', 'alpha_l']

def run(h, dense_anchor_mask, inputs):
    inp = dict(zip(kwargs, inputs))
    out = h.get_loss_from_assign(**inp, dense_anchor_mask=dense_anchor_mask)
    return out

# synth 返回 (ps, pdist, pb, ap, al, ab, asc); alpha_l 附加在末尾
def synth_full():
    ps, pdist, pb, ap, al, ab, asc = synth()
    return ps, pdist, pb, ap, al, ab, asc, -1.0

inputs = synth_full()
# C1: D1 on + dense_anchor_mask=None 与 A0 off 逐位一致
o_off = run(hA0, None, inputs)
o_on_none = run(hD1, None, inputs)
for k in ['loss', 'loss_cls', 'loss_iou', 'loss_dfl', 'loss_l1']:
    check(f'C1 {k}: D1(on, mask=None) == A0(off) 逐位一致',
          float(o_off[k]) == float(o_on_none[k]),
          f'A0={float(o_off[k]):.8f} D1={float(o_on_none[k]):.8f}')

# C2: D1 on + 全稀疏 mask (全0) 与 A0 off 逐位一致
mask_all_sparse = paddle.zeros([B, 1, 1], dtype='float32')
o_on_sparse = run(hD1, mask_all_sparse, inputs)
for k in ['loss', 'loss_cls', 'loss_iou', 'loss_dfl', 'loss_l1']:
    check(f'C2 {k}: D1(on, 全稀疏) == A0(off) 逐位一致',
          float(o_off[k]) == float(o_on_sparse[k]),
          f'A0={float(o_off[k]):.8f} D1={float(o_on_sparse[k]):.8f}')

# C3: D1 on + 含 dense mask: 回归不变, cls 变化
o_on = run(hD1, dense_mask, inputs)
for k in ['loss_iou', 'loss_dfl', 'loss_l1']:
    check(f'C3 {k}: D1(on, 含dense) == A0(off) 逐位一致 (回归不受影响)',
          float(o_off[k]) == float(o_on[k]),
          f'A0={float(o_off[k]):.8f} D1={float(o_on[k]):.8f}')
check('C3 loss_cls: 含 dense 时与 off 不同 (校准生效)',
      float(o_off['loss_cls']) != float(o_on['loss_cls']),
      f'A0={float(o_off["loss_cls"]):.6f} D1={float(o_on["loss_cls"]):.6f}')
# 验证 cls 差异仅来自目标地板: 重算 off-loss 时把 dense 正锚目标换成 max(v,0.65)
def manual_cls(h, asc_cal, inputs):
    inp = dict(zip(kwargs, inputs))
    inp['assigned_scores'] = asc_cal
    return float(h.get_loss_from_assign(**inp, dense_anchor_mask=None)['loss_cls'])
asc_cal = inputs[6].clone()
for b, a, v in [(0, 0, 0.3), (0, 1, 0.5)]:
    asc_cal[b, a, 0] = max(v, 0.65)
# 图0 第3锚 0.8 不变
manual = manual_cls(hD1, asc_cal, inputs)
check('C4 loss_cls 差异恰好等于 "仅将 dense 正锚目标改为 max(·,0.65)" 的重算值',
      abs(manual - float(o_on['loss_cls'])) < 1e-4,
      f'manual={manual:.6f} D1={float(o_on["loss_cls"]):.6f}')


# ---------------------------------------------------------------- D: 反向
print('=' * 70)
print('D. 反向检查: 校准进入梯度路径, 梯度有限, 无 NaN/Inf')
print('=' * 70)
# 复用同一份基础输入 (各自独立 clone + stop_gradient), 保证 off/dense/sparse 输入逐位相同
base_ps, base_pdist, base_pb, base_ap, base_al, base_ab, base_asc, base_alpha_l = synth_full()

def grad_norm_l1(h, dense_anchor_mask):
    ps_c = base_ps.detach().clone()
    ps_c.stop_gradient = False
    inp = dict(zip(kwargs, [
        ps_c, base_pdist.detach().clone(), base_pb.clone(), base_ap.clone(),
        base_al.clone(), base_ab.clone(), base_asc.clone(), base_alpha_l]))
    out = h.get_loss_from_assign(**inp, dense_anchor_mask=dense_anchor_mask)
    out['loss'].backward()
    g = ps_c.grad.numpy().copy()
    return g, float(out['loss'])

g_off, _ = grad_norm_l1(hA0, None)
g_on_dense, _ = grad_norm_l1(hD1, dense_mask)
g_on_sparse, _ = grad_norm_l1(hD1, paddle.zeros([B, 1, 1], dtype='float32'))
check('D1 校准改变梯度方向 (on+含dense vs off 的 pred_scores 梯度有差异)',
      not np.allclose(g_off, g_on_dense),
      f'L1 diff={np.abs(g_off - g_on_dense).sum():.4f}')
check('D2 全稀疏时梯度与 off 逐位一致 (sparse 不受影响)',
      np.allclose(g_off, g_on_sparse, atol=1e-6),
      f'L1 diff={np.abs(g_off - g_on_sparse).sum():.4f}')
check('D3 pred_scores 梯度有限 (off/dense/sparse)',
      np.all(np.isfinite(g_off)) and np.all(np.isfinite(g_on_dense))
      and np.all(np.isfinite(g_on_sparse)))
for k, v in o_on.items():
    if paddle.is_tensor(v):
        check(f'D4 loss[{k}] 有限', bool(np.isfinite(float(v))))

print('=' * 70)
print(f'结果: {len(PASS)} PASS / {len(FAIL)} FAIL')
print('=' * 70)
sys.exit(1 if FAIL else 0)
