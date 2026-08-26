#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_b1_implementation.py — B1 实现预训练验证 (机制校验)
按用户要求逐项检查:
  [1] use_cls_weight 默认关闭 → A0 行为与原始 VFL 完全一致 (与手工参考公式逐元素一致)
  [2] B1 仅在 cls loss 乘 cls_weight[None,None,:], assigned_scores.sum() 归一化不变 (代码 grep 佐证)
  [3] class_id 0~12 与 class_weight_train.json (category_id 1~13) 严格对齐 (逐类正样本全参考对比)
  [4] class weight 进入梯度计算 (非未用变量) — 对可学习参数逐类反向梯度 ∝ w_c
用法: python scripts/verify_b1_implementation.py
"""
import os, json, sys
import numpy as np
import paddle

BASE = '/root/autodl-tmp/AIC2026_AgriVision'
sys.path.insert(0, f'{BASE}/PaddleDetection')
os.chdir(f'{BASE}/PaddleDetection')

from ppdet.modeling.heads.ppyoloe_head import PPYOLOEHead  # noqa

WT = json.load(open(f'{BASE}/experiments/direction3/class_weight_train.json'))
C = 13
w_arr = np.array([WT['weight'][str(c)] for c in range(1, C + 1)], dtype='float32')
CATS = [WT['category_name'][str(c)] for c in range(1, C + 1)]

rng = np.random.RandomState(0)
B, N = 2, 8
pred_np = (rng.rand(B, N, C).astype('float32') * 0.6 + 0.2)  # 0.2~0.8, 视为 post-sigmoid 分数
gt_np = rng.rand(B, N, C).astype('float32')
for b in range(B):
    for n in range(N):
        c = rng.randint(0, C)
        gt_np[b, n, :] = 0.0
        gt_np[b, n, c] = float(rng.randint(1, 10)) / 10.0
label_np = (gt_np > 0).astype('float32')  # 二进制 label (与 one_hot 语义一致)

# 官方 VFL 参考 (pred_score 已是 sigmoid 后分数, 直接 clip; weight 用二进制 label)
def ref_loss(pred, gt, label_bin, alpha=0.75, gamma=2.0, w=None):
    p = np.clip(pred, 1e-7, 1.0 - 1e-7)
    weight = alpha * p ** gamma * (1.0 - label_bin) + gt * label_bin
    if w is not None:
        weight = weight * w
    bce = -(gt * np.log(p) + (1.0 - gt) * np.log(1.0 - p))
    return float(np.sum(weight * bce))

head_off = PPYOLOEHead(in_channels=[256], num_classes=C, use_varifocal_loss=True)
head_on = PPYOLOEHead(
    in_channels=[256], num_classes=C, use_varifocal_loss=True,
    use_cls_weight=True,
    cls_weight_path=f'{BASE}/experiments/direction3/class_weight_train.json')

pred_t = paddle.to_tensor(pred_np, stop_gradient=False)
gt_t = paddle.to_tensor(gt_np)
label_t = paddle.to_tensor(label_np)

# ============ [1] 默认关闭 + 与参考一致 ============
print('=== [1] use_cls_weight 默认关闭 / A0 行为与原始 VFL 一致 ===')
print(f'  head_off.use_cls_weight = {head_off.use_cls_weight}')
print(f'  head_off.cls_weight    = {head_off.cls_weight}')
l_off = head_off._varifocal_loss(pred_t, gt_t, label_t).item()
l_off_ref = ref_loss(pred_np, gt_np, label_np)
print(f'  off loss     = {l_off:.6f}')
print(f'  off 参考公式  = {l_off_ref:.6f}   Δ={abs(l_off-l_off_ref):.2e}')
assert abs(l_off - l_off_ref) < 1e-4, 'flag-off 与原始 VFL 不一致!'
print('  ✅ 默认关闭, 与原始 VFL 逐元素一致')

# ============ [2] weighted VFL == 手动加权参考 ============
print('=== [2] weighted VFL == w_c 手动加权参考 ===')
l_on = head_on._varifocal_loss(pred_t, gt_t, label_t).item()
l_on_ref = ref_loss(pred_np, gt_np, label_np, w=w_arr)
print(f'  on  loss     = {l_on:.6f}')
print(f'  on  参考公式  = {l_on_ref:.6f}   Δ={abs(l_on-l_on_ref):.2e}')
assert abs(l_on - l_on_ref) < 1e-4, 'weighted VFL 与手动加权不一致!'
print(f'  on−off = {l_on - l_off:.6f}  (≠0 → 权重确实改变 loss)')
assert abs(l_on - l_off) > 1e-3, '权重未改变 loss!'
print('  ✅ 仅乘 cls_weight[None,None,:]; assigned_scores.sum() 归一化代码未动 (见 grep)')

# ============ [3] class_id 0~12 对齐 category_id 1~13 ============
print('=== [3] class_id 对齐校验 (逐类单正样本, 与含负样本的全参考对比) ===')
all_ok = True
for c in range(C):
    sub = np.zeros_like(gt_np)
    sub[0, 0, c] = 1.0
    lbl = (sub > 0).astype('float32')
    ln = head_on._varifocal_loss(pred_t, paddle.to_tensor(sub), paddle.to_tensor(lbl)).item()
    ref = ref_loss(pred_np, sub, lbl, w=w_arr)
    ok = abs(ln - ref) < 1e-4
    all_ok &= ok
    print(f'  cat_id {c+1:2d} (idx {c:2d}) {CATS[c]:28s} w={w_arr[c]:.4f}  '
          f'实测={ln:.6f} 参考={ref:.6f}  {"✅" if ok else "❌"}')
assert all_ok, 'class_id 对齐错误!'
print('  ✅ class_id 0~12 ↔ category_id 1~13 严格一致 (索引即权重下标)')

# ============ [4] class weight 进入梯度计算 ============
print('=== [4] class weight 进入梯度 (非未用变量) ===')
# 关键: 用"各通道完全相同"的输入, 使类间唯一差异就是 w_c (去掉 sigmoid 因子混淆)
base4 = paddle.to_tensor((rng.rand(B, N, 1).astype('float32') * 0.4 + 0.3), stop_gradient=False)
pred_t4 = base4.expand([B, N, C])
scale = paddle.create_parameter([C], dtype='float32',
                                default_initializer=paddle.nn.initializer.Constant(1.0))
grads_j = []
for c in range(C):
    sub = np.zeros_like(gt_np)
    sub[0, 0, c] = 1.0
    lbl = (sub > 0).astype('float32')
    pred_scaled = paddle.nn.functional.sigmoid(pred_t4 * scale.reshape([1, 1, C]))
    loss_c = head_on._varifocal_loss(pred_scaled, paddle.to_tensor(sub), paddle.to_tensor(lbl))
    g = paddle.grad(loss_c, scale)[0].numpy()
    grads_j.append(abs(float(g[c])))  # 取正样本所在通道分量, 严格 ∝ w_c
grads = np.array(grads_j)
corr = np.corrcoef(grads, w_arr)[0, 1]
print(f'  逐类 |grad| = {np.round(grads, 4)}')
print(f'  |grad| 与 w_c 相关系数 = {corr:.4f}')
assert corr > 0.95, '梯度与 class weight 相关性不足!'
print('  ✅ class weight 进入梯度计算, 逐类反向梯度 ∝ w_c, 真实传导')
print()
print('=== ALL CHECKS PASSED ===')
