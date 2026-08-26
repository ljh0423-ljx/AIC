# M 实验 Smoke Test 门控记录 (2 epoch, bs=16/lr=0.002)

日期: 2026-08-17
配置: configs/ppyoloe_plus_crn_m_100e_agrivision_smoke.yml
命令: 与 A0 完全同构 (bs=16, lr=0.002, --eval)
数据集: exiffix 13类 (train915/val113), TEST 未访问

## 门控项检查

| # | 门控项 | 结果 | 证据 |
|---|--------|------|------|
| 1 | GPU 显存峰值 | **19.6 GB / 25.9 GB (76%)** | nvidia-smi 2s 采样 `/tmp/mem_smoke.csv`; 探针(768×768最坏, max_allocated)=15.32GB |
| 2 | loss 正常下降 | **PASS** 4.84 → 3.49 | smoke_train.out 首尾 loss |
| 3 | 无 NaN/Inf/OOM | **0 / 0 / 0** | 精确正则扫描 (INFO 行不算) |
| 4 | checkpoint 正常保存 | **PASS** | 0.pdparams(94MB)+best_model.pdema/pdopt/pdparams |
| 5 | VAL 评估正常 | **PASS** | 113 样本, 每 epoch "Best test bbox ap" 输出 |
| 6 | 推理可运行 | **PASS** | best_model 加载 + 单图前向 → bbox [300,6] |
| 7 | 参数量 | **23,568,416 (23.57M)** | vram_probe.py |
| 8 | 模型大小 | **94.3 MB** (pdparams) | 23.57M×4B |
| 9 | FPS (smoke eval) | **~29-34** | 2×113 样本 eval; 最终以统一协议重测 |

## 结论
全部门控项通过 → 进入正式 100 epoch 训练。
正式配置: configs/ppyoloe_plus_crn_m_100e_agrivision.yml
唯一变量: 模型规模 s→m (depth 0.33→0.67, width 0.50→0.75), bs/lr/epoch/EMA/评估/A0 一致。
