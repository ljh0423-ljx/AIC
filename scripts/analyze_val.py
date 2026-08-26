# -*- coding: utf-8 -*-
"""VAL 全面分析脚本 (S1 vs A0 统一口径)。

对给定 checkpoint 在 VAL(113 张)上推理, 计算:
  - P / R / F1 @ IoU=0.5 (类感知匹配, conf>=0.05)
  - per-class Recall / Precision
  - 混淆矩阵: 已匹配检测的 (pred_cls -> gt_cls); GT 未命中 = missed; 检测无 GT = fp
  - 主混淆类对 (pred!=gt 的 top 对) + per-class FN 统计
  - 密集场景 Recall (GT>=10 的图像子集, 与 BASELINE_DIAGNOSTIC 的 img92/img7 定义一致)
用法:
  python scripts/analyze_val.py --cfg <yaml> --weights <ckpt_prefix> --val_anno <val.json> --out <out.json>
"""
import os, sys, json, argparse
import numpy as np
import paddle

_PROJ = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(_PROJ, 'PaddleDetection'))
os.chdir(os.path.join(_PROJ, 'PaddleDetection'))

from ppdet.core.workspace import load_config, create
from ppdet.utils.checkpoint import load_weight

CLASSES = [
    'Tomato Early blight leaf', 'Tomato Septoria leaf spot', 'Tomato leaf',
    'Tomato leaf bacterial spot', 'Tomato leaf late blight',
    'Tomato leaf mosaic virus', 'Tomato leaf yellow virus', 'Tomato mold leaf',
    'Apple Scab Leaf', 'Apple leaf', 'Apple rust leaf', 'grape leaf',
    'grape leaf black rot'
]
DENSE_MIN_GT = 10  # 与 BASELINE_DIAGNOSTIC 密集场景定义一致 (img92=19GT, img7=12GT)
IOU_THR = 0.5


def box_iou(a, b):
    """a:(Na,4)xyxy, b:(Nb,4)xyxy -> (Na,Nb) IoU"""
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    xx1 = np.maximum(ax1[:, None], bx1[None, :])
    yy1 = np.maximum(ay1[:, None], by1[None, :])
    xx2 = np.minimum(ax2[:, None], bx2[None, :])
    yy2 = np.minimum(ay2[:, None], by2[None, :])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h
    aarea = (ax2 - ax1) * (ay2 - ay1)
    barea = (bx2 - bx1) * (by2 - by1)
    return inter / (aarea[:, None] + barea[None, :] - inter + 1e-12)


def load_gt(val_anno):
    with open(val_anno) as f:
        ann = json.load(f)
    imgs = {}
    for im in ann['images']:
        imgs[im['id']] = {'w': im['width'], 'h': im['height'], 'boxes': []}
    for a in ann['annotations']:
        x, y, w, h = a['bbox']
        imgs[a['image_id']]['boxes'].append(
            [x, y, x + w, y + h, a['category_id'] - 1])  # coco 1-based -> 0-based
    return imgs


def run_inference(cfg_file, weights, conf_thr=0.05):
    cfg = load_config(cfg_file)
    model = create(cfg.architecture)
    load_weight(model, weights)
    model.eval()
    eval_dataset = create('EvalDataset')()
    bs_sampler = paddle.io.BatchSampler(eval_dataset, batch_size=1)
    eval_loader = create('EvalReader')(eval_dataset, cfg.worker_num,
                                       batch_sampler=bs_sampler)
    preds = {}
    with paddle.no_grad():
        for data in eval_loader:
            for k, v in data.items():
                if isinstance(v, paddle.Tensor):
                    data[k] = v.cuda()
            outs = model(data)
            iid = data['im_id']
            img_id = int(np.asarray(iid).reshape(-1)[0])
            if outs.get('bbox') is None or len(outs['bbox']) == 0:
                preds[img_id] = np.zeros((0, 6))
                continue
            bbox = outs['bbox'].numpy()
            bbox_num = outs['bbox_num'].numpy() if 'bbox_num' in outs else None
            n = int(bbox_num[0]) if bbox_num is not None else len(bbox)
            dt = bbox[:n]
            # 列序 [cls_id, score, x1, y1, x2, y2]; 填充行 num_id=-1
            valid = (dt[:, 0] >= 0) & (dt[:, 1] >= conf_thr)
            dt = dt[valid]
            if len(dt):
                # 重排为 [x1, y1, x2, y2, score, cls]
                preds[img_id] = dt[:, [2, 3, 4, 5, 1, 0]]
            else:
                preds[img_id] = np.zeros((0, 6))
    return preds


def analyze(gt, preds):
    """IoU0.5 贪心匹配 (不要求同类), 混淆矩阵含跨类项。

    TP 计数按同类匹配: 已配对且 pred.cls==gt.cls 才算 TP;
    跨类配对 (检到但分错类) 在 COCO 口径下同时计为该 GT 的 FN 与该 pred 的 FP,
    但会进入混淆矩阵 off-diagonal, 用于分析"分错类"问题。
    """
    conf_matrix = np.zeros((13, 13))  # [gt_cls, pred_cls]
    fp_by_pred = np.zeros(13)  # 未命中任何 GT 的检测
    n_imgs = len(gt)
    gt_boxes_total = 0
    dense_imgs = 0
    dense_gt = 0
    dense_match = 0
    per_img_rec = []
    for img_id, g in gt.items():
        gb = np.array(g['boxes']) if g['boxes'] else np.zeros((0, 5))
        gt_boxes_total += len(gb)
        pb = preds.get(img_id, np.zeros((0, 6)))
        n_gt = len(gb)
        if n_gt >= DENSE_MIN_GT:
            dense_imgs += 1
            dense_gt += n_gt
        matched_gt = set()
        matched_pred = set()
        tp_img = 0
        if len(gb) and len(pb):
            iou = box_iou(pb[:, :4], gb[:, :4])  # (P,G)
            order = np.argsort(-pb[:, 4])
            for pi in order:
                if pi in matched_pred:
                    continue
                gis = np.where(iou[pi] >= IOU_THR)[0]
                gis = [g for g in gis if g not in matched_gt]
                if not gis:
                    continue
                gi = gis[int(np.argmax(iou[pi, gis]))]
                matched_gt.add(gi)
                matched_pred.add(pi)
                gc, pc = int(gb[gi, 4]), int(pb[pi, 5])
                conf_matrix[gc, pc] += 1
                if gc == pc:
                    tp_img += 1
        for pi in range(len(pb)):
            if pi not in matched_pred:
                fp_by_pred[int(pb[pi, 5])] += 1
        if n_gt >= DENSE_MIN_GT:
            dense_match += tp_img
        # recall 按整图 (检到任意框即算)
        per_img_rec.append(len(matched_gt) / n_gt if n_gt else 1.0)
    # 重新按同类口径统计 tp/fp/fn (跨类配对计 FP+FN)
    tp = int(sum(conf_matrix[c, c] for c in range(13)))
    pred_total = int(sum(len(p) for p in preds.values()))
    fp = pred_total - tp
    fn = gt_boxes_total - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.
    rec = tp / (tp + fn) if (tp + fn) else 0.
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.
    # 每类
    gt_by_cls = np.bincount(
        [int(b[4]) for g in gt.values() for b in g['boxes']],
        minlength=13).astype(float)
    pred_by_cls = np.zeros(13)
    for p in preds.values():
        if len(p):
            pred_by_cls += np.bincount(p[:, 5].astype(int), minlength=13)[:13]
    per_class = {}
    for c in range(13):
        per_class[c] = {
            'gt': int(gt_by_cls[c]),
            'pred': int(pred_by_cls[c]),
            'tp': int(conf_matrix[c, c]),
            'recall': float(conf_matrix[c, c] / gt_by_cls[c]) if gt_by_cls[c] else float('nan'),
            'precision': float(conf_matrix[c, c] / pred_by_cls[c]) if pred_by_cls[c] else float('nan'),
            'missed': int(gt_by_cls[c] - conf_matrix[c, c]),
            'fp': int(pred_by_cls[c] - conf_matrix[c, c]),
        }
    # 主混淆对 (pred != gt, 已配对)
    offdiag = []
    for gi in range(13):
        for pi in range(13):
            if gi != pi and conf_matrix[gi, pi] > 0:
                offdiag.append(
                    (CLASSES[gi], CLASSES[pi], int(conf_matrix[gi, pi])))
    offdiag.sort(key=lambda x: -x[2])
    return {
        'n_images': n_imgs,
        'gt_boxes_total': int(gt_boxes_total),
        'pred_boxes_total': pred_total,
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': prec, 'recall': rec, 'f1': f1,
        'per_class': per_class,
        'top_confusion_pairs': offdiag[:8],
        'dense_scene': {
            'min_gt': DENSE_MIN_GT,
            'n_images': dense_imgs,
            'gt_total': int(dense_gt),
            'matched': int(dense_match),
            'recall': float(dense_match / dense_gt) if dense_gt else float('nan'),
        },
        'per_image_recall_mean': float(np.mean(per_img_rec)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--val_anno', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--conf_thr', type=float, default=0.05)
    args = ap.parse_args()
    cfg_file = args.cfg if os.path.isabs(args.cfg) else os.path.join(_PROJ, args.cfg)
    weights = args.weights if os.path.isabs(args.weights) else os.path.join(_PROJ, args.weights)
    gt = load_gt(args.val_anno)
    preds = run_inference(cfg_file, weights, conf_thr=args.conf_thr)
    res = analyze(gt, preds)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
