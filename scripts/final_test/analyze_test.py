# -*- coding: utf-8 -*-
"""最终 TEST 分析 (一次): 用与 analyze_val.py 完全相同的推理/匹配/分析逻辑,
仅将 EvalDataset 指向 test.json (image_dir=images/test, anno_path=annotations/test.json),
在 test.json 上计算 P/R/F1、per-class、混淆矩阵、密集场景 Recall 与主要错误案例。

产出:
  --out test_analyze.json     (完整分析)
  --out_err test_errors.json  (每图遗漏统计 + 最差图像错误案例)
用法:
  python scripts/final_test/analyze_test.py --cfg <M cfg> --weights <M best> \
    --val_anno <test.json abs> --out <abs.json> --out_err <abs.json>
"""
from __future__ import print_function
import os, sys, json, argparse
import numpy as np
import paddle

_PROJ = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, os.path.join(_PROJ, 'PaddleDetection'))
os.chdir(os.path.join(_PROJ, 'PaddleDetection'))

from ppdet.core.workspace import load_config, create
from ppdet.utils.checkpoint import load_weight
sys.path.insert(0, os.path.join(_PROJ, 'scripts'))
import analyze_val as AV  # 复用 analyze()/load_gt()/box_iou()/CLASSES


def run_inference_test(cfg_file, weights, conf_thr=0.05):
    cfg = load_config(cfg_file)
    # 仅重定向评估数据集 -> test.json (与分析脚本其余逻辑完全一致)
    cfg['EvalDataset']['image_dir'] = 'images/test'
    cfg['EvalDataset']['anno_path'] = 'annotations/test.json'
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
            valid = (dt[:, 0] >= 0) & (dt[:, 1] >= conf_thr)
            dt = dt[valid]
            preds[img_id] = dt[:, [2, 3, 4, 5, 1, 0]] if len(dt) else np.zeros((0, 6))
    return preds


def error_cases(gt, preds, top=8):
    """每图遗漏统计 + 最差图像错误案例 (主要错误案例)。"""
    per_img = []
    for img_id, g in gt.items():
        gb = np.array(g['boxes']) if g['boxes'] else np.zeros((0, 5))
        pb = preds.get(img_id, np.zeros((0, 6)))
        n_gt, n_pred = len(gb), len(pb)
        matched = set()
        if n_gt and n_pred:
            iou = AV.box_iou(pb[:, :4], gb[:, :4])
            order = np.argsort(-pb[:, 4])
            for pi in order:
                gis = [x for x in np.where(iou[pi] >= AV.IOU_THR)[0] if x not in matched]
                if not gis:
                    continue
                gi = int(gis[np.argmax(iou[pi, gis])])
                matched.add(gi)
        missed_cls = {}
        for i in range(n_gt):
            if i not in matched:
                c = int(gb[i, 4])
                missed_cls[AV.CLASSES[c]] = missed_cls.get(AV.CLASSES[c], 0) + 1
        per_img.append({
            'img_id': img_id, 'gt': n_gt, 'pred': n_pred, 'matched': len(matched),
            'missed': n_gt - len(matched), 'missed_cls': missed_cls,
        })
    per_img.sort(key=lambda x: -x['missed'])
    return per_img[:top], per_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--val_anno', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--out_err', default=None)
    ap.add_argument('--conf_thr', type=float, default=0.5)
    args = ap.parse_args()

    gt = AV.load_gt(args.val_anno)
    preds = run_inference_test(args.cfg, args.weights, conf_thr=args.conf_thr)
    result = AV.analyze(gt, preds)
    with open(args.out, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print('test_analyze saved:', args.out)
    print('n_images=%d gt=%d pred=%d P=%.3f R=%.3f F1=%.3f dense_rec=%.3f' %
          (result['n_images'], result['gt_boxes_total'], result['pred_boxes_total'],
           result['precision'], result['recall'], result['f1'],
           result['dense_scene']['recall']))
    if args.out_err:
        top, all_img = error_cases(gt, preds)
        json.dump({'top_worst': top, 'per_image': all_img},
                  open(args.out_err, 'w'), indent=2, ensure_ascii=False, default=str)
        print('test_errors saved:', args.out_err)
        print('最差图像:')
        for e in top:
            print('  img_id=%s gt=%d matched=%d missed=%d %s' %
                  (e['img_id'], e['gt'], e['matched'], e['missed'], e['missed_cls']))


if __name__ == '__main__':
    main()
