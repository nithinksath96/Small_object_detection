# Run detection on video datasets in COCO-format

import argparse, json, pickle
from os.path import join, isfile
from time import perf_counter

from tqdm import tqdm
import numpy as np

import torch

from pycocotools.coco import COCO

import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')



from argo_data_scripts.util import mkdir2, print_stats

from argo_data_scripts.det.det_apis import init_detector, inference_detector
from argo_data_scripts.det import imread, parse_det_result, vis_det, eval_ccf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--annot-path', type=str, required=True)
    parser.add_argument('--in-scale', type=float, default=None)
    parser.add_argument('--no-mask', action='store_true', default=False)
    parser.add_argument('--no-class-mapping', action='store_true', default=False)
    parser.add_argument('--cpu-pre', action='store_true', default=False)
    parser.add_argument('--center-crop', action='store_true', default=False)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--vis-dir', type=str, default=None)
    parser.add_argument('--vis-scale', type=float, default=1)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--no-eval', action='store_true', default=False)
    parser.add_argument('--eval-mask', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False)

    opts = parser.parse_args()
    return opts

def main():
    assert torch.cuda.device_count() == 1 # mmdet only supports single GPU testing

    opts = parse_args()

    mkdir2(opts.out_dir)
    vis_out = bool(opts.vis_dir)
    if vis_out:
        mkdir2(opts.vis_dir)

    db = COCO(opts.annot_path)
    class_names = [c['name'] for c in db.dataset['categories']]
    n_class = len(class_names)
    coco_mapping = None if opts.no_class_mapping else db.dataset.get('coco_mapping', None)
    if coco_mapping is not None:
        coco_mapping = np.asarray(coco_mapping)
    seqs = db.dataset['sequences']
    seq_dirs = db.dataset['seq_dirs']

    # import pdb
    # pdb.set_trace()
    model = init_detector(opts)

    # warm up the GPU
    img = db.imgs[0]
    w_img, h_img = img['width'], img['height']
    _ = inference_detector(model, np.zeros((h_img, w_img, 3), np.uint8))
    torch.cuda.synchronize()

    results_raw = []        # image based, all 80 COCO classes
    results_ccf = []        # instance based
    runtime_all = []

    t_start = perf_counter()
    for iid, img in tqdm(db.imgs.items()):
        img_name = img['name']

        sid = img['sid']
        seq_name = seqs[sid]

        img_path = join(opts.data_root, seq_dirs[sid], img_name)
        I = imread(img_path)
        
        t1 = perf_counter()
        result = inference_detector(model, I, gpu_pre=not opts.cpu_pre)
        results_raw.append(result)
        bboxes, scores, labels, masks = \
            parse_det_result(result, coco_mapping, n_class)

        torch.cuda.synchronize()
        t2 = perf_counter()
        runtime_all.append(t2 - t1)

        if vis_out:
            vis_path = join(opts.vis_dir, seq_name, img_name[:-3] + 'jpg')
            if opts.overwrite or not isfile(vis_path):
                vis_det(
                    I, bboxes, labels,
                    class_names, masks, scores,
                    out_scale=opts.vis_scale,
                    out_file=vis_path
                )

        # convert to coco fmt
        n = len(bboxes)
        if n:
            bboxes[:, 2:] -= bboxes[:, :2]

        for i in range(n):
            result_dict = {
                'image_id': iid,
                'bbox': bboxes[i],
                'score': scores[i],
                'category_id': labels[i],
            }
            if masks is not None:
                result_dict['segmentation'] = masks[i]
            results_ccf.append(result_dict)
    t_end = perf_counter()
    t_total = t_end - t_start

    out_path = join(opts.out_dir, 'results_raw.pkl')
    if opts.overwrite or not isfile(out_path):
        pickle.dump(results_raw, open(out_path, 'wb'))

    out_path = join(opts.out_dir, 'results_ccf.pkl')
    if opts.overwrite or not isfile(out_path):
        pickle.dump(results_ccf, open(out_path, 'wb'))

    n_total = len(db.imgs)
    out_path = join(opts.out_dir, 'time_info.pkl')
    if opts.overwrite or not isfile(out_path):
        pickle.dump({
            'runtime_all': runtime_all,
            'n_total': n_total,
            't_total': t_total,
        }, open(out_path, 'wb'))

    # convert to ms for display
    s2ms = lambda x: 1e3*x

    print(f'{n_total} frames processed in {t_total:.4g}s')
    print(f'Frame rate: {n_total/t_total:.4g} (including I/O)')
    print_stats(runtime_all, 'Algorithm Runtime (ms)', cvt=s2ms)


    if not opts.no_eval:
        eval_summary = eval_ccf(db, results_ccf)
        out_path = join(opts.out_dir, 'eval_summary.pkl')
        if opts.overwrite or not isfile(out_path):
            pickle.dump(eval_summary, open(out_path, 'wb'))
        if opts.eval_mask:
            print('Evaluating instance segmentation')
            eval_summary = eval_ccf(db, results_ccf, iou_type='segm')
            out_path = join(opts.out_dir, 'eval_summary_mask.pkl')
            if opts.overwrite or not isfile(out_path):
                pickle.dump(eval_summary, open(out_path, 'wb'))

    if vis_out:
        print(f'python vis/make_videos.py "{opts.vis_dir}"')

if __name__ == '__main__':
    main()