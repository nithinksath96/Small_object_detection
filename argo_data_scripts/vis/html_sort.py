import json, socket, pickle, random
from os.path import join, basename, isfile
from glob import glob

import numpy as np

from html4vision import Col, imagetable

import sys; sys.path.insert(0, '..'); sys.path.insert(0, '.')
from util import mkdir2

data_dir = '/data2/mengtial'
data_root = join(data_dir, 'ArgoVerse1.1/tracking')
split = 'val'
annot_file = join(data_root, 'coco_fmt/' + split + '.json')

out_dir = mkdir2(join(data_dir, 'Exp/ArgoVerse1.1/vis'))
out_name = 'sort.html'
title = 'SORT'
metric = 'AP'
link_video = True
n_show = 90
n_consec = 10
align = True        # align to the stride in each sequence
stride = 30
random.seed(0)

names = [
    'Annotation',
    'Detection',
    'Det + SORT (Hungarian Matching + Kalman Filter)',
    'Det + SORT with less smoothing',
    'Det + Greedy Matching + Kalman Filter',
    'Det + Greedy Matching + Linear Forecasting',
]

dirs = [
    join(data_root, 'coco_fmt/vis/' + split + '_track'),
    join(data_dir, 'Exp/ArgoVerse1.1/vis/srt_mrcnn50_nm_ds_s0.5_2/' + split),
    join(data_dir, 'Exp/ArgoVerse1.1/vis/pps_mrcnn50_nm_ds_s0.5_sort2/' + split),
    join(data_dir, 'Exp/ArgoVerse1.1/vis/pps_sort1_mrcnn50_nm_ds_s0.5/' + split),
    join(data_dir, 'Exp/ArgoVerse1.1/vis/pps_mrcnn50_nm_ds_s0.5_2_bf_kf1/' + split),
    join(data_dir, 'Exp/ArgoVerse1.1/vis/pps_mrcnn50_nm_ds_s0.5_2_fba_iou_lin/' + split),
]

for d in dirs:
    print(f'python vis/make_videos.py "{d}" --fps 30')

srv_dir = data_dir
srv_port = 40001
# host_name = socket.gethostname()
host_name = 'trinity.vision.cs.cmu.edu'

##

db = json.load(open(annot_file))
imgs = db['images']
seqs = db['sequences']


n_img = len(imgs)

if n_consec is None:
    sel = random.choices(list(range(n_img)), k=n_show)
elif align:   
    start_idx = []
    last_sid = None
    for i, img in enumerate(imgs):
        if img['sid'] != last_sid:
            start_idx.append(i)
            last_sid = img['sid']
    start_idx = np.array(start_idx)

    sel = random.choices(list(range(n_img//n_consec)), k=n_show//n_consec)
    sel = np.array(sel)
    sel *= n_consec
    for i in range(len(sel)):
        diff = sel[i] - start_idx
        diff[diff < 0] = n_img
        nearest = np.argmin(diff)
        sel[i] -= (sel[i] - start_idx[nearest]) % stride
    # it is possible to have duplicated sel, but ignore for now
    consecs = np.arange(n_consec)
    sel = [i + consecs for i in sel]
    sel = np.array(sel).flatten().tolist()
else:
    sel = random.choices(list(range(n_img//n_consec)), k=n_show//n_consec)
    consecs = np.arange(n_consec)
    sel = [n_consec*i + consecs for i in sel]
    sel = np.array(sel).flatten().tolist()

# sel = list(range(n_show))

img_paths = []
vid_paths = []

for idx in sel:
    img = imgs[idx]
    seq = seqs[img['sid']]
    img_paths.append(join(seq, img['name'][:-3] + 'jpg'))
    vid_paths.append(seq + '.mp4')

cols = [Col('id1', 'ID')]
summary_row = [metric]

for i, name in enumerate(names):
    paths = [join(dirs[i], p) for p in img_paths]
    if link_video:
        hrefs = [join(dirs[i], p) for p in vid_paths]
    else:
        hrefs = paths
    cols.append(
        Col('img', name, paths, href=hrefs)
    )

    if 'Exp' in dirs[i]:
        eval_path = join(dirs[i].replace('/vis/', '/output/'), 'eval_summary.pkl')
        if isfile(eval_path):
            eval_summary = pickle.load(open(eval_path, 'rb'))
            summary_row.append('%.3g' % eval_summary['stats'][0])
        else:
            summary_row.append('')
    else:
        summary_row.append('')

imagetable(
    cols,
    join(out_dir, out_name),
    title,
    summary_row=summary_row,
    # imscale=0.3 + imsize=1
    imsize=(288, 180),
    sortable=True,
    sticky_header=True,
    sort_style='materialize',
    style='body {margin: 0}',
    pathrep=srv_dir,
)

url = f'http://{host_name}:{srv_port}{out_dir.replace(srv_dir, "")}/{out_name}'
print(url)