import csv, pickle
from glob import glob
from os.path import join, isfile, split

import numpy as np


data_split = 'val'
work_dir = '/data2/nsathish/results/ArgoVerse1.1/output/frcnn50_s1_scale_modified'

# header = [
#     'Method',
#     'AR@100', 'AR@300', 'AR@1000'
# ]
header = [
    'Method',
    'AP', 'AP-0.5', 'AP-0.7',
    'AP-small', 'AP-medium', 'AP-large',
    'R@1', 'R@10', 'R@100',
    'R-Small', 'R-medium', 'R-large'
]
runs = glob(join(work_dir, '*', 's*_' + data_split))

out_path = join(work_dir, 'frcnn50_s1_scale_modified.csv')
with open(out_path, 'w') as f:
    w = csv.writer(f)
    w.writerow(header)
    for r in runs:
        eval_path = join(r, 'eval_summary.pkl')
        if not isfile(eval_path):
            continue
        eval_summary = pickle.load(open(eval_path, 'rb'))['stats'][:6]

        p, data_setting = split(r)
        name = split(p)[1]
        scale = data_setting.split('_')[0][1:]

        time_path = join(r, 'time_all.pkl')
        if isfile(time_path):
            time_all = pickle.load(open(time_path, 'rb'))
            # (runtime_all, n_processed, n_total, n_small_runtime)
            runtime = np.array(time_all[0])
            time_stats = 1e3*np.array([
                runtime.mean(),
                runtime.std(ddof=1),
                runtime.min(),
                runtime.max(),
            ])
            time_stats = np.append(time_stats, time_all[3]/time_all[1])
            time_extra_path = join(r, 'time_extra.txt')
            if isfile(time_extra_path):
                time_extra = np.loadtxt(time_extra_path)
                # [miss, in_time, shifts]
                time_stats = np.append(time_stats, time_extra[0])
                time_stats = np.append(time_stats, time_extra[1]/time_all[2])
                time_stats = np.append(time_stats, time_extra[2]/time_all[2])
            time_stats = time_stats.tolist()
        else:
            time_stats = []

        w.writerow([name, scale] + eval_summary.tolist() + time_stats)

print(out_path)
