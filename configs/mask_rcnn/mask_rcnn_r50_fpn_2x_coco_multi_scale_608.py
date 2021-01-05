_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance_multi_scale_608.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]


work_dir = '/data2/nsathish/results/work_dirs/mrcnn_r50_multi_scale_608'
