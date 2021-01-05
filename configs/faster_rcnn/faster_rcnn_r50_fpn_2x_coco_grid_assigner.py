_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_grid_assigner.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

work_dir = '/data2/nsathish/results/work_dirs/frcnn_r50_grid_assigner'