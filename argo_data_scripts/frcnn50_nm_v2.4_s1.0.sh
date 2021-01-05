dataDir="/data2/mengtial"
ssdDir="/scratch/mengtial"
if [ ! -d "$ssdDir/Argoverse-1.1/tracking" ]; then
  ssdDir="$dataDir"
fi

python tools/test_modified_argo.py \
	--no-mask \
	--overwrite \
	--data-root "$ssdDir/Argoverse-1.1/tracking" \
	--annot-path "$dataDir/Argoverse-HD/annotations/val.json" \
	--config "/data2/nsathish/mmdetection-yolo/configs/faster_rcnn/faster_rcnn_r50_fpn_2x_coco.py" \
	--weights "/data2/nsathish/mmdetection-yolo/checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth" \
	--in-scale 1.0 \
	--out-dir "/data2/nsathish/Exp/Argoverse-HD/output/frcnn50_nm_v2.4_s1.0/val" \
