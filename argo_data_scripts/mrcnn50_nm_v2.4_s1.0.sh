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
	--config "/data2/nsathish/mmdetection-yolo/configs/mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py" \
	--weights "/data2/nsathish/mmdetection-yolo/checkpoints/mask_rcnn_r50_fpn_2x_coco_bbox_mAP-0.392__segm_mAP-0.354_20200505_003907-3e542a40.pth" \
	--in-scale 1.0 \
	--out-dir "/data2/nsathish/Exp/Argoverse-HD/output/mrcnn50_nm_v2.4_s1.0/val" \
