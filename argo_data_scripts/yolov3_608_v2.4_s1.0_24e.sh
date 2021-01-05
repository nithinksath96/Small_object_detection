dataDir="/data2/mengtial"
ssdDir="/scratch/mengtial"
if [ ! -d "$ssdDir/Argoverse-1.1/tracking" ]; then
  ssdDir="$dataDir"
fi

python tools/test_modified_argo.py \
	--no-mask \
	--overwrite \
	--data-root "$dataDir/Argoverse-1.1/tracking" \
	--annot-path "$dataDir/Argoverse-HD/annotations/val.json" \
	--config "/data2/nsathish/mmdetection-yolo/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py" \
	--weights "/data2/nsathish/results/work_dirs/yolov3_608_24e/epoch_24.pth" \
	--in-scale 1.0 \
	--out-dir "/data2/nsathish/Exp/Argoverse-HD/output/yolov3_608/val" \