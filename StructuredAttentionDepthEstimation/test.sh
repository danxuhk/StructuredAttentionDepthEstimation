#!/bin/bash
clear
echo $"Start testing ..."
model_defition_path="./prototxt/test_SAN.prototxt"
model_weights="./models/SAN_1_iter_20000.caffemodel"
pred_output_save="./output/SAN_1_iter_20000.npy"
kitti_data_root="/scratch/local/ssd/danxu/KITTI/kitti_raw_data/"
prediction_layer_blob_name="final_output"
if_save_depth="False"
gpu_id=1
crop_type_for_evaluation=$"garg"
python test_kitti_depth.py --model_def=$model_defition_path --weights=$model_weights --pred_file=$pred_output_save --data_root=$kitti_data_root --prediction_blob=$prediction_layer_blob_name --gpu=$gpu_id --save_depth_output=$if_save_depth
echo $"Start evaluating"
python ./utils/evaluation_depth.py --kitti_dir=$kitti_data_root --pred_file=$pred_output_save --which_crop=$crop_type_for_evaluation
