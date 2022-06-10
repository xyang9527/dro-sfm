#!/bin/bash

: '
/home/sigma/Downloads/dro-sfm-bodong/home/bodong/playground/slam/dro-sfm/results/mdoel/scannet_gt_view2_ori/
SupModelMF_DepthPoseNet_it12-h-out_epoch=27_test-test_split-groundtruth-abs_rel_pp_gt=0.057.ckpt

/home/sigma/slam/dro-sfm-xyang9527/results_20220604/model/matterport_gt/
SupModelMF_DepthPoseNet_it12-h-out_epoch=106_matterport-val_all_list-groundtruth-abs_rel_pp_gt=0.060.ckpt
'
model_matterport=$(pwd)/results_20220604/model/matterport_gt/SupModelMF_DepthPoseNet_it12-h-out_epoch=106_matterport-val_all_list-groundtruth-abs_rel_pp_gt=0.060.ckpt

: '
/mnt/datasets_open/dro-sfm_data/models/
indoor_scannet.ckpt
indoor_scannet_selfsup.ckpt
indoor_scannet_view5.ckpt
outdoor_kitti.ckpt
outdoor_kitti_selfsup.ckpt
'
model_scannet=/mnt/datasets_open/dro-sfm_data/models/indoor_scannet.ckpt

: '
/home/sigma/slam/matterport/train_val_test/
matterport005_000
matterport005_001
matterport010_000
matterport010_001

/home/sigma/slam/matterport/test
matterport005_000_0610
matterport014_000
'
data_matterport=/home/sigma/slam/matterport/test/matterport005_000_0610

: '
/home/sigma/slam/scannet_train_data/
scene0000_00
scene0100_00
scene0200_00
scene0300_00
scene0400_00
scene0500_00
scene0600_00
'
data_scannet=/home/sigma/slam/scannet_train_data/scene0600_00

is_scannet=true
if [ "${is_scannet}" = true ] ; then
  data_path=${data_scannet}/color
  data_type=scannet
else
  data_path=${data_matterport}/cam_left
  data_type=matterport
fi
echo "data_path:  ${data_path}"
model_path=${model_scannet}
echo "model_path: ${model_path}"

time_sh_start=$(date +"%s.%N")

python scripts/infer_video.py \
  --checkpoint ${model_path} \
  --input ${data_path} \
  --output $(dirname ${data_path})/infer_video/$(basename ${model_path}) \
  --sample_rate 3 \
  --data_type ${data_type} \
  --ply_mode \
  --use_depth_gt \
  --use_pose_gt \
  --max_frames 50

function text_info() {
  echo -e "\e[32m# $1\e[39m"
}

function text_warn() {
  echo -e "\e[33m# $1\e[39m"
}

time_sh_end=$(date +"%s.%N")
time_diff_sh=$(bc <<< "$time_sh_end - $time_sh_start")
text_warn "infer_video.sh elapsed:        $time_diff_sh   seconds. ($time_sh_end - $time_sh_start)"
text_info "    ${model_path}"
text_info "    ${data_path}"
