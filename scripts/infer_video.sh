#!/bin/bash

time_sh_start=$(date +"%s.%N")

: '
/home/sigma/Downloads/dro-sfm-bodong/home/bodong/playground/slam/dro-sfm/results/mdoel/scannet_gt_view2_ori/
SupModelMF_DepthPoseNet_it12-h-out_epoch=27_test-test_split-groundtruth-abs_rel_pp_gt=0.057.ckpt

train@sigma
/home/sigma/slam/dro-sfm-xyang9527/results_20220604/model/matterport_gt/
SupModelMF_DepthPoseNet_it12-h-out_epoch=106_matterport-val_all_list-groundtruth-abs_rel_pp_gt=0.060.ckpt

train@trex
/home/sigma/slam/models/24@trex_neg_xyz/
SupModelMF_DepthPoseNet_it12-h-out_epoch=162_matterport-val_all_list-groundtruth-abs_rel_pp_gt=0.066.ckpt

train@fox
/home/sigma/slam/models/26@fox_without_neg_xyz/
SupModelMF_DepthPoseNet_it12-h-out_epoch=198_matterport-val_all_list-groundtruth-abs_rel_pp_gt=0.063.ckpt
'
model_matterport=/home/sigma/slam/dro-sfm-xyang9527/neg_xyz_results_20220611/model/matterport_gt/SupModelMF_DepthPoseNet_it12-h-out_epoch=49_matterport-val_all_list-groundtruth-abs_rel_pp_gt=0.052.ckpt

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

/home/sigma/slam/matterport/test/
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

is_scannet=false
var_archive_video=/home/sigma/slam/demo

# set model_path and data_path
if [ "${is_scannet}" = true ] ; then
  var_data_path=${data_scannet}/color
  var_data_type=scannet
else
  var_data_path=${data_matterport}/cam_left
  var_data_type=matterport
fi
var_model_path=${model_matterport}

var_sample_rate=3
var_max_frames=150

echo "var_data_path:    ${var_data_path}"
echo "var_model_path:   ${var_model_path}"

echo "var_sample_rate:  ${var_sample_rate}"
echo "var_max_frames:   ${var_max_frames}"

python scripts/infer_video.py \
  --checkpoint ${var_model_path} \
  --input ${var_data_path} \
  --output $(dirname ${var_data_path})/infer_video/$(basename ${var_model_path})_sample_rate-${var_sample_rate}_max_frames_${var_max_frames} \
  --sample_rate ${var_sample_rate} \
  --data_type ${var_data_type} \
  --ply_mode \
  --use_depth_gt \
  --use_pose_gt \
  --max_frames ${var_max_frames} \
  --archive_video ${var_archive_video}

#   --mix_video_mode \

cp -f ${PWD}/logs/kneron_infer_video.log ${var_archive_video}/

function text_info() {
  echo -e "\e[32m# $1\e[39m"
}

function text_warn() {
  echo -e "\e[33m# $1\e[39m"
}

time_sh_end=$(date +"%s.%N")
time_diff_sh=$(bc <<< "$time_sh_end - $time_sh_start")
text_warn "infer_video.sh elapsed:        $time_diff_sh   seconds. ($time_sh_end - $time_sh_start)"
text_info "    ${var_model_path}"
text_info "    ${var_data_path}"
