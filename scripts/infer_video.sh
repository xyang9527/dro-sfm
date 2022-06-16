#!/bin/bash

time_sh_start=$(date +"%s.%N")

: '
/home/sigma/slam/dro-sfm-xyang9527/results/model
'
model_matterport=/home/sigma/slam/dro-sfm-xyang9527/results/model/matterport_gt/SupModelMF_DepthPoseNet_it12-h-out_epoch=52_matterport0614-val_all_list-groundtruth-abs_rel_pp_gt=0.150.ckpt

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

/home/sigma/slam/matterport0614/train_val_test
matterport005_000_0516
matterport005_001_0516
matterport005_0614
matterport010_000_0516
matterport010_001_0516
matterport010_0614
matterport047_0614
matterport063_0614
matterport071_0614

/home/sigma/slam/matterport0614/test
matterport014_000_0516
matterport014_001_0516
matterport014_0614
'
data_matterport=/home/sigma/slam/matterport0614/train_val_test/matterport005_000_0516

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

# ===== < config > =====
is_scannet=false
use_scannet_model=false

var_sample_rate=3
var_max_frames=40
# ===== </config > =====

var_archive_video=/home/sigma/slam/demo

# set model_path and data_path
if [ "${is_scannet}" = true ] ; then
  var_data_path=${data_scannet}/color
  var_data_type=scannet
else
  var_data_path=${data_matterport}/cam_left
  var_data_type=matterport
fi

if [ "${use_scannet_model}" = true ] ; then
  var_model_path=${model_scannet}
else
  var_model_path=${model_matterport}
fi

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
  --max_frames ${var_max_frames} \
  --mix_video_mode \
  --archive_video ${var_archive_video}

# --use_depth_gt \
# --use_pose_gt \

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
text_info "    var_sample_rate:  ${var_sample_rate}"
text_info "    var_max_frames:   ${var_max_frames}"

: '
conda activate dro-sfm-latest
cd slam/dro-sfm-xyang9527
bash scripts/infer_video.sh

bash scripts/infer_video.sh 2>&1 | tee -a logs/infer_video.log
'
