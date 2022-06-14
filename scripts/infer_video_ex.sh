#!/bin/bash

function text_info() {
  echo -e "\e[32m# $1\e[39m"
}

function text_warn() {
  echo -e "\e[33m# $1\e[39m"
}

time_sh_start=$(date +"%s.%N")

file_model=${PWD}/scripts/path_model_list
file_data=${PWD}/scripts/path_data_list

# echo $(cat ${file_model})
# echo $(cat ${file_data})
# echo $(wc -l ${file_model})
# echo $(wc -l ${file_data})

var_sample_rate=3
var_max_frames=450
var_archive_video=/home/sigma/slam/demo

function dro_sfm_benchmark() {
    var_filename_model=$1
    var_filename_data=$2

    while read line_model; do
        if [[ ${#line_model} -lt 1 ]]; then
            text_warn "skip empty line"
            continue
        fi

        if [[ ${line_model} =~ ^#.* ]]; then
            text_warn "ignore ${line_model}"
            continue
        fi

        while read line_data; do
            if [[ ${#line_data} -lt 1 ]]; then
                text_warn "skip empty line"
                continue
            fi

            if [[ ${line_data} =~ ^#.* ]]; then
                text_warn "ignore ${line_data}"
                continue
            fi

            IFS=' '
            read -ra bool_str <<< "${line_data}"

            # echo "  length of bool_str: ${#bool_str[@]}"
            # echo "  bool_str[0] ${bool_str[0]}"
            # echo "  bool_str[1] ${bool_str[1]}"

            if [[ ${#bool_str[@]} -ne 2 ]]; then
                text_warn "ignore ${line_data}"
                continue
            fi

            is_scannet=${bool_str[0]}
            if [ "${is_scannet}" = true ] ; then
                var_data_path=${bool_str[1]}/color
                var_data_type=scannet
            else
                var_data_path=${bool_str[1]}/cam_left
                var_data_type=matterport
            fi
            var_model_path=${line_model}

            echo "--checkpoint $(basename ${line_model})"
            echo "--input $(basename ${bool_str[1]})"

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
                --mix_video_mode \
                --archive_video ${var_archive_video}

        done < ${var_filename_data}

    done < ${var_filename_model}
}

dro_sfm_benchmark ${file_model} ${file_data}

cp -f ${PWD}/logs/kneron_infer_video.log ${var_archive_video}/

time_sh_end=$(date +"%s.%N")
time_diff_sh=$(bc <<< "$time_sh_end - $time_sh_start")
text_warn "infer_video_ex.sh elapsed:        $time_diff_sh   seconds. ($time_sh_end - $time_sh_start)"
text_info "    ${file_model}"
text_info "    ${file_data}"

: '
conda activate dro-sfm-latest
cd slam/dro-sfm-xyang9527
bash scripts/infer_video_ex.sh

bash scripts/infer_video_ex.sh 2>&1 | tee -a logs/infer_video_ex.log
'