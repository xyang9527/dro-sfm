#!/bin/bash

function text_info() {
  echo -e "\e[32m#   $1\e[39m"
}

function text_warn() {
  echo -e "\e[33m# $1\e[39m"
}

slam_home=${PWD}/..
echo "slam_home: ${slam_home}"

matterport_list=${PWD}/scripts/clean/matterport_seq.txt
scannet_list=${PWD}/scripts/clean/scannet_seq.txt
folder_list=${PWD}/scripts/clean/folder_to_remove.txt

function remove_temp_data() {
  var_filename_seq=$1
  var_filename_folder=$2

  while read line_seq; do
    if [[ ${#line_seq} -lt 1 ]]; then
      continue
    fi

    if [[ ${line_seq} =~ ^#.* ]]; then
      continue
    fi

    # is sequence existed ?
    if ! [ -d "${slam_home}/${line_seq}" ]; then
      text_info "skip ${slam_home}/${line_seq}"
      continue
    fi

    while read line_folder; do
      if [[ ${#line_folder} -lt 1 ]]; then
        continue
      fi

      if [[ ${line_folder} =~ ^#.* ]]; then
        continue
      fi

      # is folder existed ?
      if ! [ -d "${slam_home}/${line_seq}/${line_folder}" ]; then
        text_info "skip ${slam_home}/${line_seq}/${line_folder}"
        continue
      fi

      text_warn "rm -rf ${slam_home}/${line_seq}/${line_folder}"
      rm -rf "${slam_home}/${line_seq}/${line_folder}"

    done < ${var_filename_folder}
  done < ${var_filename_seq}
}

remove_temp_data ${matterport_list} ${folder_list}
remove_temp_data ${scannet_list} ${folder_list}

time_sh_start=$(date +"%s.%N")

time_sh_end=$(date +"%s.%N")
time_diff_sh=$(bc <<< "$time_sh_end - $time_sh_start")
text_warn "remove_temp_data.sh elapsed:        $time_diff_sh   seconds. ($time_sh_end - $time_sh_start)"

: '
conda activate dro-sfm-latest
cd slam/dro-sfm-xyang9527
bash scripts/clean/remove_temp_data.sh

bash scripts/clean/remove_temp_data.sh 2>&1 | tee -a logs/remove_temp_data.log
'