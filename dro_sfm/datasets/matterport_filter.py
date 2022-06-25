# -*- coding=utf-8 -*-

from typing import List
import os
import os.path as osp
import sys
lib_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(lib_dir)
import logging
import numpy as np
import subprocess
import cv2
from collections import OrderedDict
from PIL import Image
import time
import yaml
import re
import torch

from dro_sfm.utils.setup_log import setup_log
from dro_sfm.visualization.viz_image_grid import VizImageGrid, Colors
from dro_sfm.utils.horovod import print0
from dro_sfm.utils.logging import pcolor
from dro_sfm.geometry.pose_trans import matrix_to_euler_angles
from dro_sfm.visualization.viz_datasets import get_datasets
from dro_sfm.datasets.depth_filter import clip_depth, is_invalid_pose, find_idx_of_prev_n, matrix_to_6d_pose, pose_in_threshold_1
from dro_sfm.visualization.viz_datasets import generate_video


def sequence_filter():
    logging.warning('sequence_filter()')
    datasets = get_datasets()

    enable_gen_video = False

    matterport_seqs = datasets['matterport']
    dir_root = osp.dirname(osp.dirname(matterport_seqs[0]))
    split_train = osp.join(dir_root, 'splits', 'filtered_train_all_list.txt')
    split_test = osp.join(dir_root, 'splits', 'filtered_test_all_list.txt')
    with open(split_train, 'w') as f_ou_split_train, open(split_test, 'w') as f_ou_split_test:
        for idx_seq, item_seq in enumerate(matterport_seqs):
            if '/train_val_test/' in item_seq:
                is_train = True
            else:
                is_train = False
            print(f'  -> {idx_seq:2d} : {item_seq}')

            pose_dir = osp.join(item_seq, 'pose')
            depth_dir = osp.join(item_seq, 'depth')

            pose_data = []
            # load pose
            for item_file in sorted(os.listdir(pose_dir)):
                str_name, str_ext = osp.splitext(item_file)
                if str_ext != '.txt':
                    continue

                # valid pose only
                pose = np.genfromtxt(osp.join(pose_dir, item_file))
                if is_invalid_pose(pose):
                    print(f'  invalid pose: {osp.join(pose_dir, item_file)}')
                    continue

                # depth info
                depth_name = osp.join(depth_dir, str_name + '.png')
                if not osp.exists(depth_name):
                    continue

                depth_png = np.array(Image.open(depth_name), dtype=int)
                depth_png = clip_depth(depth_png)
                unique_depth = np.unique(depth_png)
                num_unique_depth = unique_depth.shape[0]

                depth_mask = depth_png <= 0
                num_pix_invalid = np.count_nonzero(depth_mask)
                num_pix_total = depth_png.shape[0] * depth_png.shape[1]

                pose_data.append((str_name, pose, num_unique_depth, num_pix_invalid, num_pix_total))

            # archive (debug)
            save_dir = osp.join(item_seq, 'filtered_split')
            if not osp.exists(save_dir):
                os.makedirs(save_dir)

            save_name = osp.join(save_dir, 'all.txt')
            with open(save_name, 'w') as f_ou:
                for idx, item in enumerate(pose_data):
                    v0, v1, v2, v3, v4 = item
                    f_ou.write(f'[{idx:4d}] {v0} {v2} {v3} {v4}\n')

            n_frames = len(pose_data)
            to_drop = [False] * n_frames
            to_split = [False] * n_frames
            num_valid = [0] * n_frames
            inited = False
            for idx_frame, item_data in enumerate(pose_data):
                str_name, pose, num_unique_depth, num_pix_invalid, num_pix_total = item_data
                # depth threshold
                if np.float(num_pix_invalid) / np.float(num_pix_total) > 0.4:
                    to_drop[idx_frame] = True
                    # print0(pcolor(f'    drop {str_name}', 'yellow'))
                    logging.info(f'    drop {str_name}')
                    if inited:
                        num_valid[idx_frame] = num_valid[idx_frame - 1]
                    continue

                # pose threshold - prev 1
                if inited and num_valid[idx_frame - 1] >= 1:
                    idx_targ = find_idx_of_prev_n(to_drop, num_valid, idx_frame, 1)
                    _, pose_targ, _, _, _ = pose_data[idx_targ]
                    pose_6d = matrix_to_6d_pose(pose, pose_targ)

                    if not pose_in_threshold_1(pose_6d):
                        to_split[idx_frame] = True
                        num_valid[idx_frame] = 1
                        continue

                # pose threshold - prev 5
                if inited and num_valid[idx_frame - 1] >= 5:
                    pass

                if not inited:
                    num_valid[idx_frame] = 1
                    inited = True
                else:
                    num_valid[idx_frame] = num_valid[idx_frame - 1] + 1
            # // for idx_frame, item_data in enumerate(pose_data)
            print(f'  total frames: {len(pose_data)}')
            print(f'  valid frames: {len(pose_data) - sum(to_drop)}')
            print(f'  split:        {sum(to_split)}')

            name_drop = osp.join(save_dir, 'invalid.txt')
            name_seq_all = osp.join(save_dir, 'seq_all.txt')
            with open(name_drop, 'w') as f_ou_drop, open(name_seq_all, 'w') as f_ou_all:
                words = item_seq.split('/')
                sub_seq = 0
                names_sub_seq = []
                for i in range(n_frames):
                    str_name, _, _, _, _ = pose_data[i]
                    if to_drop[i]:
                        f_ou_drop.write(f'{words[-2]}/{words[-1]}/cam_left {str_name}.jpg\n')
                        # f_ou_all.write(f'{words[-2]}/{words[-1]}/cam_left {idx_name}.jpg -1\n')
                        continue

                    if not to_split[i]:
                        names_sub_seq.append(str_name)
                    else:
                        name_seq = osp.join(save_dir, f'seq_{sub_seq:02d}.txt')
                        with open(name_seq, 'w') as f_ou_seq:
                            for idx_name in names_sub_seq:
                                f_ou_seq.write(f'{words[-2]}/{words[-1]}/cam_left {idx_name}.jpg\n')
                                f_ou_all.write(f'{words[-2]}/{words[-1]}/cam_left {idx_name}.jpg {sub_seq}\n')

                        # create sub sequences
                        if len(names_sub_seq) > 20:
                            dir_sub_home = osp.join(item_seq, 'sub_sequences')
                            dir_sub_seq = osp.join(dir_sub_home, f'seq_{sub_seq:02d}')
                            dir_cam_left = osp.join(dir_sub_seq, 'cam_left')
                            dir_depth = osp.join(dir_sub_seq, 'depth')
                            dir_pose = osp.join(dir_sub_seq, 'pose')
                            dirs = [dir_sub_home, dir_sub_seq, dir_cam_left, dir_depth, dir_pose]
                            for i in dirs:
                                if not osp.exists(i):
                                    os.mkdir(i)
                            tags = dir_sub_seq.split('/')
                            for idx_name in names_sub_seq:
                                subprocess.call(['cp', osp.join(item_seq, 'cam_left', f'{idx_name}.jpg'), osp.join(dir_cam_left, f'{idx_name}.jpg')])
                                subprocess.call(['cp', osp.join(item_seq, 'depth', f'{idx_name}.png'), osp.join(dir_depth, f'{idx_name}.png')])
                                subprocess.call(['cp', osp.join(item_seq, 'pose', f'{idx_name}.txt'), osp.join(dir_pose, f'{idx_name}.txt')])
                                if is_train:
                                    f_ou_split_train.write(f'{tags[-4]}/{tags[-3]}/{tags[-2]}/{tags[-1]}/cam_left {idx_name}.jpg\n')
                                else:
                                    f_ou_split_test.write(f'{tags[-4]}/{tags[-3]}/{tags[-2]}/{tags[-1]}/cam_left {idx_name}.jpg\n')

                        names_sub_seq.clear()
                        names_sub_seq.append(str_name)
                        sub_seq += 1
                # // for i in range(n_frames):
            if enable_gen_video:
                seq_to_video(item_seq)
            # break
        # // for idx_seq, item_seq in enumerate(matterport_seqs):


def seq_to_video(root_dir):
    logging.warning(f'seq_to_video({root_dir})')
    filtered_dir = osp.join(root_dir, 'filtered_split')
    if not osp.exists(filtered_dir):
        logging.warning(f'path not exist: {filtered_dir}')
        return

    sub_dirs = OrderedDict()
    sub_dirs['cam_left'] = '.jpg'
    sub_dirs['cam_left_vis'] = '.jpg'
    sub_dirs['depth'] = '.png'
    sub_dirs['pose'] = '.txt'

    # invalid frames
    filename_in = osp.join(filtered_dir, 'invalid.txt')
    filename_ou = filename_in.replace('.txt', '.avi')
    name_list = []
    with open(filename_in, 'r') as f_in:
        text = f_in.readlines()
        for line in text:
            line = line.strip()
            words = line.split(' ')
            if len(words) == 2:
                basename = osp.splitext(words[1])[0]
                name_list.append(basename)
            else:
                print(f'  unknown line: {line}')
    if len(name_list) > 0:
        generate_video(root_dir, sub_dirs, 480, 640, 2, 2, filename_ou, False, name_list)
    else:
        print0(pcolor(f'  - empty {filename_in}', 'yellow'))
        logging.warning('f  - empty {filename_in}')

    # seq_xxx.txt
    re_seq = re.compile('seq_[\d]+\.txt')
    for item in sorted(os.listdir(filtered_dir)):
        m = re_seq.match(item)
        if m is not None:
            filename_in = osp.join(filtered_dir, item)
            filename_ou = filename_in.replace('.txt', '.avi')
            name_list = []
            with open(filename_in, 'r') as f_in:
                text = f_in.readlines()
                for line in text:
                    line = line.strip()
                    words = line.split(' ')
                    if len(words) == 2:
                        basename = osp.splitext(words[1])[0]
                        name_list.append(basename)
                    else:
                        print(f'  unknown line: {line}')
            if len(name_list) > 0:
                generate_video(root_dir, sub_dirs, 480, 640, 2, 2, filename_ou, False, name_list, 3.0, True)
            else:
                print0(pcolor(f'  - empty {filename_in}', 'yellow'))
                logging.warning('f  - empty {filename_in}')

    # sub sequece frames
    filename_in = osp.join(filtered_dir, 'seq_all.txt')
    filename_ou = filename_in.replace('.txt', '.avi')
    name_list = []
    with open(filename_in, 'r') as f_in:
        text = f_in.readlines()
        for line in text:
            line = line.strip()
            words = line.split(' ')
            if len(words) == 3:
                basename = osp.splitext(words[1])[0]
                name_list.append(basename)
            else:
                print(f'  unknown line: {line}')
    if len(name_list) > 0:
        generate_video(root_dir, sub_dirs, 480, 640, 2, 2, filename_ou, False, name_list)
    else:
        print0(pcolor(f'  - empty {filename_in}', 'yellow'))
        logging.warning('f  - empty {filename_in}')


if __name__ == '__main__':
    setup_log('kneron_matterport_filter.log')
    time_beg_matterport_filter = time.time()

    sequence_filter()

    time_end_matterport_filter = time.time()
    print0(pcolor(f'matterport_filter.py elapsed {time_end_matterport_filter - time_beg_matterport_filter:.6f} seconds.', 'red'))
