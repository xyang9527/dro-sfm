# -*- coding=utf-8 -*-

import os
import os.path as osp
import sys
lib_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(lib_dir)
import logging
import numpy as np
from PIL import Image
import time
import yaml

from dro_sfm.utils.setup_log import setup_log
from dro_sfm.visualization.viz_image_grid import VizImageGrid, Colors
from dro_sfm.utils.horovod import print0
from dro_sfm.utils.logging import pcolor
from dro_sfm.visualization.viz_datasets import get_datasets


def is_invalid_pose(pose: np.ndarray) -> bool:
    has_illegal_value = False
    h, w = pose.shape

    for j in range(h):
        if has_illegal_value:
            break

        for i in range(w):
            v = pose[j, i]
            if np.isnan(v) or np.isneginf(v) or np.isposinf(v):
                has_illegal_value = True
                break
    return has_illegal_value


def sequence_filter():
    logging.warning('sequence_filter()')
    datasets = get_datasets()

    matterport_seqs = datasets['matterport']
    for idx_seq, item_seq in enumerate(matterport_seqs):
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
            unique_depth = np.unique(depth_png)
            num_unique_depth = unique_depth.shape[0]

            depth_mask = depth_png <= 0
            num_pix_invalid = np.count_nonzero(depth_mask)
            num_pix_total = depth_png.shape[0] * depth_png.shape[1]

            pose_data.append((str_name, pose, num_unique_depth, num_pix_invalid, num_pix_total))

        # load depth info
        depth_data = []
        for idx, item in enumerate(pose_data):
            # print(f'    {idx:6d} - {item}')
            pass



    pass


if __name__ == '__main__':
    setup_log('kneron_matterport_filter.log')
    time_beg_matterport_filter = time.time()

    sequence_filter()

    time_end_matterport_filter = time.time()
    print0(pcolor(f'matterport_filter.py elapsed {time_end_matterport_filter - time_beg_matterport_filter:.6f} seconds.', 'red'))
