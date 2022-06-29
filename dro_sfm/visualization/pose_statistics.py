# -*- coding=utf-8 -*-

import os
import os.path as osp
import sys
lib_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(lib_dir)
import time
import logging
import numpy as np
import math

from dro_sfm.utils.setup_log import setup_log
from dro_sfm.utils.horovod import print0
from dro_sfm.utils.logging import pcolor


def main():
    logging.warning('main()')

    slam_home = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
    clean_home = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))), 'scripts/clean')

    logging.info(f'slam_home:  {slam_home}')
    logging.info(f'clean_home: {clean_home}')

    data_list = []
    names = ['matterport_seq.txt', 'scannet_seq.txt']

    for item_seq in names:
        with open(osp.join(clean_home, item_seq), 'r') as f_in:
            lines = f_in.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith('#'):
                    continue
                if len(line) < 1:
                    continue
                data_list.append(osp.join(slam_home, line))

    for idx, item_dir in enumerate(data_list):
        logging.info(f'{idx:4d} {item_dir}')

        is_matterport = False
        if item_dir.find('matterport') != -1:
            is_matterport = True

        pose_dir = osp.join(item_dir, 'pose')
        pose_files = sorted(os.listdir(pose_dir))

        camera_position = []
        camera_timestamp = []

        for item_file in pose_files:
            if not item_file.endswith('.txt'):
                continue

            pose = np.genfromtxt(osp.join(pose_dir, item_file))
            has_nan = False
            w, h = pose.shape
            for i in range(w):
                if has_nan:
                    break
                for j in range(h):
                    if np.isnan(pose[i, j]) or np.isneginf(pose[i, j]) or np.isposinf(pose[i, j]):
                        has_nan = True
                        break

            if has_nan:
                continue

            camera_position.append((pose[0, 3], pose[1, 3], pose[2, 3]))

            if is_matterport:
                dt = np.int64(osp.splitext(item_file)[0][:-6])
                camera_timestamp.append(dt)

        # statistics
        n_frame = len(camera_position)
        pos_dis_max = None
        pos_dis_min = None
        pos_dis_sum = 0
        dt_diff_min = None
        dt_diff_max = None
        dt_diff_sum = 0

        for i in range(1, n_frame):
            x0, y0, z0 = camera_position[i-1]
            x1, y1, z1 = camera_position[i]

            dx, dy, dz = x1-x0, y1-y0, z1-z0
            pos_dis = math.sqrt(dx*dx + dy*dy + dz*dz)
            pos_dis_sum += pos_dis

            if pos_dis_max is None or pos_dis_max < pos_dis:
                pos_dis_max = pos_dis

            if pos_dis_min is None or pos_dis_min > pos_dis:
                pos_dis_min = pos_dis

            if is_matterport:
                dt_diff = camera_timestamp[i] - camera_timestamp[i-1]
                dt_diff_sum += dt_diff

                if dt_diff_max is None or dt_diff_max < dt_diff:
                    dt_diff_max = dt_diff

                if dt_diff_min is None or dt_diff_min > dt_diff:
                    dt_diff_min = dt_diff

        pos_dis_mean = np.float(pos_dis_sum) / (1.0 * (n_frame - 1))
        print(f'\n=== {osp.basename(item_dir)}: ===')
        print(f'  move:')
        print(f'    min:  {pos_dis_min:10.6f}')
        print(f'    max:  {pos_dis_max:10.6f}')
        print(f'    mean: {pos_dis_mean:10.6f}')

        if is_matterport:
            dt_diff_mean = np.float(dt_diff_sum) / (1.0 * (n_frame - 1))
            print(f'  timestamp:')
            print(f'    min:  {dt_diff_min:10d} (ms)')
            print(f'    max:  {dt_diff_max:10d} (ms)')
            print(f'    mean: {dt_diff_mean:10.6f} (ms)')


if __name__ == '__main__':
    setup_log('kneron_pose_statistics.log')
    time_beg_pose_statistics = time.time()

    np.set_printoptions(precision=6, suppress=True)
    main()

    time_end_pose_statistics = time.time()
    logging.warning(f'pose_statistics.py elapsed {time_end_pose_statistics - time_beg_pose_statistics:.6f} seconds.')
    print0(pcolor(f'pose_statistics.py elapsed {time_end_pose_statistics - time_beg_pose_statistics:.6f} seconds.', 'yellow'))
