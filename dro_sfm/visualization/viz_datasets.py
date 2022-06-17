# -*-  coding=utf-8 -*-

import os
import os.path as osp
import sys
lib_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(lib_dir)
import time
import logging
import numpy as np
import subprocess
import cv2

from dro_sfm.utils.setup_log import setup_log
from dro_sfm.utils.image import load_image
from dro_sfm.visualization.pointcloud_downsample import generate_pointcloud_NxN
from dro_sfm.visualization.viz_image_grid import VizImageGrid
from dro_sfm.utils.horovod import print0
from dro_sfm.utils.logging import pcolor


def get_datasets():
    logging.warning(f'get_datasets()')
    slam_home = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
    clean_home = osp.join(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))), 'scripts/clean')

    names = {'matterport': 'matterport_seq.txt', 'scannet': 'scannet_seq.txt'}
    datasets = {}
    for k, v in names.items():
        with open(osp.join(clean_home, v), 'r') as f_in:
            lines = f_in.readlines()

            for line in lines:
                line = line.strip()

                if line.startswith('#'):
                    continue

                if len(line) < 1:
                    continue

                datasets.setdefault(k, []).append(osp.join(slam_home, line))
    return datasets


def generate_matterport_videos(names):
    logging.warning(f'generate_matterport_videos({len(names)})')
    for item in names:
        print0(pcolor(f'  {item}', 'green'))

        sub_dirs = ['cam_left', 'cam_left_vis', 'depth_vis', 'pose']
        video_name = osp.join(item, 'summary.avi')
        generate_video(item, sub_dirs, 480, 640, 2, 2, video_name)
        break
    pass


def generate_scannet_videos(names):
    logging.warning(f'generate_scannet_videos({len(names)})')
    for item in names:
        print0(pcolor(f'  {item}', 'blue'))

        sub_dirs = ['color', 'depth', 'pose']
    pass


def generate_video(rootdir, subdirs, im_h, im_w, n_row, n_col, video_name):
    logging.warning(f'generate_video(..)')
    str_dir = osp.join(rootdir, subdirs[0])
    names = []
    for item in sorted(os.listdir(str_dir)):
        ext = osp.splitext(item)[1].lower()
        if ext in ['.jpg', '.png']:
            names.append(osp.splitext(item)[0])

    # video config
    grid = VizImageGrid(im_h, im_w, n_row, n_col)
    canvas_row = grid.canvas_row
    canvas_col = grid.canvas_col

    fps = 4.0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_name, fourcc, fps, (canvas_col, canvas_row))

    print(f'write {video_name}')

    for idx_frame, item_name in enumerate(names):
        filename = osp.join(rootdir, subdirs[0], item_name + '.jpg')
        if osp.exists(filename):
            data = cv2.imread(filename)
        else:
            continue
        grid.subplot(0, 0, data, 'Left Camera')
        video_writer.write(grid.canvas)
        if idx_frame > 100:
            break
        print(f'write {filename}')
    video_writer.release()

    pass


if __name__ == '__main__':
    setup_log('kneron_viz_datasets.log')
    time_beg_pointcloud = time.time()

    np.set_printoptions(precision=6, suppress=True)

    datasets = get_datasets()

    for v, k in datasets.items():
        print(f'{v:<12s} {len(k):3d}')

    generate_matterport_videos(datasets['matterport'])
    generate_scannet_videos(datasets['scannet'])

    time_end_pointcloud = time.time()
    print(f'viz_datasets.py elapsed {time_end_pointcloud - time_beg_pointcloud:.6f} seconds.')
