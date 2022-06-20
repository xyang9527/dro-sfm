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
from collections import OrderedDict
import torch

from dro_sfm.utils.setup_log import setup_log
from dro_sfm.utils.image import load_image
from dro_sfm.visualization.pointcloud_downsample import generate_pointcloud_NxN
from dro_sfm.visualization.viz_image_grid import VizImageGrid
from dro_sfm.utils.horovod import print0
from dro_sfm.utils.logging import pcolor
from dro_sfm.geometry.pose_trans import matrix_to_euler_angles


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

        sub_dirs = OrderedDict()
        sub_dirs['cam_left'] = '.jpg'
        sub_dirs['cam_left_vis'] = '.jpg'
        sub_dirs['depth_vis'] = '.jpg'
        sub_dirs['pose'] = '.txt'
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


def is_image(path):
    ext = osp.splitext(path)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png']:
        return True
    return False


def generate_video(rootdir, subdirs, im_h, im_w, n_row, n_col, video_name):
    logging.warning(f'generate_video(..)')
    if len(subdirs) < 1:
        return

    str_dir = osp.join(rootdir, list(subdirs.keys())[0])
    names = []
    for item in sorted(os.listdir(str_dir)):
        ext = osp.splitext(item)[1].lower()
        if ext in ['.jpg', '.png']:
            names.append(osp.splitext(item)[0])

    # video config
    grid = VizImageGrid(im_h, im_w, n_row, n_col)
    canvas_row = grid.canvas_row
    canvas_col = grid.canvas_col

    fps = 25.0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_name, fourcc, fps, (canvas_col, canvas_row))

    print(f'    -> write {video_name}')

    mats = []
    datetimes = []
    null_image = np.full((im_h, im_w, 3), 255, np.uint8)

    for idx_frame, item_name in enumerate(names):
        has_data = False
        for idx, (k, v) in enumerate(subdirs.items()):
            filename = osp.join(rootdir, k, item_name + v)

            if osp.exists(filename):
                has_data = True
                id_row = idx // n_col
                id_col = idx % n_col

                if is_image(filename):
                    data = cv2.imread(filename)
                    grid.subplot(id_row, id_col, data, f'({chr(ord("a") + idx)}) {k}')
                else:  # pose
                    data = np.genfromtxt(filename)
                    mats.append(data)
                    timestamp = int(np.int64(item_name) / 1e6)
                    datetimes.append(timestamp)

                    grid.subplot(id_row, id_col, null_image, f'({chr(ord("a") + idx)}) {k}')

                    text = []
                    if len(mats) > 1:
                        rel_pose = np.matmul(np.linalg.inv(mats[-2]), mats[-1])
                        xyz = matrix_to_euler_angles(torch.from_numpy(rel_pose[:3, :3]), 'XYZ')

                        xyz_degree = xyz.detach() * 180.0 / np.math.pi
                        dx, dy, dz = xyz_degree[:]

                        text.append(f'[{idx_frame:6d}] {item_name}')
                        text.append('to-prev-1:')
                        text.append(f'  dt:   {datetimes[-1] - datetimes[-2]:6d} ms')
                        text.append(f'  d_tx: {rel_pose[0, 3]*1000.0:6.3f} mm')
                        text.append(f'  d_ty: {rel_pose[1, 3]*1000.0:6.3f} mm')
                        text.append(f'  d_tz: {rel_pose[2, 3]*1000.0:6.3f} mm')
                        text.append(f'  d_rx: {dx:6.3f} deg')
                        text.append(f'  d_ry: {dy:6.3f} deg')
                        text.append(f'  d_rz: {dz:6.3f} deg')

                    if len(mats) > 5:
                        rel_pose = np.matmul(np.linalg.inv(mats[-6]), mats[-1])
                        xyz = matrix_to_euler_angles(torch.from_numpy(rel_pose[:3, :3]), 'XYZ')

                        xyz_degree = xyz.detach() * 180.0 / np.math.pi
                        dx, dy, dz = xyz_degree[:]
                        text.append('to-prev-5:')
                        text.append(f'  dt:   {datetimes[-1] - datetimes[-6]:6d} ms')
                        text.append(f'  d_t_xyz: ({rel_pose[0, 3]*1000.0:.3f}, {rel_pose[1, 3]*1000.0:.3f}, {rel_pose[2, 3]*1000.0:.3f})')
                        text.append(f'  d_r_xyz: ({dx:.3f}, {dy:.3f}, {dz:.3f})')
                        pass
                    grid.subtext(id_row, id_col, text)
            else:
                continue
        # // for idx, (k, v) in enumerate(subdirs.items()):

        if not has_data:
            continue

        video_writer.write(grid.canvas)
        if idx_frame > 300:
            break

    # // for idx_frame, item_name in enumerate(names):
    video_writer.release()


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
