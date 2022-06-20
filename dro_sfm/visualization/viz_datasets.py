# -*-  coding=utf-8 -*-

from typing import List
import os
import os.path as osp
import sys
lib_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(lib_dir)
import time
import logging
import numpy as np
import cv2
from collections import OrderedDict
from PIL import Image
import torch

from dro_sfm.utils.setup_log import setup_log
from dro_sfm.visualization.viz_image_grid import VizImageGrid
from dro_sfm.utils.horovod import print0
from dro_sfm.utils.logging import pcolor
from dro_sfm.geometry.pose_trans import matrix_to_euler_angles
from dro_sfm.utils.depth import viz_inv_depth


class CameraMove:
    def __init__(self, desc):
        self.desc = desc
        self.init = False

        self.d_tx = None
        self.d_ty = None
        self.d_tz = None

        self.d_rx = None
        self.d_ry = None
        self.d_rz = None

    def update(self, tx, ty, tz, rx, ry, rz):
        if not self.init:
            self.d_tx, self.d_ty, self.d_tz = tx, ty, tz
            self.d_rx, self.d_ry, self.d_rz = rx, ry, rz
            self.init = True
            return

        if np.abs(self.d_tx) < np.abs(tx):
            self.d_tx = tx
        if np.abs(self.d_ty) < np.abs(ty):
            self.d_ty = ty
        if np.abs(self.d_tz) < np.abs(tz):
            self.d_tz = tz

        if np.abs(self.d_rx) < np.abs(rx):
            self.d_rx = rx
        if np.abs(self.d_ry) < np.abs(ry):
            self.d_ry = ry
        if np.abs(self.d_rz) < np.abs(rz):
            self.d_rz = rz

    def __repr__(self) -> str:
        sqrt_t = np.math.sqrt(self.d_tx ** 2 + self.d_ty ** 2 + self.d_tz ** 2)
        sqrt_r = np.math.sqrt(self.d_rx ** 2 + self.d_ry ** 2 + self.d_rz ** 2)
        return f'\n{self.desc}:\n  max_t: ({self.d_tx}, {self.d_ty}, {self.d_tz}) @ {sqrt_t}\n  max_r: ({self.d_rx}, {self.d_ry}, {self.d_rz}) @ {sqrt_r}\n'


class HoleInfo:
    def __init__(self, desc: str, im_h: int, im_w:int , pixels_invalid: List[int]):
        self.desc = desc
        self.im_h = im_h
        self.im_w = im_w
        self.pixels_invalid = pixels_invalid

        self.n_frame = len(pixels_invalid)
        self.pixels_total = self.im_h * self.im_w

        self.max = max(self.pixels_invalid)
        self.min = min(self.pixels_invalid)
        self.mean = np.float(sum(self.pixels_invalid)) / np.float(self.n_frame)

        self.max_percent = self.max * 100.0 / np.float(self.pixels_total)
        self.min_percent = self.min * 100.0 / np.float(self.pixels_total)
        self.mean_percent = self.mean * 100.0 / np.float(self.pixels_total)

    def __repr__(self) -> str:
        text_max = f'  max:       {self.max:10d} ({self.max_percent:5.2f}%)'
        text_min = f'  min:       {self.min:10d} ({self.min_percent:5.2f}%)'
        text_mean = f'  mean:      {self.mean:10.0f} ({self.mean_percent:5.2f}%)'
        return f'\n{self.desc}\n{text_max}\n{text_min}\n{text_mean}\n'


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
        # sub_dirs['depth_vis'] = '.jpg'
        sub_dirs['depth'] = '.png'
        sub_dirs['pose'] = '.txt'
        video_name = osp.join(item, 'summary.avi')

        generate_video(item, sub_dirs, 480, 640, 2, 2, video_name, False)
        break


def generate_scannet_videos(names):
    logging.warning(f'generate_scannet_videos({len(names)})')
    for item in names:
        print0(pcolor(f'  {item}', 'blue'))

        sub_dirs = OrderedDict()
        sub_dirs['color'] = '.jpg'
        sub_dirs['depth'] = '.png'
        sub_dirs['pose'] = '.txt'
        video_name = osp.join(item, 'summary.avi')
        generate_video(item, sub_dirs, 968, 1296, 2, 2, video_name, True)
        break


def is_image(path):
    ext = osp.splitext(path)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png']:
        return True
    return False


def generate_video(rootdir, subdirs, im_h, im_w, n_row, n_col, video_name, is_scannet=False):
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
    cam_move_1 = CameraMove(f'{osp.basename(rootdir)} max( to_prev_1 )')
    cam_move_5 = CameraMove(f'{osp.basename(rootdir)} max( to_prev_5 )')
    arr_pix_invalid = []
    depth_h, depth_w = None, None

    for idx_frame, item_name in enumerate(names):
        has_data = False
        for idx, (k, v) in enumerate(subdirs.items()):
            filename = osp.join(rootdir, k, item_name + v)

            if osp.exists(filename):
                has_data = True
                id_row = idx // n_col
                id_col = idx % n_col

                if is_image(filename):
                    if k == 'depth' and v == '.png': # and is_scannet 
                        depth_png = np.array(Image.open(filename), dtype=int)
                        depth_mask = depth_png <= 0
                        depth = depth_png.astype(np.float) / 1000.0
                        data = viz_inv_depth(depth) * 255
                        data[depth_mask, :] = 0
                        data = data[..., ::-1].copy()

                        depth_h, depth_w = depth_png.shape
                        pixel_total = depth_h * depth_w
                        pixel_invalid = np.count_nonzero(depth_mask)
                        arr_pix_invalid.append(pixel_invalid)
                        percent = 100.0 * np.float(pixel_invalid) / np.float(pixel_total)

                        cv2.putText(img=data, text=f'invalid depth: {pixel_invalid:6d} ({percent:5.2f}%)',
                            org=(30, 50), fontScale=1, color=(0, 0, 255), thickness=2, fontFace=cv2.LINE_AA)
                    else:
                        data = cv2.imread(filename)

                    w, h, c = data.shape
                    if w != im_w or h != im_h:
                        data = cv2.resize(data, (im_w, im_h), interpolation=cv2.INTER_LINEAR)

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

                        text.append(f'[{idx_frame:4d}] {item_name} fps: {fps}')
                        text.append('to-prev-1:')
                        if not is_scannet:
                            text.append(f'  dt:   {datetimes[-1] - datetimes[-2]:6d} ms')
                        text.append(f'  d_tx: {rel_pose[0, 3]*1000.0:6.3f} mm')
                        text.append(f'  d_ty: {rel_pose[1, 3]*1000.0:6.3f} mm')
                        text.append(f'  d_tz: {rel_pose[2, 3]*1000.0:6.3f} mm')
                        text.append(f'  d_rx: {dx:6.3f} deg')
                        text.append(f'  d_ry: {dy:6.3f} deg')
                        text.append(f'  d_rz: {dz:6.3f} deg')
                        cam_move_1.update(rel_pose[0, 3]*1000.0, rel_pose[1, 3]*1000.0, rel_pose[2, 3]*1000.0, dx, dy, dz)

                    if len(mats) > 5:
                        rel_pose = np.matmul(np.linalg.inv(mats[-6]), mats[-1])
                        xyz = matrix_to_euler_angles(torch.from_numpy(rel_pose[:3, :3]), 'XYZ')

                        xyz_degree = xyz.detach() * 180.0 / np.math.pi
                        dx, dy, dz = xyz_degree[:]
                        text.append('to-prev-5:')
                        if not is_scannet:
                            text.append(f'  dt:   {datetimes[-1] - datetimes[-6]:6d} ms')
                        text.append(f'  d_t_xyz: ({rel_pose[0, 3]*1000.0:.3f}, {rel_pose[1, 3]*1000.0:.3f}, {rel_pose[2, 3]*1000.0:.3f})')
                        text.append(f'  d_r_xyz: ({dx:.3f}, {dy:.3f}, {dz:.3f})')
                        cam_move_5.update(rel_pose[0, 3]*1000.0, rel_pose[1, 3]*1000.0, rel_pose[2, 3]*1000.0, dx, dy, dz)

                    grid.subtext(id_row, id_col, text)
            else:
                continue
        # // for idx, (k, v) in enumerate(subdirs.items()):

        if not has_data:
            continue

        video_writer.write(grid.canvas)
        # if idx_frame > 300:
        #    break

    # // for idx_frame, item_name in enumerate(names):
    print(f'{cam_move_1}')
    print(f'{cam_move_5}')
    hole_info = HoleInfo(f'{osp.basename(rootdir)} invalid pixels', depth_h, depth_w, arr_pix_invalid)
    print(f'{hole_info}')

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
