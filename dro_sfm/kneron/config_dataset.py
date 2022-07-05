# -*- coding=utf-8 -*-

import os
import os.path as osp
import sys

from collections import OrderedDict
import copy
import datetime
import logging
import numpy as np
from PIL import Image
import subprocess
import time
import cv2

lib_dir = osp.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(lib_dir)

from dro_sfm.datasets.depth_filter import clip_depth
from dro_sfm.utils.depth import viz_inv_depth
from dro_sfm.utils.horovod import print0
from dro_sfm.utils.logging import pcolor
from dro_sfm.utils.setup_log import setup_log
from dro_sfm.visualization.viz_image_grid import VizImageGrid, Colors


def not_none(data):
    return not isinstance(data, type(None))


class KneronPose:
    def __init__(self, ts, px, py, pz, qx, qy, qz, qw):
        self.ts = ts
        self.px = px
        self.py = py
        self.pz = pz
        self.qx = qx
        self.qy = qy
        self.qz = qz
        self.qw = qw

        self.t = np.array([px, py, pz]).reshape((3, 1))

        i, j, k, r = qx, qy, qz, qw
        two_s = 2.0 / np.dot(np.array([r, i, j, k]), np.array([r, i, j, k]).transpose())
        self.R = np.array([
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j)
                ]).reshape((3, 3))

    def get_T(self):
        T = np.hstack((self.R, self.t))
        T_homogeneous = np.vstack((T, np.array([0, 0, 0, 1]).reshape((1, 4))))
        return T_homogeneous


class KneronFrame:
    def __init__(self, root_dir, name_color, name_depth, pose):
        self.root_dir = root_dir
        self.name_color = osp.join(self.root_dir, name_color)
        self.name_depth = osp.join(self.root_dir, name_depth)
        self.pose = pose

        self.data_depth = None
        self.param_px_valid = 0

    def load_depth(self):
        if not_none(self.data_depth):
            return
        self.data_depth = np.array(Image.open(self.name_depth), dtype=int)
        self.data_depth = None

    def apply_filter(self):
        self.load_depth()

    def synthetic_canvas(self, grid):
        color_png = cv2.imread(self.name_color)
        if sys.platform == 'win32':
            im_h, im_w, _ = color_png.shape
            if im_h != grid.cell_row or im_w != grid.cell_col:
                color_png = cv2.resize(color_png, (grid.cell_col, grid.cell_row))
        else:
            im_h, im_w, _ = color_png.shape
            if im_h != grid.cell_row or im_w != grid.cell_col:
                color_png = cv2.resize(color_png, (grid.cell_col, grid.cell_row))

        grid.subplot(0, 0, color_png, f'cam_left')

        depth_png = np.array(Image.open(self.name_depth), dtype=int)
        depth_vis = viz_inv_depth(depth_png.astype(np.float) / 1000.) * 255
        grid.subplot(1, 0, depth_vis[:, :, ::-1], f'depth')

        depth_png = clip_depth(depth_png)
        mask = depth_png <= 0
        depth_vis_clip = viz_inv_depth(depth_png.astype(np.float) / 1000.) * 255
        depth_vis_clip[mask, :] = 0
        grid.subplot(1, 1, depth_vis_clip[:, :, ::-1], f'depth clip')

        color_clip = copy.deepcopy(color_png)
        color_clip[mask, :] = 0
        color_clip = cv2.addWeighted(color_png, 0.25, color_clip, 0.75, 0)
        grid.subplot(0, 1, color_clip, f'cam_left clip')

        return grid


class KneronDataset:
    def __init__(self, root_dir, dataset_name, sub_dirs, file_gt_pose):
        logging.warning(f'KneronDataset::__init__({root_dir}, {dataset_name}, {sub_dirs}, {file_gt_pose})')
        self.root_dir = root_dir
        self.dataset_name = dataset_name

        self.video_name = osp.abspath(osp.join(root_dir, dataset_name, f'{osp.basename(dataset_name)}.avi'))
        print(f'self.video_name: {self.video_name}')
        if sys.platform == 'win32':
            self.demo_path = osp.join(lib_dir, '../demo/datasets')
        else:
            name_version = self.video_name.split('/')[-4]
            print(f'lib_dir: {lib_dir}')
            # self.demo_path = osp.join('/home/sigma/slam/demo/datasets', name_version)
            self.demo_path = osp.join(lib_dir, '../demo/datasets', name_version)
        if not osp.exists(self.demo_path):
            os.makedirs(self.demo_path)

        self.sub_dirs = sub_dirs
        self.file_gt_pose = file_gt_pose

        self.frames = None

        assert len(sub_dirs) == 2
        self.color_dir = None
        self.color_ext = None
        self.depth_dir = None
        self.depth_ext = None
        for str_dir, str_ext in sub_dirs:
            if str_dir == 'cam_left':
                self.color_dir, self.color_ext = str_dir, str_ext
            elif str_dir == 'depth':
                self.depth_dir, self.depth_ext = str_dir, str_ext

        assert not_none(self.color_dir) and not_none(self.color_ext) and \
            not_none(self.depth_dir) and not_none(self.depth_ext)

    def pipeline(self):
        self.check_data()
        self.load_data()
        self.filter_data()
        self.synthetic_video()
        self.align_pointcloud()

    @staticmethod
    def get_basenames(data_dir, file_ext):
        names = []
        for item in sorted(os.listdir(data_dir)):
            name, ext = osp.splitext(item)
            if ext.lower() == file_ext:
                names.append(name)
        return names

    @staticmethod
    def get_poses(filename):
        pose_data_dict = OrderedDict()
        pose_ts_set = set()
        with open(filename, 'r') as f_in:
            text = f_in.readlines()
            for line in text:
                line = line.strip()
                if len(line) < 1:
                    continue
                if line.startswith('#'):
                    continue
                if 'nan' in line:
                    continue
                words = line.split()
                if len(words) != 8:
                    print0(pcolor(f'unexpected format: {line}', 'blue'))
                params = [float(v) for v in words[1:]]
                px, py, pz, qx, qy, qz, qw = params
                time_stamp = words[0].zfill(15)
                pose_ts_set.add(time_stamp)
                pose_data_dict[time_stamp] = KneronPose(time_stamp, px, py, pz, qx, qy, qz, qw)
        assert len(pose_ts_set) == len(pose_data_dict)
        return pose_ts_set, pose_data_dict

    def check_data(self):
        logging.warning(f'check_data()')
        dataset_dir = osp.join(self.root_dir, self.dataset_name)
        print0(pcolor(f'===== {osp.basename(self.root_dir)}/{self.dataset_name} =====', 'yellow'))

        # check figures
        basenames = OrderedDict()
        for item_subdir, item_ext in self.sub_dirs:
            subdir = osp.join(dataset_dir, item_subdir)
            basenames[item_subdir] = self.get_basenames(subdir, item_ext)
        keys = list(basenames.keys())
        v_0 = basenames[keys[0]]
        for i in range(1, len(keys)):
            if basenames[keys[i]] != v_0:
                print0(pcolor(f'{len(v_0):4d} in {keys[0]} vs {len(basenames[keys[i]]):4d} in {keys[i]}', 'blue'))
                continue

        v_1 = basenames[keys[1]]
        if v_0 == v_1:
            names = v_0
        else:
            s_0 = set(v_0)
            s_1 = set(v_1)
            names = list(s_0 & s_1)

        # check pose
        pose_ts_set, _ = self.get_poses(osp.join(dataset_dir, self.file_gt_pose))
        n_missed_pose = len(names) - len(set(names) & pose_ts_set)

        print0(pcolor(f'{len(names):4d} frames in total', 'magenta'))
        print0(pcolor(f'{n_missed_pose:4d} frames without pose', 'magenta'))
        print0(pcolor(f'{len(names) - n_missed_pose:4d} valid frames', 'magenta'))

    def load_data(self):
        logging.warning(f'load_data()')
        if self.frames is not None:
            return

        dataset_dir = osp.join(self.root_dir, self.dataset_name)

        # load figure / depth names
        basenames = OrderedDict()
        for item_subdir, item_ext in self.sub_dirs:
            subdir = osp.join(dataset_dir, item_subdir)
            basenames[item_subdir] = self.get_basenames(subdir, item_ext)

        sub_dirs = list(basenames.keys())
        assert len(sub_dirs) >= 2
        basenames_common = set(basenames[sub_dirs[0]]) & set(basenames[sub_dirs[1]])
        for i in range(2, len(sub_dirs)):
            basenames_common = basenames_common & set(basenames[sub_dirs[i]])

        # load pose
        pose_ts_set, pose_data_dict = self.get_poses(osp.join(dataset_dir, self.file_gt_pose))
        basenames_common = basenames_common & pose_ts_set
        print0(pcolor(f'{len(basenames_common):4d} paired frames', 'green'))
        print0(pcolor(f'{len(pose_ts_set):4d} poses', 'blue'))

        ordered_basenames = sorted(list(basenames_common))
        self.frames = []
        for item in ordered_basenames:
            self.frames.append(
                KneronFrame(
                    dataset_dir,
                    osp.join(self.color_dir, f'{item}{self.color_ext}'),
                    osp.join(self.depth_dir, f'{item}{self.depth_ext}'),
                    pose_data_dict[item]))

    def filter_data(self):
        logging.warning(f'filter_data()')
        for item in self.frames:
            # item.apply_filter()
            pass
        pass

    def synthetic_video(self):
        logging.warning(f'systhetic_video()')

        im_h, im_w = 480, 640
        n_row, n_col = 2, 2
        grid = VizImageGrid(im_h, im_w, n_row, n_col)
        canvas_row = grid.canvas_row
        canvas_col = grid.canvas_col

        fps = 25.0
        if sys.platform == 'win32':
            fps = 1.0
        fps = 1.0

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(self.video_name, fourcc, fps, (canvas_col, canvas_row))
        print0(pcolor(f'  writing {self.video_name}', 'cyan'))
        for item in self.frames:
            grid = item.synthetic_canvas(grid)
            video_writer.write(grid.canvas)
        video_writer.release()

        if sys.platform != 'win32':
            subprocess.call(['cp', self.video_name, self.demo_path])
        else:
            cmd = f'xcopy /s /f /y /d /c /g {self.video_name} {osp.abspath(self.demo_path)}'
            print(cmd)
            os.system(cmd)

    def align_pointcloud(self):
        logging.warning(f'align_pointcloud()')
        pass


class KneronDatabase:
    def __init__(self, root_dir, dataset_names, sub_dirs, file_gt_pose):
        logging.warning(f'KneronDatabase::__init__(..)')
        self.root_dir = root_dir
        self.dataset_names = dataset_names
        self.sub_dirs = sub_dirs
        self.file_gt_pose = file_gt_pose
        self.datasets = None

    def run(self):
        for item in self.dataset_names:
            dataset = KneronDataset(self.root_dir, item, self.sub_dirs, self.file_gt_pose)
            dataset.pipeline()


def matterport0516():
    logging.warning(f'matterport0516()')
    root_dir = '/home/sigma/slam/matterport0516'
    dataset_names = [
        'test/matterport005_000_0610',
        'test/matterport014_000',
        'train_val_test/matterport005_000',
        'train_val_test/matterport005_001',
        'train_val_test/matterport010_000',
        'train_val_test/matterport010_001',
        ]
    sub_dirs = [('cam_left', '.jpg'), ('depth', '.png')]
    file_gt_pose = 'cam_pose.txt'
    db = KneronDatabase(root_dir, dataset_names, sub_dirs, file_gt_pose)
    db.run()


def matterport0516_ex():
    logging.warning(f'matterport0516_ex()')
    root_dir = '/home/sigma/slam/matterport0516_ex'
    dataset_names = [
        'test/matterport014_000_0516',
        'test/matterport014_001_0516',
        'train_val_test/matterport005_000_0516',
        'train_val_test/matterport005_001_0516',
        'train_val_test/matterport010_000_0516',
        'train_val_test/matterport010_001_0516',
        ]
    sub_dirs = [('cam_left', '.jpg'), ('depth', '.png')]
    file_gt_pose = 'cam_pose.txt'
    db = KneronDatabase(root_dir, dataset_names, sub_dirs, file_gt_pose)
    db.run()


def matterport0614():
    logging.warning(f'matterport0614()')
    root_dir = '/home/sigma/slam/matterport0614'
    dataset_names = [
        'tar.gz/matterport005_000_0516',
        'tar.gz/matterport005_001_0516',
        'tar.gz/matterport010_000_0516',
        'tar.gz/matterport010_001_0516',
        'tar.gz/matterport014_000_0516',
        'tar.gz/matterport014_001_0516',
        ]
    sub_dirs = [('cam_left', '.jpg'), ('depth', '.png')]
    file_gt_pose = 'cam_pose.txt'
    db = KneronDatabase(root_dir, dataset_names, sub_dirs, file_gt_pose)
    db.run()


def matterport0621():
    logging.warning(f'matterport0621()')
    root_dir = '/home/sigma/slam/matterport0621'
    dataset_names = [
        'tar.gz/matterport005_0621',
        'tar.gz/matterport005_0622',
        'tar.gz/matterport010_0621',
        'tar.gz/matterport010_0622',
        'tar.gz/matterport014_0622',
        'tar.gz/matterport047_0622',
        ]
    sub_dirs = [('cam_left', '.jpg'), ('depth', '.png')]
    file_gt_pose = 'cam_pose.txt'
    db = KneronDatabase(root_dir, dataset_names, sub_dirs, file_gt_pose)
    db.run()


def gazebo0629():
    logging.warning(f'gazebo0629()')
    root_dir = '/home/sigma/slam/gazebo0629'
    dataset_names = ['0628_2022_line_sim']
    sub_dirs = [('cam_left', '.jpg'), ('depth', '.png')]
    file_gt_pose = 'groundtruth.txt'
    db = KneronDatabase(root_dir, dataset_names, sub_dirs, file_gt_pose)
    db.run()


def main_tiny():
    logging.warning(f'main_tiny()')
    root_dir = '/home/sigma/slam/matterport0614'
    dataset_names = [
        'tar.gz/matterport005_000_0516',
        ]
    sub_dirs = [('cam_left', '.jpg'), ('depth', '.png')]
    file_gt_pose = 'cam_pose.txt'

    if sys.platform == 'win32':
        root_dir = 'D:/ssh.yangxl-2014-fe/gazebo_data'
        dataset_names = ['0701_2022_sim']
        file_gt_pose = 'groundtruth.txt'

    db = KneronDatabase(root_dir, dataset_names, sub_dirs, file_gt_pose)
    db.run()


def main():
    logging.warning(f'main()')
    matterport0516()
    matterport0516_ex()
    matterport0614()
    matterport0621()
    gazebo0629()


def main_ex():
    logging.warning(f'main_ex()')
    root_dir = '/datasets/gazebo'
    dataset_names = []

    for item in sorted(os.listdir(root_dir)):
        path = osp.join(root_dir, item)
        if osp.isfile(path):
            print(f'ignore {path}')
            continue

        '''
        for nested_item in os.listdir(path):
            nested_path = osp.join(path, nested_item)
            if osp.isfile(nested_path):
                print(f'ignore {nested_path}')
                continue
        '''
        extract_path = osp.join(path, 'extract')
        if not osp.exists(extract_path):
            continue

        if osp.isfile(extract_path):
            continue

        for nested_item in sorted(os.listdir(extract_path)):
            dataset_names.append(osp.join(item, 'extract', nested_item))

    print(f'{len(dataset_names)} datasets:')
    for idx, item in enumerate(dataset_names):
        print(f'  [{idx:2d}] {item}')

    sub_dirs = [('cam_left', '.jpg'), ('depth', '.png')]
    file_gt_pose = 'cam_pose.txt'
    db = KneronDatabase(root_dir, dataset_names, sub_dirs, file_gt_pose)
    db.run()


def main_latest():
    logging.warning(f'main_latest()')
    root_dir = '/home/sigma/slam/gazebo0629'
    dataset_name = ['0701_2022_sim']

    for item in dataset_name:
        data_dir = osp.join(root_dir, item)
        depth_dir = osp.join(data_dir, 'depth')
        depth_vis_dir = osp.join(data_dir, 'depth_vis')
        if not osp.exists(depth_dir):
            continue
        if not osp.exists(depth_vis_dir):
            os.mkdir(depth_vis_dir)
        for item_file in sorted(os.listdir(depth_dir)):
            path_depth = osp.join(depth_dir, item_file)
            path_depth_vis = osp.join(depth_vis_dir, item_file)
            depth_png = np.array(Image.open(path_depth), dtype=int)
            depth_vis = viz_inv_depth(depth_png.astype(np.float) / 1000.) * 255
            cv2.imwrite(path_depth_vis, depth_vis[:, :, ::-1])

    root_dir = '/home/sigma/slam/gazebo0629'
    dataset_names = ['0701_2022_sim']
    sub_dirs = [('cam_left', '.jpg'), ('depth', '.png')]
    file_gt_pose = 'groundtruth.txt'
    db = KneronDatabase(root_dir, dataset_names, sub_dirs, file_gt_pose)
    db.run()


if __name__ == '__main__':
    setup_log('kneron_config_dataset.log')
    time_beg_config_dataset = time.time()
    np.set_printoptions(precision=6, suppress=True)

    # main_tiny()
    # main()
    # main_ex()
    main_latest()


    time_end_config_dataset = time.time()
    logging.warning(f'config_dataset.py elapsed {time_end_config_dataset - time_beg_config_dataset:.6f} seconds.')
    print0(pcolor(f'config_dataset.py elapsed {time_end_config_dataset - time_beg_config_dataset:.6f} seconds.', 'yellow'))
