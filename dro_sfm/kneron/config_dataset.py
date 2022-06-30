# -*- coding=utf-8 -*-

import os
import os.path as osp
import sys
lib_dir = osp.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(lib_dir)

from collections import OrderedDict
import datetime
import logging
import numpy as np
import time

from dro_sfm.utils.setup_log import setup_log
from dro_sfm.utils.horovod import print0
from dro_sfm.utils.logging import pcolor

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
        self.name_color = name_color
        self.name_depth = name_depth
        self.pose = pose


class KneronDataset:
    def __init__(self, root_dir, dataset_name, sub_dirs, file_gt_pose):
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.sub_dirs = sub_dirs
        self.file_gt_pose = file_gt_pose
        self.frames = None
        assert len(sub_dirs) == 2

    def pipeline(self):
        self.check_data()
        self.load_data()
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
                print0(pcolor(f'  {len(v_0):4d} in {keys[0]} vs {len(basenames[keys[i]]):4d} in {keys[i]}', 'blue'))
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
        n_missed_pose = 0
        '''
        for ts in names:
            if ts not in pose_ts_set:
                # print0(pcolor(f'    missing pose of {ts}', 'cyan'))
                n_missed_pose += 1
                continue
        '''
        n_missed_pose = len(names) - len(set(names) & pose_ts_set)

        print0(pcolor(f'{len(v_0):4d} frames in total', 'magenta'))
        print0(pcolor(f'{n_missed_pose:4d} frames without pose', 'magenta'))
        print0(pcolor(f'{len(v_0) - n_missed_pose:4d} valid frames', 'magenta'))

    def load_data(self):
        if self.frames is not None:
            return

        dataset_dir = osp.join(self.root_dir, self.dataset_name)

        # load figure / depth names
        basenames = OrderedDict()
        for item_subdir, item_ext in self.sub_dirs:
            subdir = osp.join(dataset_dir, item_subdir)
            basenames[item_subdir] = self.get_basenames(subdir, item_ext)

    def synthetic_video(self):
        pass


    def align_pointcloud(self):
        pass


class KneronDatabase:
    def __init__(self, root_dir, dataset_names, sub_dirs, file_gt_pose):
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


def main():
    logging.warning(f'main()')
    matterport0516()
    matterport0516_ex()
    matterport0614()
    matterport0621()
    gazebo0629()


if __name__ == '__main__':
    setup_log('kneron_config_dataset.log')
    time_beg_config_dataset = time.time()
    np.set_printoptions(precision=6, suppress=True)

    main()

    time_end_config_dataset = time.time()
    logging.warning(f'config_dataset.py elapsed {time_end_config_dataset - time_beg_config_dataset:.6f} seconds.')
    print0(pcolor(f'config_dataset elapsed {time_end_config_dataset - time_beg_config_dataset:.6f} seconds.', 'yellow'))
