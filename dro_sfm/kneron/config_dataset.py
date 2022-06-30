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


class KneronDataset:
    def __init__(self, root_dir, dataset_names, sub_dirs, file_gt_pose):
        self.root_dir = root_dir
        self.dataset_names = dataset_names
        self.sub_dirs = sub_dirs
        self.file_gt_pose = file_gt_pose

    def pipeline(self):
        self.check_data()
        pass

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
                pose_ts_set.add(words[0])
                pose_data_dict[words[0]] = KneronPose(words[0], px, py, pz, qx, qy, qz, qw)
        return pose_ts_set, pose_data_dict


    def check_data(self):
        for item_dataset in self.dataset_names:
            dataset_dir = osp.join(self.root_dir, item_dataset)

            # check figures
            basenames = OrderedDict()
            for item_subdir, item_ext in self.sub_dirs:
                subdir = osp.join(dataset_dir, item_subdir)
                basenames[item_subdir] = self.get_basenames(subdir, item_ext)
            keys = list(basenames.keys())
            v_0 = basenames[keys[0]]
            for i in range(1, len(keys)-1):
                if basenames[keys[i]] != v_0:
                    raise ValueError

            # check pose
            pose_ts_set, pose_data_dict = self.get_poses(osp.join(dataset_dir, self.file_gt_pose))
            # pose_ts_set, pose_data_dict = self.get_poses('/home/sigma/slam/gazebo0629/groundtruth.txt')
            n_missed_pose = 0
            for ts in v_0:
                if ts not in pose_ts_set:
                    print0(pcolor(f'    missing pose of {ts}', 'cyan'))
                    n_missed_pose += 1
                    continue

            print0(pcolor(f'{len(v_0):4d} frames in total', 'magenta'))
            print0(pcolor(f'{n_missed_pose:4d} frames without pose', 'magenta'))

        pass

def main():
    root_dir = '/home/sigma/slam/gazebo0629'
    dataset_names = ['0628_2022_line_sim']
    sub_dirs = [('cam_left', '.jpg'), ('depth', '.png')]
    file_gt_pose = 'groundtruth.txt'
    ds = KneronDataset(root_dir, dataset_names, sub_dirs, file_gt_pose)
    ds.pipeline()


if __name__ == '__main__':
    setup_log('kneron_config_dataset.log')
    time_beg_config_dataset = time.time()
    np.set_printoptions(precision=6, suppress=True)

    main()

    time_end_config_dataset = time.time()
    logging.warning(f'config_dataset.py elapsed {time_end_config_dataset - time_beg_config_dataset:.6f} seconds.')
    print0(pcolor(f'config_dataset elapsed {time_end_config_dataset - time_beg_config_dataset:.6f} seconds.', 'yellow'))
