import os
import os.path as osp
import sys
from sqlalchemy import case

from sympy import root
lib_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(lib_dir)

import time
import logging

from dro_sfm.utils.setup_log import setup_log


class DataPath:
    def __init__(self):
        self.root_dir = 'D:/slam/matterport'
        self.data_case = 'test/'
        self.split_info = 'splits/val_all_list.txt'

def load_path(name):
    root_dir = 'D:/slam/matterport'
    case_dir = osp.join(root_dir, 'test/matterport014_000')
    file_color = osp.join(case_dir, f'cam_left/{name}.jpg')
    file_depth = osp.join(case_dir, f'depth/{name}.png')
    file_pose = osp.join(case_dir, f'pose/{name}.txt')


def create_obj_cloud():
    color_path = 'test/matterport014_000/cam_left/000542628000000.jpg'
    depth_path = 'test/matterport014_000/depth/000542628000000.png'
    pose_path  = 'test/matterport014_000/pose/000542628000000.txt'


if __name__ == '__main__':
    setup_log('kneron_pointcloud.log')
    time_beg_pointcloud = time.time()

    time_end_pointcloud = time.time()
    print(f'pointcloud.py elapsed {time_end_pointcloud - time_beg_pointcloud:.6f} seconds.')
