import os
import os.path as osp
import sys
lib_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(lib_dir)
import time
import logging
import numpy as np
import subprocess

from dro_sfm.utils.setup_log import setup_log
from dro_sfm.utils.depth import load_depth
from dro_sfm.utils.image import load_image
from scripts.infer import generate_pointcloud


def load_path(name):
    root_dir = '/home/sigma/slam/matterport'
    case_dir = osp.join(root_dir, 'test/matterport014_000')
    file_color = osp.join(case_dir, f'cam_left/{name}.jpg')
    file_depth = osp.join(case_dir, f'depth/{name}.png')
    file_pose = osp.join(case_dir, f'pose/{name}.txt')
    file_cloud_obj = osp.join(case_dir, f'demo/cloud_obj/{name}.obj')
    file_cloud_ply = osp.join(case_dir, f'demo/cloud_ply/{name}.ply')

    files = [file_color, file_depth, file_pose]

    for item in files:
        if not osp.exists(item):
            logging.critical(f'file not exist {item}')
            raise ValueError(f'file not exist {item}')

    if not osp.exists(osp.dirname(file_cloud_obj)):
        os.makedirs(osp.dirname(file_cloud_obj))
    if not osp.exists(osp.dirname(file_cloud_ply)):
        os.makedirs(osp.dirname(file_cloud_ply))

    subprocess.call(['cp', file_color, file_color.replace('cam_left', 'demo')])
    subprocess.call(['cp', file_depth, file_depth.replace('depth', 'demo')])
    subprocess.call(['cp', file_pose, file_pose.replace('pose', 'demo')])

    data_depth = load_depth(file_depth)
    data_image = np.array(load_image(file_color), dtype=int)
    logging.info(f'  data_depth: {type(data_depth)} {data_depth.shape} {data_depth.dtype}')
    logging.info(f'  data_image: {type(data_image)} {data_image.shape} {data_image.dtype}')

    h, w = data_depth.shape
    cx, cy = 0.5*w, 0.5*h
    fx, fy = 577.870605, 577.870605

    cloud = generate_pointcloud(data_image, data_depth, fx, fy, cx, cy, file_cloud_ply, 1.0)
    logging.info(f'  cloud: {type(cloud)} {cloud.shape}')

    cloud_xyz = cloud[:, :, :3]
    cloud_rgb = cloud[:, :, 3:]
    data_pose = np.genfromtxt(file_pose)
    print(f'pose:\n{data_pose}')

    '''

    '''



    pass


def create_obj_cloud():
    load_path('000542628000000')
    load_path('000543976000000')
    pass


if __name__ == '__main__':
    setup_log('kneron_pointcloud.log')
    time_beg_pointcloud = time.time()

    create_obj_cloud()

    time_end_pointcloud = time.time()
    print(f'pointcloud.py elapsed {time_end_pointcloud - time_beg_pointcloud:.6f} seconds.')
