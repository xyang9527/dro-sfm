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


def load_path(namelist):
    data_col = []
    for name in namelist:
        dir_root = '/home/sigma/slam/matterport'
        dir_case = osp.join(dir_root, 'test/matterport014_000')
        file_color = osp.join(dir_case, f'cam_left/{name}.jpg')
        file_depth = osp.join(dir_case, f'depth/{name}.png')
        file_pose = osp.join(dir_case, f'pose/{name}.txt')
        file_cloud_obj = osp.join(dir_case, f'demo/cloud_obj/{name}.obj')
        file_cloud_ply = osp.join(dir_case, f'demo/cloud_ply/{name}.ply')

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

        data_col.append({'cloud': cloud, 'pose': data_pose})

    data_save_dir = '/home/sigma/slam/matterport/test/matterport014_000/demo/aligned'
    if not osp.exists(data_save_dir):
        os.makedirs(data_save_dir)

    pose_init = data_col[0]['pose']
    # rel_poses = [np.matmul(np.linalg.inv(x), pose).astype(np.float32) for x in context_poses]
    n_case = len(data_col)
    for idx_c in range(n_case):
        logging.info(f'  idx_c: {idx_c}')
        pose_curr = data_col[idx_c]['pose']
        logging.info(f'  pose_curr:  {type(pose_curr)}')
        logging.info(f'  pose_init:  {type(pose_init)}')
        rel_pose = np.matmul(np.linalg.inv(pose_init), pose_curr).astype(np.float32)

        cloud = data_col[idx_c]['cloud']
        logging.info(f'  cloud: {cloud.shape}')
        cloud_xyz = cloud[:, :, :3]
        cloud_rgb = cloud[:, :, 3:]
        h, w, c = cloud_xyz.shape
        cloud_xyz = cloud_xyz.reshape((-1, 3))
        cloud_rgb = cloud_rgb.reshape((-1, 3))
        logging.info(f'  cloud_xyz: {cloud_xyz.shape}')
        logging.info(f'{cloud_xyz[:10]}')

        n = cloud_xyz.shape[0]
        cloud_xyz_hom = np.hstack((cloud_xyz, np.ones((n, 1))))
        print(f'  cloud_xyz_hom:   {cloud_xyz_hom.shape}')
        print(f'  rel_pose:\n{rel_pose}')

        new_cloud = np.dot(cloud_xyz_hom, np.transpose(rel_pose))

        with open(osp.join(data_save_dir, f'aligned_{idx_c:04d}.obj'), 'wt') as f_ou:
            for i in range(n):
                x, y, z, w = new_cloud[i]
                r, g, b = cloud_rgb[i]
                f_ou.write(f'v {x} {y} {z} {r} {g} {b}\n')
    '''
    '''

    pass


def create_obj_cloud():
    name_list = ['000542628000000', '000543976000000']
    load_path(name_list)
    pass


if __name__ == '__main__':
    setup_log('kneron_pointcloud.log')
    time_beg_pointcloud = time.time()

    create_obj_cloud()

    time_end_pointcloud = time.time()
    print(f'pointcloud.py elapsed {time_end_pointcloud - time_beg_pointcloud:.6f} seconds.')
