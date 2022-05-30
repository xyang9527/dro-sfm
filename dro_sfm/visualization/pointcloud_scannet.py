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
#from dro_sfm.utils.depth import load_depth
from dro_sfm.utils.image import load_image
from scripts.infer import generate_pointcloud

"""
/home/sigma/slam/scannet_train_data/scene0000_00
    color/000000.jpg 1296x968
    depth/000000.png 640x480

"""


def load_scannet_depth(file):
    depth_png = np.array(load_image(file), dtype=int)
    depth = depth_png.astype(np.float) / 1000.0
    depth[depth_png == 0] = -1.
    # return np.expand_dims(depth, axis=2)
    return depth


def create_obj_cloud():
    dir_root = '/home/sigma/slam/scannet_train_data/scene0000_00'
    n = len(os.listdir(osp.join(dir_root, 'color')))

    extr_color = np.genfromtxt(osp.join(dir_root, 'intrinsic/extrinsic_color.txt'))
    extr_depth = np.genfromtxt(osp.join(dir_root, 'intrinsic/extrinsic_depth.txt'))
    intr_color = np.genfromtxt(osp.join(dir_root, 'intrinsic/intrinsic_color.txt'))
    intr_depth = np.genfromtxt(osp.join(dir_root, 'intrinsic/intrinsic_depth.txt'))

    logging.info(f'extr_color:\n{extr_color}\n')
    logging.info(f'extr_depth:\n{extr_depth}\n')
    logging.info(f'intr_color:\n{intr_color}\n')
    logging.info(f'intr_depth:\n{intr_depth}\n')

    fx = intr_color[0][0]
    fy = intr_color[1][1]
    cx = intr_color[0][2]
    cy = intr_color[1][2]

    dir_cloud_ply = osp.join(dir_root, 'demo/ply')
    if not osp.exists(dir_cloud_ply):
        os.makedirs(dir_cloud_ply)
    dir_cloud_obj = osp.join(dir_root, 'demo/obj')
    if not osp.exists(dir_cloud_obj):
        os.makedirs(dir_cloud_obj)

    pose_init = None

    for idx_f in range(0, 20, 5):
        name = f'{idx_f:06d}'
        data_color = load_image(osp.join(dir_root, f'color/{name}.jpg'))
        data_depth = load_scannet_depth(osp.join(dir_root, f'depth/{name}.png'))
        data_pose = np.genfromtxt(osp.join(dir_root, f'pose/{name}.txt'))

        if idx_f == 0:
            pose_init = data_pose

        file_cloud_ply = osp.join(dir_cloud_ply, f'{name}.ply')
        data_depth_resized = cv2.resize(data_depth, data_color.size, interpolation = cv2.INTER_NEAREST)
        cloud = generate_pointcloud(np.array(data_color, dtype=int), data_depth_resized, fx, fy, cx, cy, file_cloud_ply, 1.0, True)

        if idx_f >= 0:
            rel_pose = np.matmul(pose_init, np.linalg.inv(data_pose)) # v1
            rel_pose = np.matmul(np.linalg.inv(pose_init), data_pose) # v2

            cloud_xyz = cloud[:, :3]
            cloud_rgb = cloud[:, 3:]
            # cloud_xyz = cloud_xyz.reshape((-1, 3))
            # cloud_rgb = cloud_rgb.reshape((-1, 3))

            n = cloud_xyz.shape[0]
            cloud_xyz_hom = np.transpose(np.hstack((cloud_xyz, np.ones((n, 1)))))
            cloud_xyz_align = np.dot(rel_pose, cloud_xyz_hom)
            cloud_xyz_align_t = np.transpose(cloud_xyz_align)

            with open(osp.join(dir_cloud_obj, f'v2_{name}.obj'), 'w') as f_ou_align_rgb:
                for i in range(n):
                    x, y, z, w = cloud_xyz_align_t[i]
                    r, g, b = cloud_rgb[i]
                    f_ou_align_rgb.write(f'v {x} {y} {z} {r} {g} {b}\n')
            pass

    pass

if __name__ == '__main__':
    setup_log('kneron_pointcloud_scannet.log')
    time_beg_pointcloud = time.time()

    np.set_printoptions(precision=6, suppress=True)
    create_obj_cloud()

    time_end_pointcloud = time.time()
    print(f'pointcloud_scannet.py elapsed {time_end_pointcloud - time_beg_pointcloud:.6f} seconds.')
