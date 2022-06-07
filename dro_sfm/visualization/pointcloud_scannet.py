# -*- coding=utf-8 -*-

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
from scripts.infer import generate_pointcloud
from dro_sfm.visualization.pointcloud_downsample import generate_pointcloud_NxN

"""
/home/sigma/slam/scannet_train_data/scene0000_00
    color/000000.jpg        1296x968
    depth/000000.png         640x480
    pose/000000.txt            4x4
    intrinsic/
        extrinsic_color.txt    4x4
        extrinsic_depth.txt    4x4
        intrinsic_color.txt    4x4
        intrinsic_depth.txt    4x4
"""

def load_scannet_depth(file):
    # ref: def read_png_depth(file) @ dro_sfm/datasets/scannet_dataset.py
    depth_png = np.array(load_image(file), dtype=int)
    depth = depth_png.astype(np.float) / 1000.0
    depth[depth_png == 0] = -1.
    return depth


def create_obj_cloud():
    logging.warning(f'create_obj_cloud()')
    dir_root = '/home/sigma/slam/scannet_train_data/scene0000_00'
    n = len(os.listdir(osp.join(dir_root, 'color')))
    logging.info(f'  {n} files in {dir_root}')

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

    sample_x, sample_y = 4, 4
    valid_only = True

    dir_cloud_ply_camera_coord = osp.join(dir_root, 'demo/unaligned_ply_camera_coord')
    dir_cloud_obj_camera_coord = osp.join(dir_root, 'demo/aligned_obj_camera_coord')
    dir_cloud_obj_world_coord = osp.join(dir_root, 'demo/aligned_obj_world_coord')
    dir_cloud_jpg = osp.join(dir_root, 'demo/jpg')

    dir_cloud_ply_camera_coord_downsample = osp.join(dir_root, f'demo/unaligned_ply_camera_coord_downsample_{sample_x}x{sample_y}')
    dir_cloud_obj_camera_coord_downsample = osp.join(dir_root, f'demo/aligned_obj_camera_coord_downsample_{sample_x}x{sample_y}')

    folders_need = [
        dir_cloud_ply_camera_coord,
        dir_cloud_obj_camera_coord,
        dir_cloud_obj_world_coord,
        dir_cloud_jpg,
        dir_cloud_ply_camera_coord_downsample,
        dir_cloud_obj_camera_coord_downsample
        ]
    for item_dir in folders_need:
        if not osp.exists(item_dir):
            os.makedirs(item_dir)

    pose_init = None
    cam_traj_file_camera_coord = osp.join(dir_root, 'traj_cam_pose_camera_coord.obj')
    cam_traj_file_world_coord = osp.join(dir_root, 'traj_cam_pose_world_coord.obj')

    with open(cam_traj_file_camera_coord, 'w') as f_ou_traj_camera_coord, \
        open(cam_traj_file_world_coord, 'w') as f_ou_traj_world_coord:

        n_valid = 0

        for idx_f in range(0, 1000, 3):
            name = f'{idx_f:06d}'
            print(f'  process frame: {name} ..')

            data_color = load_image(osp.join(dir_root, f'color/{name}.jpg'))
            data_depth = load_scannet_depth(osp.join(dir_root, f'depth/{name}.png'))
            data_pose = np.genfromtxt(osp.join(dir_root, f'pose/{name}.txt'))

            subprocess.call(['cp', osp.join(dir_root, f'color/{name}.jpg'), osp.join(dir_cloud_jpg, f'{name}.jpg')])
            if idx_f == 0:
                pose_init = data_pose

            file_cloud_ply = osp.join(dir_cloud_ply_camera_coord, f'{name}.ply')
            file_cloud_ply_downsample = osp.join(dir_cloud_ply_camera_coord_downsample, f'{name}.ply')
            data_depth_resized = cv2.resize(data_depth, data_color.size, interpolation = cv2.INTER_NEAREST)

            cloud = generate_pointcloud(
                np.array(data_color, dtype=int), data_depth_resized, fx, fy, cx, cy,
                file_cloud_ply, 1.0)
            cloud_downsample = generate_pointcloud_NxN(
                np.array(data_color, dtype=int), data_depth_resized, fx, fy, cx, cy,
                file_cloud_ply_downsample, sample_x, sample_y, valid_only, 1.0)

            # rel_pose = np.matmul(pose_init, np.linalg.inv(data_pose)) # v1
            rel_pose = np.matmul(np.linalg.inv(pose_init), data_pose) # v2

            n_valid += 1
            f_ou_traj_world_coord.write(f'v {data_pose[0][3]} {data_pose[1][3]} {data_pose[2][3]}\n')
            f_ou_traj_camera_coord.write(f'v {rel_pose[0][3]} {rel_pose[1][3]} {rel_pose[2][3]}\n')

            cloud_xyz = cloud[:, :3]
            cloud_rgb = cloud[:, 3:]
            cloud_xyz_hom = np.transpose(np.hstack((cloud_xyz, np.ones((cloud_xyz.shape[0], 1)))))
            cloud_xyz_align = np.dot(rel_pose, cloud_xyz_hom)
            cloud_xyz_align_t = np.transpose(cloud_xyz_align)

            with open(osp.join(dir_cloud_obj_camera_coord, f'camera_coord_{name}.obj'), 'w') as f_ou_align_rgb:
                n_vert = cloud_xyz.shape[0]
                for i in range(n_vert):
                    x, y, z, w = cloud_xyz_align_t[i]
                    r, g, b = cloud_rgb[i]
                    f_ou_align_rgb.write(f'v {x} {y} {z} {r} {g} {b}\n')

            # downsampled point cloud
            cloud_xyz_downsample = cloud_downsample[:, :3]
            cloud_rgb_downsample = cloud_downsample[:, 3:]
            cloud_xyz_hom_downsample = np.transpose(np.hstack((cloud_xyz_downsample, np.ones((cloud_xyz_downsample.shape[0], 1)))))
            cloud_xyz_align_downsample = np.dot(rel_pose, cloud_xyz_hom_downsample)
            cloud_xyz_align_t_downsample = np.transpose(cloud_xyz_align_downsample)

            with open(osp.join(dir_cloud_obj_camera_coord_downsample, f'camera_coord_{name}.obj'), 'w') as f_ou_align_rgb_downsample:
                n_vert_downsample = cloud_xyz_downsample.shape[0]
                for i in range(n_vert_downsample):
                    x, y, z, w = cloud_xyz_align_t_downsample[i]
                    r, g, b = cloud_rgb_downsample[i]
                    f_ou_align_rgb_downsample.write(f'v {x} {y} {z} {r} {g} {b}\n')

            # point cloud in world coord
            with open(osp.join(dir_cloud_obj_world_coord, f'world_coord_{name}.obj'), 'w') as f_ou_align_rgb:
                n_vert = cloud_xyz.shape[0]
                cloud_xyz_temp = np.transpose(np.dot(data_pose, cloud_xyz_hom))
                for i in range(n_vert):
                    x, y, z, w = cloud_xyz_temp[i]
                    r, g, b = cloud_rgb[i]
                    f_ou_align_rgb.write(f'v {x} {y} {z} {r} {g} {b}\n')

        for idx_p in range(1, n_valid-1, 2):
            f_ou_traj_world_coord.write(f'f {idx_p} {idx_p+1} {idx_p+2}\n')
            f_ou_traj_camera_coord.write(f'f {idx_p} {idx_p+1} {idx_p+2}\n')


if __name__ == '__main__':
    setup_log('kneron_pointcloud_scannet.log')
    time_beg_pointcloud = time.time()

    np.set_printoptions(precision=6, suppress=True)
    create_obj_cloud()

    time_end_pointcloud = time.time()
    print(f'pointcloud_scannet.py elapsed {time_end_pointcloud - time_beg_pointcloud:.6f} seconds.')
    logging.info(f'pointcloud_scannet.py elapsed {time_end_pointcloud - time_beg_pointcloud:.6f} seconds.')
