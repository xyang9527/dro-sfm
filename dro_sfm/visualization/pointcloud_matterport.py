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
from dro_sfm.visualization.gazebo_config import GazeboParam
from dro_sfm.visualization.pointcloud_downsample import generate_pointcloud_NxN

"""
/home/sigma/slam/matterport/test/matterport014_000
    cam_left/000542628000000.jpg   640x480
    cam_right/000542628000000.jpg  640x480
    depth/000542628000000.png      640x480
    pose/000542628000000.txt         4x4

    ./
        cam_delta_pose_Tij.txt      N1x(1+7)
        cam_delta_pose_Tji.txt      N1x(1+7)
        cam_pose.txt                N1x(1+7)
        groundtruth.txt             N2x(1+7)
        imu.txt                     N3x(1+10)
        odom.txt                    N4x(1+13)
"""

def load_matterport_depth(d_file):
    # ref: dro_sfm/utils/depth.py  def load_depth(file)
    depth_png = np.array(load_image(d_file), dtype=int)
    depth = depth_png.astype(np.float) / 1000.0
    depth[depth_png == 0] = -1.
    return depth


def load_data(names, data_dir):
    logging.warning(f'load_data(..)')
    n_frame = len(names)
    intr_color = np.array([[530.4669406576809,   0.0,             320.5, 0.0],
                           [0.0,               530.4669406576809, 240.5, 0.0],
                           [0.0,                 0.0,               1.0, 0.0],
                           [0.0,                 0.0,               0.0, 1.0]])
    fx = intr_color[0][0]
    fy = intr_color[1][1]
    cx = intr_color[0][2]
    cy = intr_color[1][2]

    sample_x, sample_y = 4, 4
    valid_only = True

    dir_root = data_dir
    dir_cloud_ply_camera_coord = osp.join(dir_root, 'demo/ply_camera_coord')
    dir_cloud_obj_camera_coord = osp.join(dir_root, 'demo/obj_camera_coord')
    dir_cloud_obj_world_coord = osp.join(dir_root, 'demo/obj_world_coord')
    dir_cloud_jpg = osp.join(dir_root, 'demo/jpg')

    dir_cloud_ply_camera_coord_downsample = osp.join(dir_root, f'demo/ply_camera_coord_downsample_{sample_x}x{sample_y}')
    dir_cloud_obj_camera_coord_downsample = osp.join(dir_root, f'demo/obj_camera_coord_downsample_{sample_x}x{sample_y}')

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

    '''
    T_coord_swap = np.array([[ 0.,  0.,  1.,  0.],
                             [ 1.,  0.,  0.,  0.],
                             [ 0.,  1.,  0.,  0.],
                             [ 0.,  0.,  0.,  1.]], dtype=np.float)
    T_combination = []
    for i in range(2):
        neg_x = 1.0 if i % 2 == 0 else -1.0
        T_x = copy.deepcopy(T_coord_swap)
        T_x[0, :] *= neg_x
        for j in range(2):
            neg_y = 1.0 if j % 2 == 0 else -1.0
            T_y = copy.deepcopy(T_x)
            T_y[1, :] *= neg_y
            for k in range(2):
                neg_z = 1.0 if k % 2 == 0 else -1.0
                T_z = copy.deepcopy(T_y)
                T_z[2, :] *= neg_z
                T_combination.append(T_z)
    for i in range(len(T_combination)):
        print(f"\nT_combination[{i:02d}]:\n{T_combination[i]}")
    '''

    T05 = np.array([[ 0.,  0., -1.,  0.],
                    [ 1.,  0.,  0.,  0.],
                    [ 0., -1.,  0.,  0.],
                    [ 0.,  0.,  0.,  1.]], dtype=np.float)
    T05_inv = np.linalg.inv(T05)

    pose_init = None
    pose_init_world_coord = None
    for idx_f in range(n_frame):
        name = names[idx_f]
        print(f'  process frame: [{idx_f:6d}] {name} ..')

        data_color = load_image(osp.join(dir_root, f'cam_left/{name}.jpg'))
        data_depth = load_matterport_depth(osp.join(dir_root, f'depth/{name}.png'))
        data_pose = np.genfromtxt(osp.join(dir_root, f'pose/{name}.txt'))
        data_pose_world_coord = np.genfromtxt(osp.join(dir_root, f'pose_world_coord/{name}.txt'))

        if sys.platform != 'win32':
            subprocess.call(['cp', osp.join(dir_root, f'cam_left/{name}.jpg'), osp.join(dir_cloud_jpg, f'{name}.jpg')])
        else:
            src_file = osp.join(dir_root, f'cam_left/{name}.jpg')
            dst_file = osp.join(dir_cloud_jpg, f'{name}.jpg')
            os.system(f'xcopy /s {src_file} {dst_file}')

        if idx_f == 0:
            pose_init = data_pose
            pose_init_world_coord = data_pose_world_coord

        file_cloud_ply = osp.join(dir_cloud_ply_camera_coord, f'{name}.ply')
        file_cloud_ply_downsample = osp.join(dir_cloud_ply_camera_coord_downsample, f'{name}.ply')
        data_depth_resized = cv2.resize(data_depth, data_color.size, interpolation = cv2.INTER_NEAREST)

        cloud = generate_pointcloud(
            np.array(data_color, dtype=int), data_depth_resized, fx, fy, cx, cy,
            file_cloud_ply, 1.0)
        cloud_downsample = generate_pointcloud_NxN(
            np.array(data_color, dtype=int), data_depth_resized, fx, fy, cx, cy,
            file_cloud_ply_downsample, sample_x, sample_y, valid_only, 1.0)

        rel_pose = np.matmul(np.linalg.inv(pose_init), data_pose)

        # initial point cloud in camera coord
        cloud_xyz = cloud[:, :3]
        cloud_rgb = cloud[:, 3:]
        cloud_xyz_hom = np.transpose(np.hstack((cloud_xyz, np.ones((cloud_xyz.shape[0], 1)))))
        cloud_xyz_align = np.dot(rel_pose, cloud_xyz_hom)
        cloud_xyz_align_t = np.transpose(cloud_xyz_align)

        with open(osp.join(dir_cloud_obj_camera_coord, f'camera_coord_pose_T05_{name}.obj'), 'w') as f_ou_align_rgb:
            n_vert = cloud_xyz.shape[0]
            for i in range(n_vert):
                x, y, z, w = cloud_xyz_align_t[i]
                r, g, b = cloud_rgb[i]
                f_ou_align_rgb.write(f'v {x} {y} {z} {r} {g} {b}\n')

        # downsampled point cloud in camera coord
        cloud_xyz_downsample = cloud_downsample[:, :3]
        cloud_rgb_downsample = cloud_downsample[:, 3:]
        cloud_xyz_hom_downsample = np.transpose(np.hstack((cloud_xyz_downsample, np.ones((cloud_xyz_downsample.shape[0], 1)))))
        cloud_xyz_align_downsample = np.dot(rel_pose, cloud_xyz_hom_downsample)
        cloud_xyz_align_t_downsample = np.transpose(cloud_xyz_align_downsample)

        with open(osp.join(dir_cloud_obj_camera_coord_downsample, f'camera_coord_pose_T05_{name}.obj'), 'w') as f_ou_align_rgb_downsample:
            n_vert_downsample = cloud_xyz_downsample.shape[0]
            for i in range(n_vert_downsample):
                x, y, z, w = cloud_xyz_align_t_downsample[i]
                r, g, b = cloud_rgb_downsample[i]
                f_ou_align_rgb_downsample.write(f'v {x} {y} {z} {r} {g} {b}\n')

        # point cloud in world coord
        with open(osp.join(dir_cloud_obj_world_coord, f'world_coord_pose_T05_{name}.obj'), 'w') as f_ou_align_rgb:
            n_vert = cloud_xyz.shape[0]
            cloud_xyz_temp = np.transpose(np.dot(data_pose_world_coord, np.dot(T05, cloud_xyz_hom)))
            for i in range(n_vert):
                x, y, z, w = cloud_xyz_temp[i]
                r, g, b = cloud_rgb[i]
                f_ou_align_rgb.write(f'v {x} {y} {z} {r} {g} {b}\n')

        # ==================================================================== #
        # check np.dot(np.matmul(A, B), C) == np.dot(A, np.dot(B, C))
        rel_pose_world_coord = np.matmul(np.linalg.inv(pose_init_world_coord), data_pose_world_coord)
        cloud_xyz_align_world_coord = np.dot(T05_inv, np.dot(rel_pose_world_coord, np.dot(T05, cloud_xyz_hom)))
        is_same = np.allclose(cloud_xyz_align, cloud_xyz_align_world_coord)
        mean_diff = np.mean(cloud_xyz_align - cloud_xyz_align_world_coord)
        if not is_same:
            print(f'    is_same:    {is_same},   mean_diff:    {mean_diff}')

        cloud_xyz_align_A = np.dot(data_pose, cloud_xyz_hom)
        cloud_xyz_align_B = np.dot(data_pose_world_coord, np.dot(T05, cloud_xyz_hom))
        is_same_AB = np.allclose(cloud_xyz_align_A, cloud_xyz_align_B)
        mean_diff_AB = np.mean(cloud_xyz_align_A - cloud_xyz_align_B)
        if not is_same_AB:
            print(f'    is_same_AB: {is_same_AB},   mean_diff_AB: {mean_diff_AB}')
        # ==================================================================== #


def create_obj_cloud():
    data_cols = [{'dir': '/home/sigma/slam/matterport/test/matterport014_000', 'space': 100},
                 {'dir': '/home/sigma/slam/matterport/test/matterport014_000_0601', 'space': 5}]
    data_cols = [{'dir': '/home/sigma/slam/matterport/test/matterport014_000', 'space': 100}]
    for item_data in data_cols:
        data_dir = item_data['dir']
        space = item_data['space']
        names = []
        for item in sorted(os.listdir(osp.join(data_dir, 'pose'))):
            names.append(osp.splitext(item)[0])
        load_data(names[::space], data_dir)


if __name__ == '__main__':
    setup_log('kneron_pointcloud_matterport.log')
    time_beg_pointcloud = time.time()

    np.set_printoptions(precision=6, suppress=True)
    create_obj_cloud()

    time_end_pointcloud = time.time()
    print(f'pointcloud.py elapsed {time_end_pointcloud - time_beg_pointcloud:.6f} seconds.')
