import os.path as osp
import sys
lib_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(lib_dir)
import logging
import numpy as np
import time
import torch

from dro_sfm.utils.horovod import print0
from dro_sfm.utils.logging import pcolor
from dro_sfm.utils.setup_log import setup_log
from dro_sfm.geometry.pose_trans import matrix_to_euler_angles
from dro_sfm.visualization.gazebo_config import GazeboPose


def get_frame_pose_v1(p, q):
    T_c2w = np.array([
            [ 0.0,  0.0, -1.0, 0.0],
            [ 1.0,  0.0,  0.0, 0.0],
            [ 0.0, -1.0,  0.0, 0.0],
            [ 0.0,  0.0,  0.0, 1.0]], dtype=float)
    T_r = GazeboPose(q[0], q[1], q[2], q[3], 0., 0., 0.).get_T()
    T_t = np.array([
        [ 1.,  0.,  0.,  p[0]],
        [ 0.,  1.,  0.,  p[1]],
        [ 0.,  0.,  1.,  p[2]],
        [ 0.,  0.,  0.,  1.]], dtype=np.float)
    inv_T_t = np.linalg.inv(T_t)

    return inv_T_t @ T_r @ T_c2w


def get_frame_pose_v2(p, q):
    T_c2w = np.array([
            [ 0.0,  0.0,  1.0, 0.0],
            [-1.0,  0.0,  0.0, 0.0],
            [ 0.0, -1.0,  0.0, 0.0],
            [ 0.0,  0.0,  0.0, 1.0]], dtype=float)
    T_r = GazeboPose(q[0], q[1], q[2], q[3], 0., 0., 0.).get_T()
    T_t = np.array([
        [ 1.,  0.,  0.,  p[0]],
        [ 0.,  1.,  0.,  p[1]],
        [ 0.,  0.,  1.,  p[2]],
        [ 0.,  0.,  0.,  1.]], dtype=np.float)

    return T_t @ T_r @ T_c2w


def get_relative_pose(T_i, T_j):
    return np.linalg.inv(T_i) @ T_j


def f(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))


def get_cloud():
    # https://www.tutorialspoint.com/matplotlib/matplotlib_3d_contour_plot.htm
    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)

    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    x_flatten = X.flatten()
    y_flatten = Y.flatten()
    z_flatten = Z.flatten()
    xyz = np.vstack((x_flatten, y_flatten, z_flatten))

    return xyz


if __name__ == '__main__':
    setup_log('kneron_check_matrix.log')

    q_i = [0.1, 0.2, 0.3, 0.4]
    q_j = [0.3, 0.7, 0.5, 0.2]

    p_i = [0.8, 2.3, 6.0]
    p_j = [6.3, 1.1, 8.0]

    T_i_v1 = get_frame_pose_v1(p_i, q_i)
    T_j_v1 = get_frame_pose_v1(p_j, q_j)

    T_i_v2 = get_frame_pose_v2(p_i, q_i)
    T_j_v2 = get_frame_pose_v2(p_j, q_j)

    T_v1 = get_relative_pose(T_i_v1, T_j_v1)
    T_v2 = get_relative_pose(T_i_v2, T_j_v2)

    euler_v1 = matrix_to_euler_angles(torch.from_numpy(T_v1[:3, :3]), 'XYZ').detach() * 180.0 / np.math.pi
    euler_v2 = matrix_to_euler_angles(torch.from_numpy(T_v2[:3, :3]), 'XYZ').detach() * 180.0 / np.math.pi

    print(f'=== v1: ===\nR: {euler_v1.numpy()} t: {T_v1[:3, 3]}\n')
    print(f'{T_v1}\n')
    print(f'=== v2: ===\nR: {euler_v2.numpy()} t: {T_v2[:3, 3]}\n')
    print(f'{T_v2}\n')

    # cloud = np.array(np.arange(0, 1500), dtype=float).reshape((-1, 3)).T
    # cloud = np.random.rand(3, 10000)
    cloud = get_cloud()

    # print(f'{cloud}')
    cloud_h = np.vstack((cloud, np.array([1.0] * cloud.shape[1], dtype=np.float)))
    print(f'\ncloud_h:\n{cloud_h}')

    # print(f'=== T_i_v1 ===\n{T_i_v1}')
    # print(f'=== T_i_v2 ===\n{T_i_v2}')
    # print(f'=== T_j_v1 ===\n{T_j_v1}')
    # print(f'=== T_j_v2 ===\n{T_j_v2}')

    cloud_h_v1 = T_v1 @ cloud_h
    cloud_h_v2 = T_v2 @ cloud_h

    print(f'\ncloud_h_v1:\n{cloud_h_v1[:3, :]}')
    print(f'\ncloud_h_v2:\n{cloud_h_v2[:3, :]}')

    data_dir = '/home/sigma/data'
    obj_init = osp.join(data_dir, 'cloud_h.obj')
    obj_v1 = osp.join(data_dir, 'cloud_h_v1.obj')
    obj_v2 = osp.join(data_dir, 'cloud_h_v2.obj')

    with open(obj_init, 'wt') as f_init, open(obj_v1, 'wt') as f_v1, open(obj_v2, 'wt') as f_v2:
        n = cloud_h.shape[1]
        for i in range(n):
            f_init.write(f'v {cloud_h[0, i]} {cloud_h[1, i]} {cloud_h[2, i]}\n')
            f_v1.write(f'v {cloud_h_v1[0, i]} {cloud_h_v1[1, i]} {cloud_h_v1[2, i]}\n')
            f_v2.write(f'v {cloud_h_v2[0, i]} {cloud_h_v2[1, i]} {cloud_h_v2[2, i]}\n')
    pass
