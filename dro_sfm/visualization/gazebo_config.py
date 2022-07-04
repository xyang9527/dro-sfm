# -*- coding=utf-8 -*-

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


'''
  root@Notebook-PC:/work/gazebo_robomaker# ./roboMaker.sh --info
  RoboMaker Tools
          --install
          --clean
          --info
          --run      [app_name]
                      app_name:
                        bookstore,hospital,small_house,small_warehouse,
                        matterport005,matterport010,matterport014,matterport047,matterport063,matterport071,
  =============================

  Camera intrinsics
          fx: 530.4669406576809, cx: 320.5
          fy: 530.4669406576809, cy: 240.5

  From Camera0 to Camera1
          Tcoi q: [ 0, 0, 0, 1 ]
               p: [ 0, 0.07, 0 ]

  From Camera0 to IMU
          Tcoi q: [ 0, 0, 0, 1 ]
               p: [ -0.076, -0.000, -0.025 ]

  From IMU to Odometry
          Tio  q: [ 0, 0, 0, 1 ]
               p: [ 0, 0, -0.045 ]

  From IMU to Groundtruth
          Tig  q: [ 0, 0, 0, 1 ]
               p: [ 0, 0, -0.068 ]


https://gazebosim.org/api/gazebo/6.0/spherical_coordinates.html
  Coordinates for the world origin

        gazebo world                              camera
                                                     __________________ X
           | Z                                       |\
           |                                         | \
           |                                         |  \
           |                                         |   \
           |______________                           |    \
           /               Y                         |     \
          /                                          |      \
         /                                           |       \
        /                                            |        \ Z (forward [into the screen])
       /                                             | Y
       X

          X ---------------------------------------  -Z
          Y ---------------------------------------   X
          Z ---------------------------------------  -Y

cam_to_gazebo_world (T05)

        0  0 -1  0
        1  0  0  0
        0 -1  0  0
        0  0  0  1


========================================================================================================================
https://classic.gazebosim.org/tutorials?tut=import_mesh
  This tutorial describes how to import 3D meshes into Gazebo.
  Gazebo uses a right-hand coordinate system where +Z is up (vertical), +X is forward (into the screen), and +Y is to the left.

                 gazebo world                       camera

    [into the screen]
            X                                          ___________ X
              \      | Z                               |\
               \     |                                 | \
                \    |                                 |  \
                 \   |                                 |   \
                  \  |                                 |    \
                   \ |                                 |     \
    Y ______________\|                                 |      \  Z (forward [into the screen])
                                                       | Y

            X ----------------------------------------  Z
            Y ---------------------------------------- -X
            Z ---------------------------------------- -Y

cam_to_gazebo_world

        0  0  1  0
       -1  0  0  0
        0 -1  0  0
        0  0  0  1

'''


class GazeboPose:
    def __init__(self, qx, qy, qz, qw, px, py, pz):
        i, j, k, r = qx, qy, qz, qw
        # r, i, j, k = qx, qy, qz, qw
        print(f'\n{r} {i} {j} {k}')
        two_s = 2.0 / np.dot(np.array([r, i, j, k]), np.array([r, i, j, k]).transpose())
        logging.warning(f'  two_s: {two_s:.6f}')

        # References:
        #     https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
        #         Conversion Quaternion to Matrix
        #     def quaternion_to_matrix(quaternions) @ dro_sfm/geometry/pose_trans.py
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
        self.t = np.array([px, py, pz]).reshape((3, 1))

    def get_T(self):
        T = np.hstack((self.R, self.t))
        T_homogeneous = np.vstack((T, np.array([0, 0, 0, 1]).reshape((1, 4))))
        return T_homogeneous


class GazeboParam:
    def __init__(self):
        self.cam2imu = GazeboPose(0, 0, 0, 1, -0.076, -0.0, -0.025).get_T()
        self.imu2gt = GazeboPose(0, 0, 0, 1, 0, 0, -0.068).get_T()

        self.cam2gt = np.matmul(self.imu2gt, self.cam2imu)
        self.gt2cam = np.linalg.inv(self.cam2gt)

        self.cam2gazebo = np.array([
            [ 0.0,  0.0, -1.0, 0.0],
            [ 1.0,  0.0,  0.0, 0.0],
            [ 0.0, -1.0,  0.0, 0.0],
            [ 0.0,  0.0,  0.0, 1.0]], dtype=np.float)

        logging.info(f'\n========== GazeboParam ==========')
        logging.info(f'  cam2imu:\n{self.cam2imu}\n')
        logging.info(f'  imu2gt: \n{self.imu2gt}\n')
        logging.info(f'  cam2gt: \n{self.cam2gt}\n')
        logging.info(f'  gt2cam: \n{self.gt2cam}\n')

    @property
    def get_cam2gt(self):
        return self.cam2gt

    @property
    def get_gt2cam(self):
        return self.gt2cam

    @property
    def get_cam2gazebo(self):
        return self.cam2gazebo


def kneron_print(text):
    logging.info(text)
    print(text)


def check_matrix_v1():
    # weita   x: -0.5, y: -0.5, z: 0.5, w: 0.5
    # bodong: -0.5 0.5 -0.5 0.5
    T_weita = GazeboPose(-0.5, -0.5, 0.5, 0.5, 0, 0, 0)
    print(f'\n=== T_weita: ===\n{T_weita.get_T()}')

    print(f'\n=== T_weita.inv: ===\n{np.linalg.inv(T_weita.get_T())}')


    T_bodong = GazeboPose(-0.5, 0.5, -0.5, 0.5, 0, 0, 0)
    print(f'\n=== T_bodong: ===\n{T_bodong.get_T()}')

    print(f'\n=== T_bodong.inv: ===\n{np.linalg.inv(T_bodong.get_T())}')


    T_05 = np.array([
            [ 0.0,  0.0, -1.0, 0.0],
            [ 1.0,  0.0,  0.0, 0.0],
            [ 0.0, -1.0,  0.0, 0.0],
            [ 0.0,  0.0,  0.0, 1.0]], dtype=float)
    print(f'\n=== T_05: ===\n{T_05}')

    print(f'\n=== T_05.inv: ===\n{np.linalg.inv(T_05)}')


    T_mirror = np.array([
        [-1.0,  0.0,  0.0,  0.0],
        [ 0.0, -1.0,  0.0,  0.0],
        [ 0.0,  0.0,  1.0,  0.0],
        [ 0.0,  0.0,  0.0,  1.0]], dtype=float)
    T_new = np.matmul(T_mirror, T_05)
    print(f'\n=== T_new: ===\n{T_new}')

    print(f'\n=== T_new.inv: ===\n{np.linalg.inv(T_new)}')

def get_r_mat(qx, qy, qz, qw):
    return  GazeboPose(qx, qy, qz, qw, 0., 0., 0.).get_T()

def get_r_mat_ex(q):
    return  GazeboPose(q[0], q[1], q[2], q[3], 0., 0., 0.).get_T()

def get_t_mat(px, py, pz):
    mat_t = np.array([
        [ 1.,  0.,  0.,  px],
        [ 0.,  1.,  0.,  py],
        [ 0.,  0.,  1.,  pz],
        [ 0.,  0.,  0.,  1.]], dtype=np.float)
    return mat_t

def get_t_mat_ex(p):
    return get_t_mat(p[0], p[1], p[2])

def check_matrix_v2():
    T_05 = np.array([
        [ 0.,  0., -1., 0.],
        [ 1.,  0.,  0., 0.],
        [ 0., -1.,  0., 0.],
        [ 0.,  0.,  0., 1.]], dtype=np.float)

    T_new = np.array([
        [ 0.,  0.,  1., 0.],
        [-1.,  0.,  0., 0.],
        [ 0., -1.,  0., 0.],
        [ 0.,  0.,  0., 1.]], dtype=np.float)

    T_mirror = np.array([
        [-1.0,  0.0,  0.0,  0.0],
        [ 0.0, -1.0,  0.0,  0.0],
        [ 0.0,  0.0,  1.0,  0.0],
        [ 0.0,  0.0,  0.0,  1.0]], dtype=float)
    T_new = np.matmul(T_mirror, T_05)

    # pz --> 0.0

    px = 1.
    py = 100.
    pz = 10000.
    qx = 0.5
    qy = 0.3
    qz = -0.9
    qw = -0.7

    T_t = np.array([
        [ 1.,  0.,  0.,  px],
        [ 0.,  1.,  0.,  py],
        [ 0.,  0.,  1.,  pz],
        [ 0.,  0.,  0.,  1.]], dtype=np.float)

    # T_r = GazeboPose(qx, qy, qz, qw, 0., 0., 0.).get_T()
    T_r = get_r_mat(qx, qy, qz, qw)

    inv_T_t = np.linalg.inv(T_t)

    # ======================================================================== #
    # T_bodong = GazeboPose(-0.5, 0.5, -0.5, 0.5, 0, 0, 0).get_T()
    T_bodong = get_r_mat(-0.5, 0.5, -0.5, 0.5)
    kneron_print(f'\n=== T_bodong: ===\n{T_bodong}')

    kneron_print(f'\n=== T_05: ===\n{T_05}')
    kneron_print(f'\n=== T_mirror: ===\n{T_mirror}')
    kneron_print(f'\n=== T_new: ===\n{T_new}')

    kneron_print(f'\n=== T_r: ===\n{T_r}')
    kneron_print(f'\n=== T_t: ===\n{T_t}')
    kneron_print(f'\n=== inv_T_t: ===\n{inv_T_t}')

    kneron_print(f'\n\n\n')
    T_rt = T_t @ T_r
    kneron_print(f'\n=== T_rt: ===\n{T_rt}')
    T_rt_neg = np.linalg.inv(T_t) @ T_r
    kneron_print(f'\n=== T_rt_neg: ===\n{T_rt_neg}')

    # old op
    # T_old_op = T_05 @ T_rt_neg
    # T_new_op = T_new @ T_rt
    T_old_op = T_rt_neg @ T_05
    T_new_op = T_rt @ T_new
    kneron_print(f'\n=== T_old_op: ===\n{T_old_op}')
    kneron_print(f'\n=== T_new_op: ===\n{T_new_op}')

    # todo: set frame prev and frame curr
    # T_r_prev = GazeboPose(0.1, 0.2, 0.3, 0.4, 0., 0., 0.).get_T()
    # T_r_curr = GazeboPose(0.3, 0.7, 0.5, 0.2, 0., 0., 0.).get_T()
    q_i = [0.1, 0.2, 0.3, 0.4]
    q_j = [0.3, 0.7, 0.5, 0.2]

    T_r_i = get_r_mat_ex(q_i)
    T_r_j = get_r_mat_ex(q_j)

    p_i = [0.8, 2.3, 0.0]
    p_j = [6.3, 1.1, 0.0]

    T_t_i = get_t_mat_ex(p_i)
    T_t_j = get_t_mat_ex(p_j)

    T_i_rt_no_neg = T_t_i @ T_r_i
    T_i_rt_neg = np.linalg.inv(T_t_i) @ T_r_i

    T_j_rt_no_neg = T_t_j @ T_r_j
    T_j_rt_neg = np.linalg.inv(T_t_j) @ T_r_j

    kneron_print(f'\n### T_i_rt_no_neg: ###\n{T_i_rt_no_neg}')
    kneron_print(f'\n### T_i_rt_neg: ###\n{T_i_rt_neg}')
    kneron_print(f'\n### T_j_rt_no_neg: ###\n{T_j_rt_no_neg}')
    kneron_print(f'\n### T_j_rt_neg: ###\n{T_j_rt_neg}')

    T_i_no_neg = T_t_i @ T_r_i @ T_new
    T_i_neg = np.linalg.inv(T_t_i) @ T_r_i @ T_05

    T_j_no_neg = T_t_j @ T_r_j @ T_new
    T_j_neg = np.linalg.inv(T_t_j) @ T_r_j @ T_05

    kneron_print(f'\n=== T_i_no_neg: ===\n{T_i_no_neg}')
    kneron_print(f'\n=== T_i_neg: ===\n{T_i_neg}')
    kneron_print(f'\n=== T_j_no_neg: ===\n{T_j_no_neg}')
    kneron_print(f'\n=== T_j_neg: ===\n{T_j_neg}')

    # rel_pose = np.matmul(np.linalg.inv(pose_init), data_pose)
    rel_pose_no_neg = np.linalg.inv(T_i_no_neg) @ T_j_no_neg
    rel_pose_neg = np.linalg.inv(T_i_neg) @ T_j_neg

    kneron_print(f'\n=== rel_pose_no_neg: ===\n{rel_pose_no_neg}')
    kneron_print(f'\n=== rel_pose_neg: ===\n{rel_pose_neg}')

    xyz_rel_pose_no_neg = matrix_to_euler_angles(torch.from_numpy(rel_pose_no_neg[:3, :3]), 'XYZ').detach() * 180.0 / np.math.pi
    xyz_rel_pose_neg = matrix_to_euler_angles(torch.from_numpy(rel_pose_neg[:3, :3]), 'XYZ').detach() * 180.0 / np.math.pi
    kneron_print(f'\n=== xyz_rel_pose_no_neg: ===\n{xyz_rel_pose_no_neg}')
    kneron_print(f'\n=== xyz_rel_pose_neg: ===\n{xyz_rel_pose_neg}')

    inv_rel_pose_no_neg = np.linalg.inv(rel_pose_no_neg)
    inv_rel_pose_neg = np.linalg.inv(rel_pose_neg)

    kneron_print(f'\n=== inv_rel_pose_no_neg: ===\n{inv_rel_pose_no_neg}')
    kneron_print(f'\n=== inv_rel_pose_neg: ===\n{inv_rel_pose_neg}')

    kneron_print(f'\nsys.platform: {sys.platform}')


if __name__ == '__main__':
    setup_log('kneron_gazebo_config.log')
    time_beg_gazebo_config = time.time()
    np.set_printoptions(precision=6, suppress=True)

    # check_matrix_v1()
    check_matrix_v2()

    time_end_gazebo_config = time.time()
    logging.warning(f'gazebo_config.py elapsed {time_end_gazebo_config - time_beg_gazebo_config:.6f} seconds.')
    print0(pcolor(f'gazebo_config.py elapsed {time_end_gazebo_config - time_beg_gazebo_config:.6f} seconds.', 'yellow'))
