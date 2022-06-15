# -*- coding=utf-8 -*-

import os.path as osp
import sys
lib_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(lib_dir)
import logging
import numpy as np


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
        /                                            |        \ Z
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
'''


class GazeboPose:
    def __init__(self, qx, qy, qz, qw, px, py, pz):
        r, i, j, k = qx, qy, qz, qw
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


if __name__ == '__main__':
    np.set_printoptions(precision=6, suppress=True)

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
