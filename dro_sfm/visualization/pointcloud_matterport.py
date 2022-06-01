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


'''
  root@Notebook-PC:/work/gazebo_robomaker# ./roboMaker.sh --info
  RoboMaker Tools
          --install
          --clean
          ---info
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

        gazebo robot                             camera
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

cam_to_gazebo_robot

        0  0 -1  0
        1  0  0  0
        0 -1  0  0
        0  0  0  1


https://ux.stackexchange.com/questions/79561/why-are-x-y-and-z-axes-represented-by-red-green-and-blue
  /media/figs/matterport_world_coord.png
  /media/figs/matterport_robot_vs_world.png


           gazebo world                              gazebo robot
                                                         Z
               | Z                                       |
               |                                         |
               |                                         |
 X             |                                         |
 ______________|                                         |_____________ Y
               /                                        /
              /                                        /
             /                                        /
            /                                        /
           /                                        / X
           Y

           X --------------------------------------- -Y
           Y ---------------------------------------  X
           Z ---------------------------------------  Z

gazebo_robot_to_gazebo_world

        0 -1  0  0
        1  0  0  0
        0  0  1  0
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

        self.cam2gazeborobot = np.array([
            [ 0.0,  0.0, -1.0, 0.0],
            [ 1.0,  0.0,  0.0, 0.0],
            [ 0.0, -1.0,  0.0, 0.0],
            [ 0.0,  0.0,  0.0, 1.0]], dtype=np.float)

        self.gazeborobot2gazeboworld = np.array([
            [ 0.0, -1.0,  0.0, 0.0],
            [ 1.0,  0.0,  0.0, 0.0],
            [ 0.0,  0.0,  1.0, 0.0],
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
    def get_cam2gazeborobot(self):
        return self.cam2gazeborobot

    @property
    def get_gazeborobot2gazeboworld(self):
        return self.gazeborobot2gazeboworld


def load_matterport_depth(d_file):
    # ref: dro_sfm/utils/depth.py  def load_depth(file)
    depth_png = np.array(load_image(d_file), dtype=int)
    depth = depth_png.astype(np.float) / 1000.0
    depth[depth_png == 0] = -1.
    return depth


def load_data(names):
    logging.warning(f'load_data(..)')
    n = len(names)
    logging.info(f'  {n} files to process ..')

    intr_color = np.array([[530.4669406576809,   0.0,             320.5, 0.0],
                           [0.0,               530.4669406576809, 240.5, 0.0],
                           [0.0,                 0.0,               1.0, 0.0],
                           [0.0,                 0.0,               0.0, 1.0]])
    fx = intr_color[0][0]
    fy = intr_color[1][1]
    cx = intr_color[0][2]
    cy = intr_color[1][2]

    gazebo_param = GazeboParam()

    dir_root = '/home/sigma/slam/matterport/test/matterport014_000_0601'
    dir_cloud_ply = osp.join(dir_root, 'demo/ply')
    dir_cloud_obj = osp.join(dir_root, 'demo/obj')
    dir_cloud_jpg = osp.join(dir_root, 'demo/jpg')
    folders_need = [dir_cloud_ply, dir_cloud_obj, dir_cloud_jpg]
    for item_dir in folders_need:
        if not osp.exists(item_dir):
            os.makedirs(item_dir)

    pose_init = None
    for idx_f in range(n):
        name = names[idx_f]
        print(f'  process frame: {name} ..')
        data_color = load_image(osp.join(dir_root, f'cam_left/{name}.jpg'))
        data_depth = load_matterport_depth(osp.join(dir_root, f'depth/{name}.png'))
        data_pose = np.genfromtxt(osp.join(dir_root, f'pose/{name}.txt'))

        subprocess.call(['cp', osp.join(dir_root, f'cam_left/{name}.jpg'), osp.join(dir_cloud_jpg, f'{name}.jpg')])
        if idx_f == 0:
            pose_init = data_pose

        file_cloud_ply = osp.join(dir_cloud_ply, f'{name}.ply')
        data_depth_resized = cv2.resize(data_depth, data_color.size, interpolation = cv2.INTER_NEAREST)
        cloud = generate_pointcloud(np.array(data_color, dtype=int), data_depth_resized, fx, fy, cx, cy, file_cloud_ply, 1.0)

        # rel_pose = np.matmul(pose_init, np.linalg.inv(data_pose)) # v1
        rel_pose = np.matmul(np.linalg.inv(pose_init), data_pose) # v2

        cloud_xyz = cloud[:, :3]
        cloud_rgb = cloud[:, 3:]

        n = cloud_xyz.shape[0]
        cloud_xyz_hom = np.transpose(np.hstack((cloud_xyz, np.ones((n, 1)))))

        cloud_xyz_hom_robot = np.dot(gazebo_param.get_cam2gazeborobot, cloud_xyz_hom)
        cloud_xyz_hom_world = np.dot(gazebo_param.get_gazeborobot2gazeboworld, cloud_xyz_hom_robot)
        logging.info(f'    use cloud_xyz_hom_world')

        trans_in_all = rel_pose # v1                                                                          F
        # trans_in_all = np.linalg.inv(rel_pose) # v2                                                           F
        trans_in_all = np.matmul(rel_pose, gazebo_param.get_cam2gazeborobot) # v3                             F
        # trans_in_all = np.matmul(gazebo_param.get_cam2gazeborobot, rel_pose) # v4                             F
        # trans_in_all = np.matmul(np.linalg.inv(rel_pose), gazebo_param.get_cam2gazeborobot) # v5              F
        # trans_in_all = np.matmul(gazebo_param.get_cam2gazeborobot, np.linalg.inv(rel_pose)) # v6              F
        # trans_in_all = np.matmul(rel_pose, np.matmul(gazebo_param.get_gazeborobot2gazeboworld, gazebo_param.get_cam2gazeborobot)) # v7                                        F
        # trans_in_all = np.matmul(rel_pose, np.matmul(gazebo_param.get_cam2gt, np.matmul(gazebo_param.get_gazeborobot2gazeboworld, gazebo_param.get_cam2gazeborobot))) # v8    F
        # trans_in_all = np.matmul(rel_pose, np.matmul(gazebo_param.get_gt2cam, np.matmul(gazebo_param.get_gazeborobot2gazeboworld, gazebo_param.get_cam2gazeborobot))) # v9    F
        # trans_in_all = np.matmul(np.matmul(gazebo_param.get_gazeborobot2gazeboworld, gazebo_param.get_cam2gazeborobot), rel_pose) # v10                                       F
        # trans_in_all = np.matmul(np.matmul(gazebo_param.get_cam2gt, np.matmul(gazebo_param.get_gazeborobot2gazeboworld, gazebo_param.get_cam2gazeborobot)), rel_pose) # v11   F
        # trans_in_all = np.matmul(np.matmul(gazebo_param.get_gt2cam, np.matmul(gazebo_param.get_gazeborobot2gazeboworld, gazebo_param.get_cam2gazeborobot)), rel_pose) # v12   F

        cloud_xyz_align = np.dot(trans_in_all, cloud_xyz_hom)
        cloud_xyz_align_t = np.transpose(cloud_xyz_align)

        with open(osp.join(dir_cloud_obj, f'v3_{name}.obj'), 'w') as f_ou_align_rgb:
            for i in range(n):
                x, y, z, w = cloud_xyz_align_t[i]
                r, g, b = cloud_rgb[i]
                f_ou_align_rgb.write(f'v {x} {y} {z} {r} {g} {b}\n')


def create_obj_cloud():
    names = ['000542628000000', '000543360000000', '000544048000000', '000544712000000', '000545368000000',
             '000546008000000', '000546648000000', '000547312000000', '000547840000000', '000548340000000',
             '000548868000000', '000549400000000', '000549932000000', '000550472000000', '000550996000000',
             '000551504000000', '000552020000000', '000552520000000', '000553028000000', '000553528000000',
             '000554052000000', '000554604000000', '000555100000000', '000555592000000', '000556112000000',
             '000556612000000', '000557108000000', '000557592000000', '000558112000000', '000558772000000',
             '000559456000000', '000560148000000', '000560824000000', '000561488000000', '000562168000000',
             '000562976000000', '000563884000000', '000564664000000', '000565328000000', '000565932000000']
    data_dir = '/home/sigma/slam/matterport/test/matterport014_000_0601/pose'
    names = []
    for item in sorted(os.listdir(data_dir)):
        # print(item)
        names.append(osp.splitext(item)[0])
    load_data(names[:10])


if __name__ == '__main__':
    setup_log('kneron_pointcloud_matterport.log')
    time_beg_pointcloud = time.time()

    np.set_printoptions(precision=6, suppress=True)
    create_obj_cloud()

    time_end_pointcloud = time.time()
    print(f'pointcloud.py elapsed {time_end_pointcloud - time_beg_pointcloud:.6f} seconds.')
