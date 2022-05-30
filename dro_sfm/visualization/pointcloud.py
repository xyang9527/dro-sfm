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
# from dro_sfm.utils.depth import load_depth
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
'''


class GazeboPose:
    def __init__(self, qx, qy, qz, qw, px, py, pz):
        r, i, j, k = qx, qy, qz, qw
        two_s = 2.0 / np.dot(np.array([r, i, j, k]), np.array([r, i, j, k]).transpose())
        logging.warning(f'  two_s: {two_s:.6f}')
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
        self.fx = 530.4669406576809
        self.fy = 530.4669406576809
        self.cx = 320.5
        self.cy = 240.5

        self.cam2imu = GazeboPose(0, 0, 0, 1, -0.076, -0.0, -0.025).get_T()
        self.imu2gt = GazeboPose(0, 0, 0, 1, 0, 0, -0.068).get_T()

        self.cam2gt = np.matmul(self.imu2gt, self.cam2imu)
        self.gt2cam = np.linalg.inv(self.cam2gt)

        logging.info(f'\n========== GazeboParam ==========')
        logging.info(f'  cam2imu:\n{self.cam2imu}\n')
        logging.info(f'  imu2gt: \n{self.imu2gt}\n')
        logging.info(f'  cam2gt: \n{self.cam2gt}\n')
        logging.info(f'  gt2cam: \n{self.gt2cam}\n')

    @property
    def get_fx(self):
        return self.fx

    @property
    def get_fy(self):
        return self.fy

    @property
    def get_cx(self):
        return self.cx

    @property
    def get_cy(self):
        return self.cy

    @property
    def get_cam2gt(self):
        return self.cam2gt

    @property
    def get_gt2cam(self):
        return self.gt2cam


def load_depth(d_file):
    # ref: dro_sfm/utils/depth.py  def load_depth(file)
        depth_png = np.array(load_image(d_file), dtype=int)
        assert (np.max(depth_png) > 255), 'Wrong .png depth file'
        return depth_png.astype(np.float) / 1000.0


def get_data(namelist, gazebo_param):
    dir_root = '/home/sigma/slam/matterport'
    data_col = []

    aligned_save_dir = osp.join(dir_root, 'test/matterport014_000/demo/aligned')
    if not osp.exists(aligned_save_dir):
        os.makedirs(aligned_save_dir)

    for name in namelist:
        dir_case = osp.join(dir_root, 'test/matterport014_000')

        file_color = osp.join(dir_case, f'cam_left/{name}.jpg')
        file_depth = osp.join(dir_case, f'depth/{name}.png')
        file_pose = osp.join(dir_case, f'pose/{name}.txt')

        file_cloud_obj = osp.join(dir_case, f'demo/cloud_obj/{name}.obj')
        file_cloud_ply = osp.join(dir_case, f'demo/cloud_ply/{name}.ply')

        files_read = [file_color, file_depth, file_pose]
        for item in files_read:
            if not osp.exists(item):
                logging.critical(f'file not exist {item}')
                raise ValueError(f'file not exist {item}')

        subprocess.call(['cp', file_color, file_color.replace('cam_left', 'demo')])
        subprocess.call(['cp', file_depth, file_depth.replace('depth', 'demo')])
        subprocess.call(['cp', file_pose, file_pose.replace('pose', 'demo')])

        files_write = [file_cloud_obj, file_cloud_ply]
        for item in files_write:
            file_dir = osp.dirname(item)
            if not osp.exists(file_dir):
                os.makedirs(file_dir)

        data_depth = load_depth(file_depth)
        data_image = np.array(load_image(file_color), dtype=int)

        fx, fy, cx, cy = gazebo_param.fx, gazebo_param.fy, gazebo_param.cx, gazebo_param.cy
        cloud = generate_pointcloud(data_image, data_depth, fx, fy, cx, cy, file_cloud_ply, 1.0)

        data_pose = np.genfromtxt(file_pose)
        data_col.append({'cloud': cloud, 'pose': data_pose, 'name': name})

    return data_col, aligned_save_dir


def load_path(namelist):
    gazebo_param = GazeboParam()
    data_col, data_save_dir = get_data(namelist, gazebo_param)

    pose_init = data_col[0]['pose']

    n_case = len(data_col)
    for idx_c in range(n_case):
        pose_curr = data_col[idx_c]['pose']
        name_curr = data_col[idx_c]['name']

        cloud = data_col[idx_c]['cloud']
        cloud_xyz = cloud[:, :, :3]
        cloud_rgb = cloud[:, :, 3:]
        cloud_xyz = cloud_xyz.reshape((-1, 3))
        cloud_rgb = cloud_rgb.reshape((-1, 3))

        logging.info(f'{name_curr} curr_pose:\n{pose_curr}\n')

        # rel_poses = [np.matmul(np.linalg.inv(x), pose).astype(np.float32) for x in context_poses]
        pose_curr_to_init_gt = np.matmul(np.linalg.inv(pose_init), pose_curr).astype(np.float32)
        logging.info(f'{name_curr} pose_curr_to_init_gt:\n{pose_curr_to_init_gt}\n')

        pose_curr_to_init_cam = np.matmul(gazebo_param.get_gt2cam, pose_curr_to_init_gt)
        logging.info(f'{name_curr} pose_curr_to_init_cam:\n{pose_curr_to_init_cam}\n')

        coord_swap = np.array([
            0.0, 0.0, -1.0, 0.0,
            0.0, -1.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0]).reshape((4, 4))

        pose_curr_to_init_cam_swap = np.matmul(pose_curr_to_init_cam, coord_swap)
        logging.info(f'{name_curr} pose_curr_to_init_cam_swap:\n{pose_curr_to_init_cam_swap}')

        n = cloud_xyz.shape[0]
        cloud_xyz_hom = np.transpose(np.hstack((cloud_xyz, np.ones((n, 1)))))

        cloud_world = np.transpose(np.dot(pose_curr_to_init_cam, cloud_xyz_hom))
        cloud_cam = np.transpose(np.dot(gazebo_param.get_gt2cam, cloud_xyz_hom))

        with open(osp.join(data_save_dir, f'aligned_rgb_{name_curr}_{idx_c:04d}.obj'), 'wt') as f_ou_align_rgb, \
            open(osp.join(data_save_dir, f'local_rgb_{name_curr}_{idx_c:04d}.obj'), 'wt') as f_ou_local_rgb, \
            open(osp.join(data_save_dir, f'aligned_gray_{name_curr}_{idx_c:04d}.obj'), 'wt') as f_ou_align_gray, \
            open(osp.join(data_save_dir, f'local_gray_{name_curr}_{idx_c:04d}.obj'), 'wt') as f_ou_local_gray:
            for i in range(n):
                x, y, z, w = cloud_world[i]
                r, g, b = cloud_rgb[i]
                f_ou_align_rgb.write(f'v {x} {y} {z} {r} {g} {b}\n')
                f_ou_align_gray.write(f'v {x} {y} {z}\n')
                x, y, z, w = cloud_cam[i]
                f_ou_local_rgb.write(f'v {x} {y} {z} {r} {g} {b}\n')
                f_ou_local_gray.write(f'v {x} {y} {z}\n')



def create_obj_cloud():
    names = ['000542628000000', '000543008000000', '000543496000000', '000543976000000']
    load_path(names)


if __name__ == '__main__':
    setup_log('kneron_pointcloud.log')
    time_beg_pointcloud = time.time()

    np.set_printoptions(precision=3, suppress=True)
    create_obj_cloud()

    time_end_pointcloud = time.time()
    print(f'pointcloud.py elapsed {time_end_pointcloud - time_beg_pointcloud:.6f} seconds.')
