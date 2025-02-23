# generate split for matterport dataset

"""
    ├── ./matterport005_000
    │   ├── ./matterport005_000/cam_left       4199
    │   ├── ./matterport005_000/cam_right
    │   └── ./matterport005_000/depth
    ├── ./matterport005_001
    │   ├── ./matterport005_001/cam_left       5500
    │   ├── ./matterport005_001/cam_right
    │   └── ./matterport005_001/depth
    ├── ./matterport010_000
    │   ├── ./matterport010_000/cam_left       4186
    │   ├── ./matterport010_000/cam_right
    │   └── ./matterport010_000/depth
    ├── ./matterport010_001
    │   ├── ./matterport010_001/cam_left       3452
    │   ├── ./matterport010_001/cam_right
    │   └── ./matterport010_001/depth
    ├── ./matterport014_000
    │   ├── ./matterport014_000/cam_left       3624
    │   ├── ./matterport014_000/cam_right
    │   └── ./matterport014_000/depth

    train:
      matterport005_000[:-600]
      matterport005_001[:-600]
      matterport010_000[:-600]
      matterport010_001[:-600]
    val:
      matterport005_000[-600:-100]
      matterport005_001[-600:-100]
      matterport010_000[-600:-100]
      matterport010_001[-600:-100]
    test:
      matterport005_000[-100:]
      matterport005_001[-100:]
      matterport010_000[-100:]
      matterport010_001[-100:]
      matterport014_000

/opt/slam/matterport/split$ wc -l test.txt
4024 test.txt
/opt/slam/matterport/split$ wc -l val.txt
2000 val.txt
/opt/slam/matterport/split$ wc -l train.txt
15671 train.txt
"""

import logging
import numpy as np
import os
import os.path as osp
import sys
import time

lib_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(lib_dir)

from dro_sfm.utils.setup_log import setup_log
from dro_sfm.utils.horovod import print0
from dro_sfm.utils.logging import pcolor

def generate_split():
    use_data_0516 = False
    if use_data_0516:
        dir_root = '/home/sigma/slam/matterport0516'

        if not osp.exists(dir_root):
            raise ValueError(f'path not exist: {dir_root}')

        dir_save = osp.join(dir_root, 'splits')
        if not osp.exists(dir_save):
            os.mkdir(dir_save)

        subdirs_train_val_test = [
            "train_val_test/matterport005_000",
            "train_val_test/matterport005_001",
            "train_val_test/matterport010_000",
            "train_val_test/matterport010_001"
        ]
        subdirs_test = [
            "test/matterport014_000"
        ]

    else:
        # matterport0516_ex
        dir_root = '/home/sigma/slam/matterport0516_ex'

        if not osp.exists(dir_root):
            raise ValueError(f'path not exist: {dir_root}')

        dir_save = osp.join(dir_root, 'splits')
        if not osp.exists(dir_save):
            os.mkdir(dir_save)

        subdirs_train_val_test = [
            "train_val_test/matterport005_000_0516",
            "train_val_test/matterport005_001_0516",
            "train_val_test/matterport010_000_0516",
            "train_val_test/matterport010_000_0516"
        ]
        subdirs_test = [
            # "test/matterport014_000_0516",
            "test/matterport014_001_0516",
        ]

    T05 = np.array([[ 0.,  0., -1.,  0.],
                    [ 1.,  0.,  0.,  0.],
                    [ 0., -1.,  0.,  0.],
                    [ 0.,  0.,  0.,  1.]], dtype=np.float)

    enable_neg_xyz = True

    # create pose file
    subdirs_pose = []
    for item in subdirs_train_val_test:
        subdirs_pose.append(item)
    for item in subdirs_test:
        subdirs_pose.append(item)

    for item in subdirs_pose:
        cam_pose_file = osp.join(dir_root, item, 'cam_pose.txt')
        cam_traj_file_world_coord = osp.join(dir_root, item, 'traj_cam_pose_world_coord.obj')
        cam_traj_file_camera_coord = osp.join(dir_root, item, 'traj_cam_pose_camera_coord.obj')

        if not osp.exists(cam_pose_file):
            logging.warning(f'file not exist: {cam_pose_file}')
            continue

        pose_dir = osp.join(dir_root, item, 'pose')
        pose_dir_world_coord = osp.join(dir_root, item, 'pose_world_coord')

        folders_need = [pose_dir, pose_dir_world_coord]
        for item_dir in folders_need:
            if not osp.exists(item_dir):
                os.makedirs(item_dir)

        has_pose_init = False
        pose_init = None

        with open(cam_pose_file, 'r') as f_in, \
            open(cam_traj_file_world_coord, 'w') as f_ou_traj_world_coord, \
            open(cam_traj_file_camera_coord, 'w') as f_ou_traj_camera_coord:

            lines = f_in.readlines()
            n_valid = 0

            for idx_line, line in enumerate(lines):
                line = line.strip()
                if line.startswith('#'):
                    continue

                if 'nan' in line:
                    print(f'{line} @ line: {idx_line} @ {cam_pose_file}')
                    continue

                words = line.split()
                if len(words) != 8:
                    print(f'unexpected format: {words}')
                params = [float(v) for v in words[1:]]

                # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
                #     q = [q0 q1 q2 q3]^T = [qw qx qy qz]^T
                #     |q|^2 = q0^2 + q1^2 + q2^2 + q3^2 = qw^2 + qx^2 + qy^2 + qz^2 = 1
                x, y, z, i, j, k, r = params
                if enable_neg_xyz:
                    x, y, z = -x, -y, -z

                with open(osp.join(pose_dir, words[0].zfill(15) + '.txt'), 'w') as f_ou, \
                    open(osp.join(pose_dir_world_coord, words[0].zfill(15) + '.txt'), 'w') as f_ou_world_coord:
                    # References:
                    # dro_sfm/geometry/pose_trans.py   def quaternion_to_matrix(quaternions)
                    # https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
                    #     Maths - Conversion Quaternion to Matrix
                    two_s = 2.0 / np.dot(np.array([r, i, j, k]), np.array([r, i, j, k]).transpose())
                    if np.fabs(two_s - 2.0) > 1e-5:
                        logging.warning(f'  two_s: {two_s:.6f}')
                    part_R = np.array([
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

                    part_t = np.array([x, y, z]).reshape((3, 1))
                    T = np.hstack((part_R, part_t))
                    T_hom = np.vstack((T, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float).reshape((1, 4))))

                    # https://www.delftstack.com/howto/numpy/numpy-dot-vs-matmul/
                    #     NumPy dot vs matmul in Python
                    # mat = np.matmul(T_hom, T05)
                    mat = np.dot(T_hom, T05)

                    # debug / check
                    mat_matmul = np.matmul(T_hom, T05)
                    mat_dot = np.dot(T_hom, T05)
                    all_close = np.allclose(mat_matmul, mat_dot)
                    if not all_close:
                        print(f'      all_close: {all_close}')
                        logging.info(f'      all_close: {all_close}')

                    if not has_pose_init and pose_init is None:
                        pose_init = mat
                        has_pose_init = True

                    # camera coordinate
                    f_ou.write(f'{mat[0][0]} {mat[0][1]} {mat[0][2]} {mat[0][3]}\n'
                               f'{mat[1][0]} {mat[1][1]} {mat[1][2]} {mat[1][3]}\n'
                               f'{mat[2][0]} {mat[2][1]} {mat[2][2]} {mat[2][3]}\n'
                               f'{mat[3][0]} {mat[3][1]} {mat[3][2]} {mat[3][3]}\n')

                    # world coordinate
                    f_ou_world_coord.write(f'{T_hom[0][0]} {T_hom[0][1]} {T_hom[0][2]} {T_hom[0][3]}\n'
                                           f'{T_hom[1][0]} {T_hom[1][1]} {T_hom[1][2]} {T_hom[1][3]}\n'
                                           f'{T_hom[2][0]} {T_hom[2][1]} {T_hom[2][2]} {T_hom[2][3]}\n'
                                           f'{T_hom[3][0]} {T_hom[3][1]} {T_hom[3][2]} {T_hom[3][3]}\n')

                n_valid += 1
                f_ou_traj_world_coord.write(f'v {x} {y} {z}\n')
                vec_pos = np.dot(np.linalg.inv(pose_init), np.array([x, y, z, 1.0]).T).T
                f_ou_traj_camera_coord.write(f'v {vec_pos[0]} {vec_pos[1]} {vec_pos[2]}\n')

            for idx_p in range(1, n_valid-1, 2):
                f_ou_traj_world_coord.write(f'f {idx_p} {idx_p+1} {idx_p+2}\n')
                f_ou_traj_camera_coord.write(f'f {idx_p} {idx_p+1} {idx_p+2}\n')

        # ==================================================================== #
        # groundtruth.txt
        gt_pose_file = osp.join(dir_root, item, 'groundtruth.txt')
        gt_traj_file_world_coord = osp.join(dir_root, item, 'traj_groundtruth_world_coord.obj')
        gt_traj_file_camera_coord = osp.join(dir_root, item, 'traj_groundtruth_camera_coord.obj')
        if not osp.exists(gt_pose_file):
            logging.warning(f'file not exist: {gt_pose_file}')
            continue

        with open(gt_pose_file, 'r') as f_in, \
            open(gt_traj_file_world_coord, 'w') as f_ou_traj_world_coord, \
            open(gt_traj_file_camera_coord, 'w') as f_ou_traj_camera_coord:

            lines = f_in.readlines()
            n_valid = 0
            for idx_line, line in enumerate(lines):
                line = line.strip()
                if line.startswith('#'):
                    continue

                if 'nan' in line:
                    print(f'{line} @ line: {idx_line} @ {gt_pose_file}')
                    continue

                words = line.split()
                if len(words) != 8:
                    print(f'unexpected format: {words}')
                params = [float(v) for v in words[1:]]
                x, y, z, r, i, j, k = params
                if enable_neg_xyz:
                    x, y, z = -x, -y, -z

                n_valid += 1
                f_ou_traj_world_coord.write(f'v {x} {y} {z}\n')
                vec_pos = np.dot(np.linalg.inv(pose_init), np.array([x, y, z, 1.0]).T).T
                f_ou_traj_camera_coord.write(f'v {vec_pos[0]} {vec_pos[1]} {vec_pos[2]}\n')

            for idx_p in range(1, n_valid-1, 2):
                f_ou_traj_world_coord.write(f'f {idx_p} {idx_p+1} {idx_p+2}\n')
                f_ou_traj_camera_coord.write(f'f {idx_p} {idx_p+1} {idx_p+2}\n')
        # ==================================================================== #
    # end of "for item in subdirs_pose:"

    image_dir = 'cam_left'
    n_frame_missing_pose_info = 0

    # split dataset
    with open(osp.join(dir_save, 'train_all_list.txt'), 'w') as f_train, \
        open(osp.join(dir_save, 'val_all_list.txt'), 'w') as f_val, \
        open(osp.join(dir_save, 'test_all_list.txt'), 'w') as f_test:
            # test part
            n_case = len(subdirs_test)
            for idx_case in range(n_case):
                case_dir = osp.join(dir_root, subdirs_test[idx_case])
                if osp.exists(case_dir):
                    for item in sorted(os.listdir(osp.join(case_dir, image_dir))):
                        if item.endswith('.jpg'):
                            path_jpg = osp.join(dir_root, subdirs_test[0], image_dir, item)
                            path_txt = path_jpg.replace('cam_left', 'pose').replace('.jpg', '.txt')
                            if not osp.exists(path_txt):
                                print(f'skip {item} as missing {path_txt}')
                                logging.info(f'skip {item} as missing {path_txt}')
                                n_frame_missing_pose_info += 1
                                continue
                            f_test.write(f'{subdirs_test[0]}/{image_dir} {item}\n')
                else:
                    logging.warning(f'path not exist: {case_dir}')

            # train_val_test part
            n_case = len(subdirs_train_val_test)
            for id_case in range(n_case):
                case_dir = osp.join(dir_root, subdirs_train_val_test[id_case])
                if osp.exists(case_dir):
                    image_names = []
                    for item in sorted(os.listdir(osp.join(case_dir, image_dir))):
                        if item.endswith('.jpg'):
                            path_jpg = osp.join(dir_root, subdirs_train_val_test[id_case], image_dir, item)
                            path_txt = path_jpg.replace('cam_left', 'pose').replace('.jpg', '.txt')
                            if not osp.exists(path_txt):
                                print(f'skip {item} as missing {path_txt}')
                                logging.info(f'skip {item} as missing {path_txt}')
                                n_frame_missing_pose_info += 1
                                continue
                            image_names.append(item)
                    # train
                    for item in image_names[:-600]:
                        f_train.write(f'{subdirs_train_val_test[id_case]}/{image_dir} {item}\n')
                    # val
                    for item in image_names[-600:-100]:
                        f_val.write(f'{subdirs_train_val_test[id_case]}/{image_dir} {item}\n')
                    # test
                    for item in image_names[-100:]:
                        f_test.write(f'{subdirs_train_val_test[id_case]}/{image_dir} {item}\n')
                else:
                    logging.warning(f'path not exist: {case_dir}')
    print(f'n_frame_missing_pose_info: {n_frame_missing_pose_info}')


if __name__ == '__main__':
    setup_log('kneron_matterport_split_gen_old.log')
    time_beg_matterport_split_gen_old = time.time()

    generate_split()

    time_end_matterport_split_gen_old = time.time()
    logging.warning(f'matterport_split_gen_old.py elapsed {time_end_matterport_split_gen_old - time_beg_matterport_split_gen_old:.6f} seconds.')
    print0(pcolor(f'matterport_split_gen_old.py elapsed {time_end_matterport_split_gen_old - time_beg_matterport_split_gen_old:.6f} seconds.', 'yellow'))
