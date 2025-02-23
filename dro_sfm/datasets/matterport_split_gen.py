# -*- coding=utf-8 -*-

# generate split for matterport dataset

import logging
import numpy as np
import os
import os.path as osp
import sys
import time
from PIL import Image
import cv2
import copy

lib_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(lib_dir)

from dro_sfm.utils.setup_log import setup_log
from dro_sfm.utils.depth import viz_inv_depth
from dro_sfm.utils.image import write_image
from dro_sfm.utils.horovod import print0
from dro_sfm.utils.logging import pcolor
from dro_sfm.datasets.depth_filter import clip_depth


def generate_depth_vis(dir_root, subdirs):
    logging.warning(f'generate_depth_vis({dir_root}, {subdirs})')

    for item in subdirs:
        logging.info(f'  {item} ..')
        print0(pcolor(f'  generate_depth_vis( {item} ) ..', 'green'))

        case_dir = osp.join(dir_root, item)
        if not osp.exists(case_dir):
            logging.info(f'  skip {case_dir}')
            continue

        dir_depth_in = osp.join(case_dir, 'depth')
        if not osp.exists(dir_depth_in):
            logging.warning(f'  no depth in {case_dir}')
            continue

        dir_depth_vis = osp.join(case_dir, 'depth_vis')
        if not osp.exists(dir_depth_vis):
            os.mkdir(dir_depth_vis)
        else:
            logging.warning(f'  skip {dir_depth_vis}')
            continue

        dir_image_in = osp.join(case_dir, 'cam_left')
        dir_image_vis = osp.join(case_dir, 'cam_left_vis')
        if not osp.exists(dir_image_vis):
            os.mkdir(dir_image_vis)

        depth_files = sorted(os.listdir(dir_depth_in))
        for item_file in depth_files:
            if not item_file.endswith('.png'):
                continue

            img = np.array(Image.open(osp.join(dir_depth_in, item_file)), dtype=int)
            img = clip_depth(img)

            img_mask = img <= 0

            img_float = img.astype(np.float) / 1000.0
            img_vis = viz_inv_depth(img_float) * 255

            img_vis[img_mask, :] = 0

            save_name = osp.join(dir_depth_vis, item_file).replace('.png', '.jpg')
            write_image(save_name, img_vis)

            # image with mask
            name_src_image = osp.join(dir_image_in, osp.basename(item_file).replace('.png', '.jpg'))
            name_dst_image = osp.join(dir_image_vis, osp.basename(item_file).replace('.png', '.jpg'))
            if osp.exists(name_src_image):
                src_image = np.array(Image.open(name_src_image))

                src_image_copy = copy.deepcopy(src_image)
                src_image_copy[img_mask, :] = 0

                tmp_src_image = cv2.bitwise_and(src_image, src_image_copy)
                dst_image = cv2.addWeighted(src_image, 0.25, tmp_src_image, 0.75, 0)

                write_image(name_dst_image, dst_image)
            else:
                print(f'  skip {name_src_image}')
                logging.warning(f'  skip {name_src_image}')


def generate_split():
    use_data_0516 = False
    gen_vis_data = False

    if use_data_0516:
        # matterport dataset 0516
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
            "test/matterport014_000",
            "test/matterport005_000_0610"
        ]

    else:
        # matterport dataset 0614
        dir_root = '/home/sigma/slam/matterport0614'

        if not osp.exists(dir_root):
            raise ValueError(f'path not exist: {dir_root}')

        dir_save = osp.join(dir_root, 'splits')
        if not osp.exists(dir_save):
            os.mkdir(dir_save)

        subdirs_train_val_test = [
            "train_val_test/matterport005_000_0516", # 4199
            "train_val_test/matterport005_001_0516", # 5500
            "train_val_test/matterport010_000_0516", # 4186
            "train_val_test/matterport010_001_0516", # 3452
        ]
        subdirs_test = [
            "test/matterport014_000_0516", # 3624
            "test/matterport014_001_0516", # 7787
        ]

    T05 = np.array([[ 0.,  0., -1.,  0.],
                    [ 1.,  0.,  0.,  0.],
                    [ 0., -1.,  0.,  0.],
                    [ 0.,  0.,  0.,  1.]], dtype=np.float)
    T_mirror = np.array([
        [-1.0,  0.0,  0.0,  0.0],
        [ 0.0, -1.0,  0.0,  0.0],
        [ 0.0,  0.0,  1.0,  0.0],
        [ 0.0,  0.0,  0.0,  1.0]], dtype=float)
    T05 = np.matmul(T_mirror, T05)

    # create pose file
    subdirs_pose = []
    for item in subdirs_train_val_test:
        subdirs_pose.append(item)
    for item in subdirs_test:
        subdirs_pose.append(item)

    if gen_vis_data:
        # depth visualization
        generate_depth_vis(dir_root, subdirs_pose)

    for item in subdirs_pose:
        print0(pcolor(f'  pose: {item} ..', 'blue'))

        cam_pose_file = osp.join(dir_root, item, 'cam_pose.txt')
        cam_traj_file_world_coord = osp.join(dir_root, item, 'camera_trajectory_w.obj')
        cam_traj_file_camera_coord = osp.join(dir_root, item, 'camera_trajectory_c.obj')

        if not osp.exists(cam_pose_file):
            logging.warning(f'file not exist: {cam_pose_file}')
            print(f'file not exist: {cam_pose_file}')
            continue

        pose_dir = osp.join(dir_root, item, 'pose')
        pose_dir_world_coord = osp.join(dir_root, item, 'pose_w')

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
        gt_traj_file_world_coord = osp.join(dir_root, item, 'gt_trajectory_w.obj')
        gt_traj_file_camera_coord = osp.join(dir_root, item, 'gt_trajectory_c.obj')
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
            print0(pcolor(f'  split dateset ..', 'blue'))

            # test part
            n_case = len(subdirs_test)
            for idx_case in range(n_case):
                case_dir = osp.join(dir_root, subdirs_test[idx_case])
                if osp.exists(case_dir):
                    for item in sorted(os.listdir(osp.join(case_dir, image_dir))):
                        if item.endswith('.jpg'):
                            path_jpg = osp.join(dir_root, subdirs_test[idx_case], image_dir, item)
                            path_txt = path_jpg.replace('cam_left', 'pose').replace('.jpg', '.txt')
                            if not osp.exists(path_txt):
                                # print(f'skip {item} as missing {path_txt}')
                                logging.debug(f'skip {item} as missing {path_txt}')
                                n_frame_missing_pose_info += 1
                                continue
                            f_test.write(f'{subdirs_test[idx_case]}/{image_dir} {item}\n')
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
                                # print(f'skip {item} as missing {path_txt}')
                                logging.debug(f'skip {item} as missing {path_txt}')
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
    print0(pcolor(f'\nn_frame_missing_pose_info: {n_frame_missing_pose_info}', 'yellow'))


if __name__ == '__main__':
    setup_log('kneron_matterport_split_gen.log')
    time_beg_matterport_split_gen = time.time()

    generate_split()

    time_end_matterport_split_gen = time.time()
    logging.warning(f'matterport_split_gen.py elapsed {time_end_matterport_split_gen - time_beg_matterport_split_gen:.6f} seconds.')
    print0(pcolor(f'matterport_split_gen.py elapsed {time_end_matterport_split_gen - time_beg_matterport_split_gen:.6f} seconds.', 'yellow'))
