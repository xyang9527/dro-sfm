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

import os
import os.path as osp
import sys
lib_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(lib_dir)
import numpy as np
import time
import logging
from dro_sfm.utils.setup_log import setup_log


def generate_split():
    dir_root = 'D:/slam/matterport'

    if not osp.exists(dir_root):
        raise ValueError(f'path not exist: {dir_root}')
        return

    dir_save = osp.join(dir_root, 'splits')
    if not osp.exists(dir_save):
        os.mkdir(dir_save)

    subdirs_train_val_test = [
        "train_val_test/matterport005_000",
        "train_val_test/matterport005_001",
        "train_val_test/matterport010_000",
        "train_val_test/matterport010_000"
    ]
    subdirs_test = [
        "test/matterport014_000"
    ]

    # create pose file
    subdirs_pose = []
    for item in subdirs_train_val_test:
        subdirs_pose.append(item)
    for item in subdirs_test:
        subdirs_pose.append(item)

    for item in subdirs_pose:
        cam_pose_file = osp.join(dir_root, item, 'cam_pose.txt')
        if not osp.exists(cam_pose_file):
            logging.warning(f'file not exist: {cam_pose_file}')
            continue

        pose_dir = osp.join(dir_root, item, 'pose')
        if not osp.exists(pose_dir):
            os.mkdir(pose_dir)

        with open(cam_pose_file, 'r') as f_in:
            lines = f_in.readlines()
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
                # x, y, z, qx, qy, qz, qw = params
                x, y, z, r, i, j, k = params
                with open(osp.join(pose_dir, words[0].zfill(15) + '.txt'), 'w') as f_ou:
                    # ref: dro_sfm/geometry/pose_trans.py   def quaternion_to_matrix(quaternions)
                    '''
                    r, i, j, k = torch.unbind(quaternions, -1)
                    two_s = 2.0 / (quaternions * quaternions).sum(-1)

                    o = torch.stack(
                        (
                            1 - two_s * (j * j + k * k),
                            two_s * (i * j - k * r),
                            two_s * (i * k + j * r),
                            two_s * (i * j + k * r),
                            1 - two_s * (i * i + k * k),
                            two_s * (j * k - i * r),
                            two_s * (i * k - j * r),
                            two_s * (j * k + i * r),
                            1 - two_s * (i * i + j * j),
                        ),
                        -1,
                    )
                    return o.reshape(quaternions.shape[:-1] + (3, 3))
                    '''
                    # https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
                    #     Maths - Conversion Quaternion to Matrix
                    two_s = 2.0 / np.dot(np.array([r, i, j, k]), np.array([r, i, j, k]).transpose())
                    mat = np.array([
                            1 - two_s * (j * j + k * k),
                            two_s * (i * j - k * r),
                            two_s * (i * k + j * r),
                            two_s * (i * j + k * r),
                            1 - two_s * (i * i + k * k),
                            two_s * (j * k - i * r),
                            two_s * (i * k - j * r),
                            two_s * (j * k + i * r),
                            1 - two_s * (i * i + j * j)
                            ])
                    # print(f'two_s: {two_s}')
                    f_ou.write(f'{mat[0]} {mat[1]} {mat[2]} {x}\n'
                                '{mat[3]} {mat[4]} {mat[5]} {y}\n'
                                '{mat[6]} {mat[7]} {mat[8]} {z}\n'
                                '0.000000 0.000000 0.000000 1.000000\n')

    image_dir = 'cam_left'
    n_frame_missing_pose_info = 0

    with open(osp.join(dir_save, 'train_all_list.txt'), 'w') as f_train, \
        open(osp.join(dir_save, 'val_all_list.txt'), 'w') as f_val, \
        open(osp.join(dir_save, 'test_all_list.txt'), 'w') as f_test:
            # test part
            case_dir = osp.join(dir_root, subdirs_test[0])
            if osp.exists(case_dir):
                for item in sorted(os.listdir(osp.join(case_dir, image_dir))):
                    if item.endswith('.jpg'):
                        path_jpg = osp.join(dir_root, subdirs_test[0], image_dir, item)
                        path_txt = path_jpg.replace('cam_left', 'pose').replace('.jpg', '.txt')
                        if not osp.exists(path_txt):
                            print(f'skip {item} {path_jpg} as missing {path_txt}')
                            logging.info(f'skip {item} {path_jpg} as missing {path_txt}')
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
                                logging.info(f'skip {item} {path_jpg} as missing {path_txt}')
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
    setup_log('kneron_train_split_gen.log')
    time_beg_matterport_split_gen = time.time()

    generate_split()

    time_end_matterport_split_gen = time.time()
    print(f'matterport_split_gen.py elapsed {time_end_matterport_split_gen - time_beg_matterport_split_gen:.6f} seconds.')
