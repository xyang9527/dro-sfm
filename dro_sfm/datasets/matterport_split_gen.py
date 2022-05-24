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
import os
import os.path as osp
import datetime
import time

def generate_split():
    dir_root = '/opt/slam/matterport'

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
    
    image_dir = 'cam_left'
    
    with open(osp.join(dir_save, 'train_all_list.txt'), 'w') as f_train, \
        open(osp.join(dir_save, 'val_all_list.txt'), 'w') as f_val, \
        open(osp.join(dir_save, 'test_all_list.txt'), 'w') as f_test:
            # test part
            case_dir = osp.join(dir_root, subdirs_test[0])
            if osp.exists(case_dir):
                for item in sorted(os.listdir(osp.join(case_dir, image_dir))):
                    if item.endswith('.jpg'):
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
                    
                    pass
                else:
                    logging.warning(f'path not exist: {case_dir}')            
            pass
    pass

if __name__ == '__main__':
    time_beg_matterport_split_gen = time.time()

    generate_split()

    time_end_matterport_split_gen = time.time()
    print(f'matterport_split_gen.py elapsed {time_end_matterport_split_gen - time_beg_matterport_split_gen:.6f} seconds.')
