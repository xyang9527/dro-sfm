
# ref: scannet_test_dataset.py

import re
from collections import defaultdict
import os
import logging

from torch.utils.data import Dataset
import numpy as np
from dro_sfm.utils.image import load_image
import IPython, cv2
########################################################################################################################
#### FUNCTIONS
########################################################################################################################

def dummy_calibration(image):
    w, h = [float(d) for d in image.size]
    return np.array([[1000. , 0.    , w / 2. - 0.5],
                     [0.    , 1000. , h / 2. - 0.5],
                     [0.    , 0.    , 1.          ]])

def get_idx(filename):
    return int(re.search(r'\d+', filename).group())

def read_files(directory, ext=('.png', '.jpg', '.jpeg', '.ppm'), skip_empty=True):
    files = defaultdict(list)
    for entry in os.scandir(directory):
        relpath = os.path.relpath(entry.path, directory)
        if entry.is_dir():
            color_path = os.path.join(entry.path, 'cam_left')
            d_files = read_files(color_path, ext=ext, skip_empty=skip_empty)
            if skip_empty and not len(d_files):
                continue
            files[relpath + '/cam_left'] = d_files[color_path]
        elif entry.is_file():
            if ext is None or entry.path.lower().endswith(tuple(ext)):
                pose_path = entry.path.replace('cam_left', 'pose').replace('.jpg', '.txt')
                pose = np.genfromtxt(pose_path)
                if not np.isinf(pose).any():
                    files[directory].append(relpath)
    return files

def read_npz_depth(file, depth_type):
    """Reads a .npz depth map given a certain depth_type."""
    depth = np.load(file)[depth_type + '_depth'].astype(np.float32)
    return np.expand_dims(depth, axis=2)

def read_png_depth(file):
    """Reads a .png depth map."""
    depth_png = np.array(load_image(file), dtype=int)

    # assert (np.max(depth_png) > 255), 'Wrong .png depth file'
    if (np.max(depth_png) > 255):
        depth = depth_png.astype(np.float) / 1000.
    else:
        depth = depth_png.astype(np.float)
    depth[depth_png == 0] = -1.
    return np.expand_dims(depth, axis=2)

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

########################################################################################################################
#### DATASET
########################################################################################################################

class MatterportTestDataset(Dataset):
    def __init__(self, root_dir, split, data_transform=None,
                 forward_context=0, back_context=0, strides=(5,),
                 depth_type=None, **kwargs):
        logging.warning(f'MatterportTestDataset::__init__('
                        f'\n  root_dir={root_dir},'
                        f'\n  split={split},'
                        f'\n  data_transform={data_transform},'
                        f'\n  forward_context={forward_context},'
                        f'\n  back_context={back_context},'
                        f'\n  strides={strides},'
                        f'\n  depth_type={depth_type}, ..)')
        super().__init__()

        self.depth_type = depth_type
        self.with_depth = depth_type is not '' and depth_type is not None
        self.root_dir = root_dir
        self.split = split

        self.backward_context = back_context
        self.forward_context = forward_context
        self.has_context = self.backward_context + self.forward_context > 0
        self.strides = strides[0]

        self.files = []

        source = ''
        if source == 'Folder':
            # ================= load from folder ====================
            self.file_tree = read_files(root_dir)
            # sort
            for k in self.file_tree:
                self.file_tree[k].sort(key=lambda x: int(x.split('.jpg')[0]))
            # save train list
            fo = open(os.path.join(os.path.dirname(self.root_dir), "splits/test_all_list.txt"), "w")
            for k, v in self.file_tree.items():
                for data in v:
                    fo.write(k + ' ' + data + '\n')
            fo.close()
        else:
            # =================== load from txt ====================
            self.file_tree = defaultdict(list)
            with open(os.path.join(self.root_dir, "splits/test_all_list.txt"), "r") as f:
                split_data = f.readlines()
            for data in split_data:
                scene, filename = data.split()
                self.file_tree[scene].append(filename)

        # =================== load from txt ====================
        with open(os.path.join(self.root_dir, self.split), "r") as f:
            split_data = f.readlines()

        split_data1 = split_data[::2]
        split_data2 = split_data[1:][::2]

        files = [x.replace(' ', '/').split()[0] for x in split_data1]
        context_files = [x.replace(' ', '/').split()[0] for x in split_data2]
        self.files = []
        self.context_files = []
        self.context2_files = []
        self.context3_files = []
        self.context4_files = []

        logging.warning(f'========== file_tree: ==========')
        for k, v in self.file_tree.items():
            logging.info(f'    {k}: {len(v)}')
        logging.info(f'  split_data:     {len(split_data)}')
        logging.info(f'  split_data1:    {len(split_data1)}')
        logging.info(f'  split_data2:    {len(split_data2)}')
        logging.info(f'  files:          {len(files)}')
        logging.info(f'  context_files:  {len(context_files)}')

        is_continuous_id = True

        if is_continuous_id:
            for d0, d1 in zip(files, context_files):
                sce = d0.split('/cam_left/')[0] + '/cam_left'
                id0 = d0.split('/cam_left/')[1]
                id1 = d1.split('/cam_left/')[1]

                # logging.info(f'  sce: {sce},  id0: {id0}, id1: {id1}')

                if sce in self.file_tree.keys() and os.path.exists(os.path.join(self.root_dir, sce)):
                    if int(id1.split('.')[0]) > int(id0.split('.')[0]):
                        # id0    id1    id2    id3    id4
                        # id0    id1    id0-5  id0+5  id0-10
                        id2 = '{:06d}.jpg'.format(int(id0.split('.')[0]) - 5)
                        id3 = '{:06d}.jpg'.format(int(id0.split('.')[0]) + 5)
                        id4 = '{:06d}.jpg'.format(int(id0.split('.')[0]) - 10)
                    else:
                        # id1    id0    id2    id3    id4
                        # id1    id0    id0+5  id0-5  id0+10
                        id2 = '{:06d}.jpg'.format(int(id0.split('.')[0]) + 5)
                        id3 = '{:06d}.jpg'.format(int(id0.split('.')[0]) - 5)
                        id4 = '{:06d}.jpg'.format(int(id0.split('.')[0]) + 10)

                    if self.forward_context == 1 and self.backward_context == 0:
                        self.files.append(d0)
                        self.context_files.append(d1)

                    if self.forward_context == 1 and self.backward_context == 1:
                        if os.path.exists(os.path.join(self.root_dir, sce, id2)):
                            self.files.append(d0)
                            self.context_files.append(d1)
                            self.context2_files.append(os.path.join(sce, id2))

                    if self.forward_context == 2 and self.backward_context == 2:
                        if os.path.exists(os.path.join(self.root_dir, sce, id2)) and    \
                        os.path.exists(os.path.join(self.root_dir, sce, id3)) and    \
                        os.path.exists(os.path.join(self.root_dir, sce, id4)):
                            self.files.append(d0)
                            self.context_files.append(d1)
                            self.context2_files.append(os.path.join(sce, id2))
                            self.context3_files.append(os.path.join(sce, id3))
                            self.context4_files.append(os.path.join(sce, id4))

        else:
            # TODO: filename: discrete timestamp
            n_frames = min(len(files), len(context_files))
            frame_id_list = [i for i in range(n_frames)]
            for idx_frame, d0, d1 in zip(frame_id_list, files, context_files):
                sce = d0.split('/cam_left/')[0] + '/cam_left'
                id0 = d0.split('/cam_left/')[1]
                id1 = d1.split('/cam_left/')[1]

                if sce in self.file_tree.keys() and os.path.exists(os.path.join(self.root_dir, sce)):
                    if self.forward_context == 1 and self.backward_context == 0:
                        self.files.append(d0)
                        self.context_files.append(d1)

                    if self.forward_context == 1 and self.backward_context == 1:
                        raise NotImplementedError

                    if self.forward_context == 2 and self.backward_context == 2:
                        raise NotImplementedError
            pass

        assert len(self.files) == len(self.context_files), \
            'len(files) != len(context_files) !!!!!'

        self.data_transform = data_transform

    def __len__(self):
        return len(self.files)

    def _change_idx(self, idx, filename):
        _, ext = os.path.splitext(os.path.basename(filename))
        return str(idx) + ext

    def _has_context(self, session, filename, file_list):
        context_paths = self._get_context_file_paths(filename, file_list)
        return all([f in file_list for f in context_paths])

    def _get_context_file_paths(self, filename, filelist):
        # fidx = get_idx(filename)
        fidx = filelist.index(filename)
        idxs = list(np.arange(-self.backward_context * self.strides, 0, self.strides)) + \
               list(np.arange(0, self.forward_context * self.strides, self.strides) + self.strides)
        return [filelist[fidx+i] if 0 <= fidx+i < len(filelist) else 'none' for i in idxs]

    def _read_rgb_context_files(self, session, filename):
        context_paths = self._get_context_file_paths(filename, self.file_tree[session])
        
        return [load_image(os.path.join(self.root_dir, session, filename))
                for filename in context_paths]

    def _read_rgb_file(self, session, filename):
        return load_image(os.path.join(self.root_dir, session, filename))

########################################################################################################################
#### DEPTH
########################################################################################################################

    def _read_depth(self, depth_file):
        """Get the depth map from a file."""
        if self.depth_type in ['velodyne']:
            return read_npz_depth(depth_file, self.depth_type)
        elif self.depth_type in ['groundtruth']:
            return read_png_depth(depth_file)
        else:
            raise NotImplementedError(
                'Depth type {} not implemented'.format(self.depth_type))

    def _get_depth_file(self, image_file):
        """Get the corresponding depth file from an image file."""
        depth_file = image_file.replace('cam_left', 'depth').replace('image', 'depth')
        depth_file = depth_file.replace('jpg', 'png')
        return depth_file

    def __getitem__(self, idx):
        color_path = self.files[idx]
        if self.forward_context == 1 and self.backward_context == 0:
            context_paths = [self.context_files[idx]]
        if self.forward_context == 1 and self.backward_context == 1:
            context_paths = [self.context_files[idx], self.context2_files[idx]]
        if self.forward_context == 2 and self.backward_context == 2:
            context_paths = [self.context_files[idx], self.context2_files[idx],
                             self.context3_files[idx], self.context4_files[idx],]

        image = load_image(os.path.join(self.root_dir, color_path))

        depth = self._read_depth(self._get_depth_file(os.path.join(self.root_dir, color_path)))
        resized_depth = cv2.resize(depth, image.size, interpolation = cv2.INTER_NEAREST)

        intr = np.array([[577.870605, 0.000000, 319.500000],
                    [0.000000, 577.870605, 239.500000],
                    [0.000000, 0.000000, 1.000000]])
        
        context_images = [load_image(os.path.join(self.root_dir, filename))
                                for filename in context_paths]
        pose_path = os.path.join(self.root_dir, color_path).replace('cam_left', 'pose').replace('.jpg', '.txt')
        pose = np.genfromtxt(pose_path)
        context_pose_paths = [os.path.join(self.root_dir, x).replace('cam_left', 'pose').
                                replace('.jpg', '.txt') for x in context_paths]
        context_poses = [np.genfromtxt(x) for x in context_pose_paths]

        # rel_poses = [np.matmul(x, np.linalg.inv(pose)).astype(np.float32) for x in context_poses]
        rel_poses = [np.matmul(np.linalg.inv(x), pose).astype(np.float32) for x in context_poses]

        sample = {
            'idx': idx,
            'filename': '%s_%s' % (color_path.split('/')[0], os.path.splitext(color_path.split('/')[-1])[0]),
            'rgb': image,
            'intrinsics': intr,
            'pose_context': rel_poses
        }

        # Add depth information if requested
        if self.with_depth:
            sample.update({
                'depth': depth,
            })

        if self.has_context:
            sample['rgb_context'] = context_images

        if self.data_transform:
            sample = self.data_transform(sample)

        return sample

########################################################################################################################