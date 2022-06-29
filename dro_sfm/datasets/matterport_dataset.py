
# ref: scannet_dataset.py

import re
from collections import defaultdict
import os
import os.path as osp
import sys
lib_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(lib_dir)
import logging

from torch.utils.data import Dataset
import numpy as np
from dro_sfm.utils.image import load_image
import IPython, cv2

from dro_sfm.datasets.depth_filter import matrix_to_6d_pose, pose_in_threshold_5, pose_in_threshold_1


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

    depth = depth_png.astype(np.float) / 1000.
    # assert (np.max(depth_png) > 1000.), 'Wrong .png depth file'
    # if (np.max(depth_png) > 1000.):
    #     depth = depth_png.astype(np.float) / 1000.
    # else:
    #     depth = depth_png.astype(np.float)
    depth[depth_png == 0] = -1.
    return np.expand_dims(depth, axis=2)

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def adaptive_downsample(root_dir, data_dir, image_names, step):
    logging.warning(f'adaptive_downsample({root_dir}, {data_dir}, {len(image_names)}, {step})')

    n_frames = len(image_names)
    if n_frames <= step:
        return image_names

    arr_pose = []
    for item in image_names:
        jpg_name = osp.join(root_dir, data_dir, item)
        txt_name = jpg_name.replace('cam_left', 'pose').replace('jpg', 'txt')
        if not osp.exists(jpg_name):
            print(f'path not exist: {jpg_name}')
            raise ValueError
        if not osp.exists(txt_name):
            print(f'path not exist: {txt_name}')
            raise ValueError
        arr_pose.append(np.genfromtxt(txt_name))

    if len(image_names) != len(arr_pose):
        raise ValueError

    selected_image_names = []
    selected_idx = []
    curr_idx = 0
    while curr_idx < n_frames - step:
        selected_image_names.append(image_names[curr_idx])
        selected_idx.append(curr_idx)

        pose_6d = matrix_to_6d_pose(arr_pose[curr_idx], arr_pose[curr_idx + 1])
        if not pose_in_threshold_1(pose_6d):
            raise ValueError

        next_idx = curr_idx + 1
        all_in_thr = True
        for offset in range(step):
            next_idx = curr_idx + 1 + offset
            pose_6d = matrix_to_6d_pose(arr_pose[curr_idx], arr_pose[next_idx])
            if not pose_in_threshold_5(pose_6d):
                # print(f'break at: {curr_idx} {next_idx}')
                curr_idx += offset
                all_in_thr = False
                break
        if all_in_thr:
            curr_idx += step

    # print(f'image_names:          {len(image_names)}')
    # print(f'selected_image_names: {len(selected_image_names)}')
    # print(f'selected_idx:         {len(selected_idx)}')
    # print(f'                      {selected_idx}')
    if len(selected_image_names) != len(selected_idx):
        raise ValueError
    n_selected = len(selected_image_names)
    for i in range(1, n_selected):
        if selected_idx[i] - selected_idx[i-1] > step:
            raise ValueError
    return selected_image_names

########################################################################################################################
#### DATASET
########################################################################################################################

class MatterportDataset(Dataset):
    def __init__(self, root_dir, split, data_transform=None,
                 forward_context=0, back_context=0, strides=(1,),
                 depth_type=None, **kwargs):
        logging.warning(f'MatterportDataset::__init__('
                        f'\n  root_dir={root_dir},'
                        f'\n  split={split},'
                        f'\n  data_transform={data_transform},'
                        f'\n  forward_context={forward_context},'
                        f'\n  back_context={back_context},'
                        f'\n  strides={strides},'
                        f'\n  depth_type={depth_type}, ..)')
        super().__init__()
        # Asserts
        # assert depth_type is None or depth_type == '', \
        #     'ImageDataset currently does not support depth types'
        assert len(strides) == 1 and strides[0] == 1, \
            'ImageDataset currently only supports stride of 1.'

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
            # test split
            with open(os.path.join(os.path.dirname(self.root_dir), "splits/test_split.txt"), "r") as f:
                test_data = f.readlines()
            test_scenes = [d.split('/')[0] for d in test_data]

            self.file_tree = read_files(root_dir)
            # remove test scenes
            for scene in test_scenes:
                key = scene + '/cam_left'
                if key in self.file_tree:
                    self.file_tree.pop(key, None)
                    print('remove test scene:', scene)
            # sort
            for k in self.file_tree:
                self.file_tree[k].sort(key=lambda x: int(x.split('.jpg')[0]))
            # save train list
            fo = open(os.path.join(os.path.dirname(self.root_dir), "splits/train_all_list.txt"), "w")
            for k, v in self.file_tree.items():
                for data in v:
                    fo.write(k + ' ' + data + '\n')
            fo.close()
        else:
            # =================== load from txt ====================
            self.file_tree = defaultdict(list)
            with open(os.path.join(self.root_dir, self.split), "r") as f:
                split_data = f.readlines()
            for data in split_data:
                scene, filename = data.split()
                self.file_tree[scene].append(filename)

        logging.warning(f'========== before downsample file_tree: ==========')
        total_frames = 0
        for k, v in self.file_tree.items():
            logging.info(f'    {k}: {len(v)}')
            total_frames += len(v)
        logging.info(f'    total frames:               {total_frames:6d}')
        logging.info(f'    len(self.files):            {len(self.files):6d}')
        logging.info(f'    len(self.file_tree.keys()): {len(self.file_tree.keys()):6d}')

        use_adaptive_downsample = True
        # downsample
        for k in self.file_tree:
            step = 5
            if use_adaptive_downsample:
                # cut sequences: percent of invalid depth / large camera movement between consecutive frames
                self.file_tree[k] = adaptive_downsample(self.root_dir, k, self.file_tree[k], step)
            else:
                self.file_tree[k] = self.file_tree[k][::step]

        for k, v in self.file_tree.items():
            file_list = v
            files = [fname for fname in file_list if self._has_context(k, fname, file_list)]
            self.files.extend([[k, fname] for fname in files])

        logging.warning(f'========== after downsample file_tree: ==========')
        total_frames = 0
        for k, v in self.file_tree.items():
            logging.info(f'    {k}: {len(v)}')
            total_frames += len(v)
        logging.info(f'    total frames:               {total_frames:6d}')
        logging.info(f'    len(self.files):            {len(self.files):6d}')
        logging.info(f'    len(self.file_tree.keys()): {len(self.file_tree.keys()):6d}')

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
            # logging.info(f'  depth_file: {depth_file}')
            return read_png_depth(depth_file)
        else:
            raise NotImplementedError(
                'Depth type {} not implemented'.format(self.depth_type))

    def _get_depth_file(self, image_file):
        """Get the corresponding depth file from an image file."""
        # logging.info(f'  image_file: {image_file}')
        # depth_file = image_file.replace('color', 'depth').replace('image', 'depth')
        depth_file = image_file.replace('cam_left', 'depth')
        depth_file = depth_file.replace('jpg', 'png')
        # logging.info(f'  image_file: {image_file}')
        # logging.info(f'  depth_file: {depth_file}')
        return depth_file

    def __getitem__(self, idx):
        session, filename = self.files[idx]
        image = self._read_rgb_file(session, filename)

        if self.with_depth:
            depth = self._read_depth(self._get_depth_file(os.path.join(self.root_dir, session, filename)))
            resized_depth = cv2.resize(depth, image.size, interpolation = cv2.INTER_NEAREST)

        # intr_path = os.path.join(self.root_dir, session, filename).split('color')[0] + 'intrinsic/intrinsic_color.txt'
        # intr = np.genfromtxt(intr_path)[:3, :3]

        #
        # scannet: 
        #     color: 000000.jpg   1296x968
        #     depth: 000000.png    640x480
        #
        #     pose:  000000.txt
        #
        #                0.344057 -0.251783 0.904561 3.861332
        #                -0.937243 -0.034044 0.347011 2.690903
        #                -0.056577 -0.967185 -0.247695 1.806742
        #                0.000000 0.000000 0.000000 1.000000
        #
        #
        #     intrinsic/intrinsic_color.txt
        #
        #                1170.187988 0.000000 647.750000 0.000000
        #                0.000000 1170.187988 483.750000 0.000000
        #                0.000000 0.000000 1.000000 0.000000
        #                0.000000 0.000000 0.000000 1.000000
        #
        #
        #     intrinsic/intrinsic_depth.txt
        #
        #                577.870605 0.000000 319.500000 0.000000
        #                0.000000 577.870605 239.500000 0.000000
        #                0.000000 0.000000 1.000000 0.000000
        #                0.000000 0.000000 0.000000 1.000000
        #
        #
        # matterport:
        #     cam_left/001512072000000.jpg      640x480
        #     depth/001512072000000.png         640x480

        intr = np.array([[577.870605, 0.000000, 319.500000],
                         [0.000000, 577.870605, 239.500000],
                         [0.000000, 0.000000, 1.000000]])

        context_paths = self._get_context_file_paths(filename, self.file_tree[session])
        context_images = [load_image(os.path.join(self.root_dir, session, filename))
                                for filename in context_paths]
        # pose_path = os.path.join(self.root_dir, session, filename).replace('color', 'pose').replace('.jpg', '.txt')
        pose_path = os.path.join(self.root_dir, session, filename).replace('cam_left', 'pose').replace('.jpg', '.txt')
        pose = np.genfromtxt(pose_path)
        # logging.info(f'  load pose: {pose_path}')
        context_pose_paths = [os.path.join(self.root_dir, session, x).replace('cam_left', 'pose').
                                replace('.jpg', '.txt') for x in context_paths]
        context_poses = [np.genfromtxt(x) for x in context_pose_paths]

        #rel_poses = [np.matmul(x, np.linalg.inv(pose)).astype(np.float32) for x in context_poses]
        # current_pose_to_context_pose
        #   x as world pose
        rel_poses = [np.matmul(np.linalg.inv(x), pose).astype(np.float32) for x in context_poses]

        sample = {
            'idx': idx,
            'filename': '%s_%s' % (session.split('/')[0], os.path.splitext(filename)[0]),
            'rgb': image,
            'intrinsics': intr,
            'pose_context': rel_poses
        }

        # print(filename, context_paths)

        # Add depth information if requested
        if self.with_depth:
            sample.update({
                'depth': resized_depth,
            })

        if self.has_context:
            sample['rgb_context'] = context_images

        if self.data_transform:
            sample = self.data_transform(sample)

        return sample

########################################################################################################################
