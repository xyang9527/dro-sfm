# -*- coding=utf-8 -*-

import sys
import os
import os.path as osp
lib_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(lib_dir)

import argparse
from ast import parse
import numpy as np
import torch
from glob import glob
import logging
import time
import datetime
from PIL import Image
import subprocess
import git
import copy
from collections import OrderedDict

from dro_sfm.models.model_wrapper import ModelWrapper
from dro_sfm.utils.horovod import hvd_disable
from dro_sfm.datasets.augmentations import resize_image, to_tensor
from dro_sfm.utils.image import load_image
from dro_sfm.utils.config import parse_test_file
from dro_sfm.utils.load import set_debug
from dro_sfm.utils.depth import write_depth, inv2depth, viz_inv_depth
from dro_sfm.utils.image import write_image
from scripts import vis
from multiprocessing import Queue
import cv2
from dro_sfm.utils.setup_log import setup_log, git_info
from dro_sfm.utils.horovod import print0
from dro_sfm.utils.logging import pcolor
from dro_sfm.visualization.viz_image_grid import VizImageGrid, Colors


def init_model(ckpt):
    logging.warning(f'init_model()')
    hvd_disable()
    config, state_dict = parse_test_file(ckpt)
    print0(pcolor(f'model: {ckpt}', 'red'))
    set_debug(True)
    image_shape = (240, 320)
    model_wrapper = ModelWrapper(config, load_datasets=False)
    model_wrapper.load_state_dict(state_dict)
    if torch.cuda.is_available():
        model_wrapper = model_wrapper.to('cuda')
    else:
        raise RuntimeError("cuda is not available")
    model_wrapper.eval()
    return model_wrapper, image_shape

def get_intrinsics(image_shape_raw, image_shape):
    intr = np.array([530.4669406576809,   0.0,             320.5,
                        0.0,               530.4669406576809, 240.5,
                        0.0,                 0.0,               1.0], dtype=np.float32).reshape(3, 3)
    orig_w, orig_h = image_shape_raw
    out_h, out_w = image_shape

    # Scale intrinsics
    intr[0] *= out_w / orig_w
    intr[1] *= out_h / orig_h

    return intr

def process_image(filename, image_shape_raw, image_shape):
    image = load_image(filename)
    intr = get_intrinsics(image_shape_raw, image_shape)
    image = resize_image(image, image_shape)
    image = to_tensor(image).unsqueeze(0)
    intr = torch.from_numpy(intr).unsqueeze(0)

    if torch.cuda.is_available():
        image = image.to('cuda')
        intr = intr.to('cuda')
    return image, intr

def compare_depth(path_1, path_2):
    for item in os.listdir(path_1):
        if osp.splitext(item)[1] == '.npy':
            file_1 = osp.join(path_1, item)
            file_2 = osp.join(path_2, item)
            if not osp.exists(file_2):
                print0(pcolor(f'missing {file_2}', 'magenta'))
                raise ValueError
                continue
            data_1 = np.load(file_1)
            data_2 = np.load(file_2)
            if data_1.dtype != data_2.dtype or data_1.shape != data_2.shape:
                raise ValueError
            if np.allclose(data_1, data_2, atol=1e-10):
                print0(pcolor(f'  allclose: {file_1} {file_2}', 'yellow'))
            else:
                print0(pcolor(f'  unmatched value: {file_1} {file_2}', 'red'))
                data_diff = data_1 - data_2
                for desc, data_t in zip(['data_diff', 'data_1', 'data_2'],
                                        [data_diff, data_1, data_2]):
                    print0(pcolor(f'  {desc:10s} amax: {np.amax(data_t)}, amin: {np.amin(data_t)}, mean: {np.mean(data_t)}', 'blue'))
                
                # raise ValueError
            

def infer_depth(ckpt, input_dir, output_dir, cloud_dir='', model_name=''):
    '''
    python scripts/infer_video.py \
        --checkpoint /home/sigma/slam/models/lynx@27/SupModelMF_DepthPoseNet_it12-h-out_epoch=116_matterport0516_ex-test_all_list-groundtruth-abs_rel_pp_gt=0.265.ckpt \
        --input /home/sigma/slam/gazebo0629/0701_2022_sim/cam_left \
        --output /home/sigma/slam/gazebo0629/0701_2022_sim/infer_video \
        --sample_rate 1 \
        --data_type matterport \
        --ply_mode \
        --max_frame 1000
    '''
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    print0(pcolor(f'infer_depth(\n\t{ckpt},\n\t{input_dir},\n\t{output_dir})', 'yellow'))

    model_wrapper, image_shape = init_model(ckpt)
    print0(pcolor(f'  input image shape:{image_shape}', 'yellow'))
    files = []
    for ext in ['png', 'jpg', 'bmp']:
        files.extend(glob((osp.join(input_dir, f'*.{ext}'))))
    files.sort()
    print(f'{len(files)} in {input_dir}')
    for idx, item in enumerate(files):
        logging.debug(f'[{idx:4d}] {osp.basename(item)}')
    list_of_files = list(zip(files[:-2],
                             files[1:-1],
                             files[2:]))
    time_beg_run_model = time.time()
    for idx_frame, fns in enumerate(list_of_files):
        fn1, fn2, fn3 = fns
        print0(pcolor(f'{osp.basename(fn1)} vs {osp.basename(fn2)} vs {osp.basename(fn3)}', 'yellow'))
        base_name = osp.splitext(osp.basename(fn2))[0]
        image_raw_wh = load_image(fn2).size

        input_file_refs = [fn1, fn3]
        image_ref = [process_image(input_file_ref, image_raw_wh, image_shape)[0] for input_file_ref in input_file_refs]
        image, intrinsics = process_image(fn2, image_raw_wh, image_shape)

        batch = {'rgb': image, 'rgb_context': image_ref, "intrinsics": intrinsics}

        output = model_wrapper(batch)
        inv_depth = output['inv_depths'][0] #(1, 1, h, w)
        depth = inv2depth(inv_depth)[0, 0].detach().cpu().numpy() #(h, w)

        depth_upsample = cv2.resize(depth, image_raw_wh, interpolation=cv2.INTER_NEAREST)
        np.save(os.path.join(output_dir, f"{base_name}.npy"), depth_upsample)

        enable_debug = False
        if enable_debug:
            print(f'  depth:          {depth.shape} {depth.dtype}')
            print(f'  depth_upsample: {depth_upsample.shape} {depth_upsample.dtype}')
            image_cpu = image.detach().cpu().numpy()
            print(f'  image_cpu:      {image_cpu.shape} {image_cpu.dtype}')
            debug_path = osp.abspath(osp.join(output_dir, '../input_image'))
            if not osp.exists(debug_path):
                os.makedirs(debug_path)
            np.save(osp.join(debug_path, f'{base_name}.npy'), image_cpu)
            print(f'  intrinsics:\n{intrinsics}')

        enable_cloud = True
        if enable_cloud:
            depth_map = DepthMap(depth_upsample, cv2.imread(fn2), base_name, sample=3)
            save_name = osp.join(cloud_dir, model_name + '_' + base_name + '.obj')
            cloud = depth_map.get_cloud(True)
            depth_map.save_obj(cloud, save_name)
            pass

    time_end_run_model = time.time()
    time_diff = time_end_run_model - time_beg_run_model
    fps = np.float(len(files) - 2) / time_diff
    print0(pcolor(f'  fps: {fps:.3f}', 'yellow'))


def synthetic_canvas(grid, frame_id, name, dataset_dir, model_names, is_first,  prev_min, prev_max, prev_mean):
    name_color = osp.join(dataset_dir, 'cam_left', name + '.jpg')
    color_img = cv2.imread(name_color)
    cv2.putText(img=color_img, text=f'[{frame_id:4d}] {name}',
        org=(30, 50), fontScale=1, color=(0, 0, 255), thickness=2, fontFace=cv2.LINE_AA)
    grid.subplot(0, 0, color_img, f'cam_left')

    for idx, item in enumerate(model_names):
        idx_row = (idx + 1) // grid.grid_col
        idx_col = (idx + 1) % grid.grid_col
        name_depth = osp.join(dataset_dir, 'infer_depth', item, name + '.npy')
        depth_img = np.load(name_depth)
        # depth_vis = viz_inv_depth(depth_img.astype(np.float) / 1000., normalizer=10.0, filter_zeros=True) * 255
        depth_vis = viz_inv_depth(depth_img.astype(np.float) / 1000., normalizer=0.01, filter_zeros=True) * 255

        if is_first:
            cv2.putText(img=depth_vis, text=f'min_depth:  {np.amin(depth_img):.6f}',
                org=(30, 50), fontScale=1, color=(0, 0, 255), thickness=2, fontFace=cv2.LINE_AA)
            cv2.putText(img=depth_vis, text=f'max_depth:  {np.amax(depth_img):.6f}',
                org=(30, 100), fontScale=1, color=(0, 0, 255), thickness=2, fontFace=cv2.LINE_AA)
            cv2.putText(img=depth_vis, text=f'mean_depth: {np.mean(depth_img):.6f}',
                org=(30, 150), fontScale=1, color=(0, 0, 255), thickness=2, fontFace=cv2.LINE_AA)
            prev_min[idx] = np.amin(depth_img)
            prev_max[idx] = np.amax(depth_img)
            prev_mean[idx] = np.mean(depth_img)

        else:
            cv2.putText(img=depth_vis, text=f'min_depth:  {np.amin(depth_img):.6f}  diff_to_prev: {np.amin(depth_img) - prev_min[idx]:.6f}',
                org=(30, 50), fontScale=1, color=(0, 0, 255), thickness=2, fontFace=cv2.LINE_AA)
            cv2.putText(img=depth_vis, text=f'max_depth:  {np.amax(depth_img):.6f}  diff_to_prev: {np.amax(depth_img) - prev_max[idx]:.6f}',
                org=(30, 100), fontScale=1, color=(0, 0, 255), thickness=2, fontFace=cv2.LINE_AA)
            cv2.putText(img=depth_vis, text=f'mean_depth: {np.mean(depth_img):.6f}  diff_to_prev: {np.mean(depth_img) - prev_mean[idx]:.6f}',
                org=(30, 150), fontScale=1, color=(0, 0, 255), thickness=2, fontFace=cv2.LINE_AA)

            prev_min[idx] = np.amin(depth_img)
            prev_max[idx] = np.amax(depth_img)
            prev_mean[idx] = np.mean(depth_img)

        grid.subplot(idx_row, idx_col, depth_vis[:, :, ::-1], f'{item}')
    return grid, prev_min, prev_max, prev_mean


def depth_to_video(dataset_dir, model_names):
    depth_dir = osp.join(dataset_dir, 'infer_depth', model_names[0])
    if not osp.exists(depth_dir):
        raise ValueError

    base_names = []
    for item in sorted(os.listdir(depth_dir)):
        str_name, str_ext = osp.splitext(item)
        if str_ext != '.npy':
            raise ValueError
        base_names.append(str_name)

    print0(pcolor(f'{len(base_names):4d} in {dataset_dir}', 'yellow'))

    video_name = dataset_dir + '_depth_demo.avi'
    im_h, im_w = 720, 1280
    n_row, n_col = 2, 3

    grid = VizImageGrid(im_h, im_w, n_row, n_col)
    canvas_row = grid.canvas_row
    canvas_col = grid.canvas_col
    fps = 1.0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(video_name, fourcc, fps, (canvas_col, canvas_row))
    print0(pcolor(f'  writing {video_name}', 'cyan'))

    is_first = True
    n_model = len(model_names)
    prev_min = [0.] * n_model
    prev_max = [0.] * n_model
    prev_mean = [0.] * n_model

    for frame_id, item in enumerate(base_names):
        grid, prev_min, prev_max, prev_mean = synthetic_canvas(grid, frame_id, item, dataset_dir, model_names, is_first, prev_min, prev_max, prev_mean)

        is_first = False
        video_writer.write(grid.canvas)
    video_writer.release()

def main():
    logging.warning(f'main()')
    ckpt = '/home/sigma/slam/models/lynx@27/SupModelMF_DepthPoseNet_it12-h-out_epoch=116_matterport0516_ex-test_all_list-groundtruth-abs_rel_pp_gt=0.265.ckpt'
    input_dir = '/home/sigma/slam/gazebo0629/0701_2022_sim/cam_left'
    output_dir = '/home/sigma/slam/gazebo0629/0701_2022_sim/infer_depth'
    infer_depth(ckpt, input_dir, output_dir)

    output_ref = '/home/sigma/slam/gazebo0629/0701_2022_sim/infer_video/tmp/depth'

    compare_depth(output_dir, output_ref)

    input_image_dir = '/home/sigma/slam/gazebo0629/0701_2022_sim/input_image'
    input_image_ref = '/home/sigma/slam/gazebo0629/0701_2022_sim/infer_video/tmp/input_image'
    compare_depth(input_image_dir, input_image_ref)

'''
trex@24
scp xuelian@10.200.210.24:/home/xuelian/slam/dro-sfm/results/model/matterport_gt/SupModelMF_DepthPoseNet_it12-h-out_epoch=653_matterport0516-val_all_list-groundtruth-abs_rel_pp_gt=0.067.ckpt ~/slam/models/trex@24/

fox@26
scp xuelian@10.200.210.26:/home/xuelian/slam/dro-sfm/results/model/matterport_gt/SupModelMF_DepthPoseNet_it12-h-out_epoch=1375_matterport0516_ex-val_all_list-groundtruth-abs_rel_pp_gt=0.064.ckpt ~/slam/models/fox@26/

lynx@27
scp xuelian@10.200.210.27:/home/xuelian/slam/dro-sfm/results/model/matterport_gt/SupModelMF_DepthPoseNet_it12-h-out_epoch=198_matterport0516_ex-test_all_list-groundtruth-abs_rel_pp_gt=0.268.ckpt ~/slam/models/lynx@27/

x@28
scp xuelian@10.200.210.28:/home/xuelian/slam/dro-sfm/results/model/matterport_gt/SupModelMF_DepthPoseNet_it12-h-out_epoch=146_matterport0516_ex-test_all_list-groundtruth-abs_rel_pp_gt=0.271.ckpt ~/slam/models/x@28/
'''

def main_ex():
    ckpt = OrderedDict()

    ckpt['scannet'] = '/mnt/datasets_open/dro-sfm_data/models/indoor_scannet.ckpt'
    # ckpt['trex@24'] = '/home/sigma/slam/models/trex@24/SupModelMF_DepthPoseNet_it12-h-out_epoch=403_matterport0516-val_all_list-groundtruth-abs_rel_pp_gt=0.067.ckpt'
    # ckpt['fox@26'] = '/home/sigma/slam/models/fox@26/SupModelMF_DepthPoseNet_it12-h-out_epoch=484_matterport0516_ex-val_all_list-groundtruth-abs_rel_pp_gt=0.064.ckpt'
    # ckpt['lynx@27'] = '/home/sigma/slam/models/lynx@27/SupModelMF_DepthPoseNet_it12-h-out_epoch=116_matterport0516_ex-test_all_list-groundtruth-abs_rel_pp_gt=0.265.ckpt'
    # ckpt['x@28'] = '/home/sigma/slam/models/x@28/SupModelMF_DepthPoseNet_it12-h-out_epoch=97_matterport0516_ex-test_all_list-groundtruth-abs_rel_pp_gt=0.273.ckpt'

    ckpt['lynx@27'] = '/home/sigma/slam/models/lynx@27/SupModelMF_DepthPoseNet_it12-h-out_epoch=198_matterport0516_ex-test_all_list-groundtruth-abs_rel_pp_gt=0.268.ckpt'
    ckpt['x@28'] = '/home/sigma/slam/models/x@28/SupModelMF_DepthPoseNet_it12-h-out_epoch=146_matterport0516_ex-test_all_list-groundtruth-abs_rel_pp_gt=0.271.ckpt'
    ckpt['trex@24'] = '/home/sigma/slam/models/trex@24/SupModelMF_DepthPoseNet_it12-h-out_epoch=653_matterport0516-val_all_list-groundtruth-abs_rel_pp_gt=0.067.ckpt'
    ckpt['fox@26'] = '/home/sigma/slam/models/fox@26/SupModelMF_DepthPoseNet_it12-h-out_epoch=1375_matterport0516_ex-val_all_list-groundtruth-abs_rel_pp_gt=0.064.ckpt'

    root_dir = '/home/sigma/slam/gazebo0629_v0'
    datasets = ['0628_2022_line_sim', '0628_2022_sim', '0629_2022_sim', '0701_2022_sim']
    model_names = list(ckpt.keys())

    enable_infer_depth = True
    if enable_infer_depth:
        for item_model in model_names:
            ckpt_file = ckpt[item_model]
            for item_dataset in datasets:
                path_dataset = osp.join(root_dir, item_dataset)
                input_dir = osp.join(path_dataset, 'cam_left')
                output_dir = osp.join(path_dataset, 'infer_depth', item_model)
                if not osp.exists(input_dir):
                    print0(pcolor(f'skip {input_dir}', 'yellow'))
                    continue
                if not osp.exists(output_dir):
                    os.makedirs(output_dir)

                cloud_dir = osp.join(path_dataset, 'point_cloud', item_model)
                if not osp.exists(cloud_dir):
                    os.makedirs(cloud_dir)

                infer_depth(ckpt_file, input_dir, output_dir, cloud_dir, item_model)

    enable_viz_depth = True
    if enable_viz_depth:
        for item_dataset in datasets:
            path_dataset = osp.join(root_dir, item_dataset)
            depth_to_video(path_dataset, model_names)


class DepthMap:
    def __init__(self, depth_data, color_data, name, sample=1, scale=1.):
        self.depth = depth_data
        self.color = color_data
        self.name = name
        self.scale = scale
        self.sample = sample

        h, w = depth_data.shape
        self.im_h = h
        self.im_w = w
        self.cx = 0.5 * (np.float(self.im_w) - 1.0)
        self.cy = 0.5 * (np.float(self.im_h) - 1.0)
        self.f = 577.870605
        self.fx = self.f
        self.fy = self.f
        self.intr = np.array([[self.f,      0.,  self.cx],
                              [    0.,  self.f,  self.cy],
                              [    0.,      0.,       1.]], dtype=np.float)

    def intr(self):
        return self.intr

    def get_cloud(self, valid_only=True):
        points = []
        h, w = self.depth.shape
        cloud = np.zeros((h*w, 6), dtype=np.float32)
        n_valid = 0

        for v in range(0, h, self.sample):
            for u in range(0, w, self.sample):
                color = self.color[v, u]
                Z = self.depth[v, u] / self.scale
                if Z <= 0: continue
                X = (u - self.cx) * Z / self.fx
                Y = (v - self.cy) * Z / self.fy

                points.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, color[0], color[1], color[2]))

                cloud[n_valid, 0] = X
                cloud[n_valid, 1] = Y
                cloud[n_valid, 2] = Z
                cloud[n_valid, 3] = float(color[0])
                cloud[n_valid, 4] = float(color[1])
                cloud[n_valid, 5] = float(color[2])
                n_valid += 1

        if valid_only:
            if n_valid == 0:
                logging.warning(f'invalid depth: {self.name}')
            return cloud[:n_valid, :]
        return cloud

    @staticmethod
    def save_obj(cloud, name):
        dirname = osp.dirname(name)
        if not osp.exists(dirname):
            os.makedirs(dirname)

        with open(name, 'wt') as f_ou:
            n_vert = cloud.shape[0]
            for i in range(n_vert):
                x, y, z, r, g, b = cloud[i, :]
                f_ou.write(f'v {x} {y} {z} {r} {g} {b}\n')


def viz_depth():
    root_dir = '/home/sigma/slam/gazebo0629/0701_2022_sim'
    color_name = 'cam_left'
    depth_name = 'infer_video/tmp/depth'
    save_name = 'infer_video/tmp/cloud'

    color_dir = osp.join(root_dir, color_name)
    depth_dir = osp.join(root_dir, depth_name)
    save_dir = osp.join(root_dir, save_name)
    if not osp.exists(save_dir):
        os.mkdir(save_dir)

    for idx, item in enumerate(sorted(os.listdir(depth_dir))):
        basename = osp.splitext(item)[0]
        npy_file = osp.join(depth_dir, item)
        data = np.load(npy_file)
        print(f'[{idx:4d}] {item} {data.shape} {data.dtype} max: {np.amax(data)}, min: {np.amin(data)}, mean: {np.mean(data)}')

        depth_data = np.load(npy_file)
        color_data = cv2.imread(osp.join(color_dir, basename + '.jpg'))
        depth_map = DepthMap(depth_data, color_data, basename)
        save_name = osp.join(save_dir, basename + '.obj')
        cloud = depth_map.get_cloud(True)
        depth_map.save_obj(cloud, save_name)


if __name__ == '__main__':
    setup_log('kneron_infer_depth.log')
    time_beg_infer_depth = time.time()

    np.set_printoptions(precision=6, suppress=True)
    # main()
    main_ex()
    # viz_depth()

    time_end_infer_depth = time.time()
    logging.warning(f'infer_depth.py elapsed {time_end_infer_depth - time_beg_infer_depth:.6f} seconds.')
    print0(pcolor(f'\ninfer_depth.py elapsed {time_end_infer_depth - time_beg_infer_depth:.6f} seconds.\n', 'yellow'))
