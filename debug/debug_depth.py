# -*- coding=utf-8 -*-

import os
import os.path as osp
import sys
lib_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(lib_dir)
import logging
import numpy as np
from PIL import Image
import time

from dro_sfm.utils.setup_log import setup_log
from dro_sfm.visualization.viz_image_grid import VizImageGrid, Colors
from dro_sfm.utils.horovod import print0
from dro_sfm.utils.logging import pcolor
from dro_sfm.visualization.viz_datasets import get_datasets


def depth_discontinuous():
    logging.warning(f'depth_discontinuous()')

    name = '000911261000000'
    dataset = '/home/sigma/slam/matterport0614/train_val_test/matterport071_0614'

    depth_path = osp.join(dataset, 'depth', name + '.png')
    depth_png = np.array(Image.open(depth_path), dtype=int)
    print(f'depth_png: {type(depth_png)} {depth_png.dtype} {depth_png.shape}')
    print(f'  amax:   {np.amax(depth_png)}')
    print(f'  amin:   {np.amin(depth_png)}')
    print(f'  mean:   {np.mean(depth_png)}')
    print(f'  median: {np.median(depth_png)}')
    unique_depth, occur_count = np.unique(depth_png, return_counts=True)
    print(f'unique_depth: {type(unique_depth)} {unique_depth.dtype} {unique_depth.shape} {unique_depth.tolist()}')
    print(f'occur_count:  {type(occur_count)} {occur_count.dtype} {occur_count.shape} {occur_count.tolist()}')

    '''
    unique_value_max = 0
    depth_dir = osp.join(dataset, 'depth')
    for item in os.listdir(depth_dir):
        filename = osp.join(depth_dir, item)
        depth_png = np.array(Image.open(filename), dtype=int)
        unique_depth, occur_count = np.unique(depth_png, return_counts=True)

        if unique_value_max < unique_depth.shape[0]:
            unique_value_max = unique_depth.shape[0]
        print(f'{osp.splitext(item)[0]}:')
        for v, n in zip(unique_depth.tolist(), occur_count.tolist()):
            print(f'  {v:6d} : {n:6d}')

    print0(pcolor(f'unique_value_max: {unique_value_max}', 'yellow'))
    '''


def depth_discontinuous_ex():
    logging.warning(f'depth_discontinuous_ex()')

    datasets = get_datasets()

    for k, v in datasets.items():
        for base in v:
            unique_value_max = 0
            depth_dir = osp.join(base, 'depth')

            for item in os.listdir(depth_dir):
                filename = osp.join(depth_dir, item)
                depth_png = np.array(Image.open(filename), dtype=int)
                unique_depth, _ = np.unique(depth_png, return_counts=True)

                if unique_value_max < unique_depth.shape[0]:
                    unique_value_max = unique_depth.shape[0]
            print0(pcolor(f'  {osp.basename(base):30s}: {unique_value_max:8d}', 'yellow'))
        # // for base in v:
    # // for k, v in datasets.items():


if __name__ == '__main__':
    setup_log('kneron_debug_depth.log')
    time_beg_debug_depth = time.time()

    depth_discontinuous()
    depth_discontinuous_ex()

    time_end_debug_depth = time.time()
    print0(pcolor(f'debug_depth.py elapsed {time_end_debug_depth - time_beg_debug_depth:.6f} seconds.', 'red'))
