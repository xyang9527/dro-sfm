# -*- coding=utf-8 -*-

from typing import List
import os
import os.path as osp
import sys
lib_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(lib_dir)
import logging
import numpy as np
from PIL import Image
import time
import yaml
import torch

from dro_sfm.utils.setup_log import setup_log
from dro_sfm.visualization.viz_image_grid import VizImageGrid, Colors
from dro_sfm.utils.horovod import print0
from dro_sfm.utils.logging import pcolor
from dro_sfm.geometry.pose_trans import matrix_to_euler_angles
from dro_sfm.visualization.viz_datasets import get_datasets
from dro_sfm.datasets.depth_filter import clip_depth


def is_invalid_pose(pose: np.ndarray):
    """
    Returns
    -------
    bool
    """
    has_illegal_value = False
    h, w = pose.shape

    for j in range(h):
        if has_illegal_value:
            break

        for i in range(w):
            v = pose[j, i]
            if np.isnan(v) or np.isneginf(v) or np.isposinf(v):
                has_illegal_value = True
                break
    return has_illegal_value


def find_idx_of_prev_n(arr_is_invalid: List[bool], arr_accum_valid: List[int], curr_idx: int, prev_n: int):
    """
    Returns
    -------
    int
    """
    assert curr_idx > 0, f'curr_idx = {curr_idx}'
    assert prev_n > 0, f'prev_n = {prev_n}'
    assert arr_accum_valid[curr_idx - 1] >= prev_n

    n = prev_n
    for idx in range(curr_idx - 1, -1, -1):
        if arr_is_invalid[idx]:
            continue
        n -= 1
        if n == 0:
            return idx
    raise ValueError


def matrix_to_6d_pose(pose_curr: np.ndarray, pose_prev: np.ndarray):
    """
    Returns
    -------
    List[float]
    """
    rel_pose = np.matmul(np.linalg.inv(pose_prev), pose_curr)

    xyz = matrix_to_euler_angles(torch.from_numpy(rel_pose[:3, :3]), 'XYZ')
    xyz_degree = xyz.detach() * 180.0 / np.math.pi
    d_rx, d_ry, d_rz = xyz_degree[:]
    d_tx, d_ty, d_tz = rel_pose[0, 3] * 1000.0, rel_pose[1, 3] * 1000.0, rel_pose[2, 3] * 1000.0
    return [d_tx, d_ty, d_tz, d_rx, d_ry, d_rz]


def pose_in_thr(pose_6d: List[float], d_t, d_ts, d_r, d_rs):
    """
    Returns
    -------
    bool
    """
    tx, ty, tz, rx, ry, rz = pose_6d
    ts = np.math.sqrt(tx ** 2 + ty ** 2 + tz ** 2)
    rs = np.math.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
    if ts > d_ts or rs > d_rs:
        return False

    for t in [tx, ty, tz]:
        if np.abs(t) > np.abs(d_t):
            return False

    for r in [rx, ry, rz]:
        if np.abs(r) > np.abs(d_r):
            return False
    return True


def pose_in_threshold_1(pose_6d: List[float]):
    """
    Returns
    -------
    bool
    """
    # ref: statistical info of viz_scene0600_00.avi
    thr_d_t = 90.0    # mm
    thr_d_ts = 120.0  # mm
    thr_d_r = 5.0     # degree
    thr_d_rs = 7.5    # degree
    return pose_in_thr(pose_6d, thr_d_t, thr_d_ts, thr_d_r, thr_d_rs)


def pose_in_threshold_5(pose_6d: List[float]):
    """
    Returns
    -------
    bool
    """
    thr_d_t = 145.0   # mm
    thr_d_ts = 205.0  # mm
    thr_d_r = 14.5    # degree
    thr_d_rs = 21.5   # degree
    return pose_in_thr(pose_6d, thr_d_t, thr_d_ts, thr_d_r, thr_d_rs)


def sequence_filter():
    logging.warning('sequence_filter()')
    datasets = get_datasets()

    matterport_seqs = datasets['matterport']
    # matterport_seqs = ['/home/sigma/slam/matterport0621/test/matterport005_0621']
    for idx_seq, item_seq in enumerate(matterport_seqs):
        print(f'  -> {idx_seq:2d} : {item_seq}')

        pose_dir = osp.join(item_seq, 'pose')
        depth_dir = osp.join(item_seq, 'depth')

        pose_data = []
        # load pose
        for item_file in sorted(os.listdir(pose_dir)):
            str_name, str_ext = osp.splitext(item_file)
            if str_ext != '.txt':
                continue

            # valid pose only
            pose = np.genfromtxt(osp.join(pose_dir, item_file))
            if is_invalid_pose(pose):
                print(f'  invalid pose: {osp.join(pose_dir, item_file)}')
                continue

            # depth info
            depth_name = osp.join(depth_dir, str_name + '.png')
            if not osp.exists(depth_name):
                continue

            depth_png = np.array(Image.open(depth_name), dtype=int)
            depth_png = clip_depth(depth_png)
            unique_depth = np.unique(depth_png)
            num_unique_depth = unique_depth.shape[0]

            depth_mask = depth_png <= 0
            num_pix_invalid = np.count_nonzero(depth_mask)
            num_pix_total = depth_png.shape[0] * depth_png.shape[1]

            pose_data.append((str_name, pose, num_unique_depth, num_pix_invalid, num_pix_total))

        # archive (debug)
        save_dir = osp.join(item_seq, 'filtered_split')
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

        save_name = osp.join(save_dir, 'all.txt')
        with open(save_name, 'w') as f_ou:
            for idx, item in enumerate(pose_data):
                v0, v1, v2, v3, v4 = item
                f_ou.write(f'[{idx:4d}] {v0} {v2} {v3} {v4}\n')

        n_frames = len(pose_data)
        to_drop = [False] * n_frames
        to_split = [False] * n_frames
        num_valid = [0] * n_frames
        inited = False
        for idx_frame, item_data in enumerate(pose_data):
            str_name, pose, num_unique_depth, num_pix_invalid, num_pix_total = item_data
            # depth threshold
            if np.float(num_pix_invalid) / np.float(num_pix_total) > 0.4:
                to_drop[idx_frame] = True
                # print0(pcolor(f'    drop {str_name}', 'yellow'))
                logging.info(f'    drop {str_name}')
                continue

            # pose threshold - prev 1
            if inited and num_valid[idx_frame - 1] >= 1:
                idx_targ = find_idx_of_prev_n(to_drop, num_valid, idx_frame, 1)
                _, pose_targ, _, _, _ = pose_data[idx_targ]
                pose_6d = matrix_to_6d_pose(pose, pose_targ)
                if not pose_in_threshold_1(pose_6d):
                    to_split[idx_frame] = True
                    continue

            # pose threshold - prev 5
            if inited and num_valid[idx_frame - 1] >= 5:
                pass

            if not inited:
                num_valid[idx_frame] = 1
                inited = True
            else:
                num_valid[idx_frame] = num_valid[idx_frame - 1] + 1

        # // for idx_frame, item_data in enumerate(pose_data)
        print(f'  total frames: {len(pose_data)}')
        print(f'  valid frames: {len(pose_data) - sum(to_drop)}')
        print(f'  split:        {sum(to_split)}')

        name_drop = osp.join(save_dir, 'invalid.txt')
        name_seq_all = osp.join(save_dir, 'seq_all.txt')
        with open(name_drop, 'w') as f_ou_drop, open(name_seq_all, 'w') as f_ou_all:
            words = item_seq.split('/')
            sub_seq = 0
            names_sub_seq = []
            for i in range(n_frames):
                str_name, _, _, _, _ = pose_data[i]
                if to_drop[i]:
                    f_ou_drop.write(f'{words[-2]}/{words[-1]}/cam_left {str_name}.jpg\n')
                    continue
                if not to_split[i]:
                    names_sub_seq.append(str_name)
                else:
                    name_seq = osp.join(save_dir, f'seq_{sub_seq:02d}.txt')
                    with open(name_seq, 'w') as f_ou_seq:
                        for idx_name in names_sub_seq:
                            f_ou_seq.write(f'{words[-2]}/{words[-1]}/cam_left {idx_name}.jpg\n')
                            f_ou_all.write(f'{words[-2]}/{words[-1]}/cam_left {idx_name}.jpg {sub_seq:2d}\n')
                    names_sub_seq.clear()
                    names_sub_seq.append(str_name)
                    sub_seq += 1
        # break
    # // for idx_seq, item_seq in enumerate(matterport_seqs):


if __name__ == '__main__':
    setup_log('kneron_matterport_filter.log')
    time_beg_matterport_filter = time.time()

    sequence_filter()
    # for idx, i in enumerate(range(10, 0, -1)):
    #    print(f' [{idx:4d}] -> {i}')

    '''
    T05 = np.array([
                [-1.,  0.,  0.,  0.],
                [ 0.,  0.,  1.,  0.],
                [ 0.,  1.,  0.,  0.],
                [ 0.,  0.,  0.,  1.]], dtype=np.float)

    xyz = matrix_to_euler_angles(torch.from_numpy(T05[:3, :3]), 'XYZ')
    xyz_degree = xyz.detach() * 180.0 / np.math.pi
    d_rx, d_ry, d_rz = xyz_degree[:]
    print(f'  d_rx: {d_rx}')
    print(f'  d_ry: {d_ry}')
    print(f'  d_rz: {d_rz}')
    '''

    time_end_matterport_filter = time.time()
    print0(pcolor(f'matterport_filter.py elapsed {time_end_matterport_filter - time_beg_matterport_filter:.6f} seconds.', 'red'))
