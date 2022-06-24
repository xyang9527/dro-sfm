# -*- coding=utf-8 -*-

from typing import List
import os.path as osp
import sys
lib_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(lib_dir)
import numpy as np
import torch

from dro_sfm.geometry.pose_trans import matrix_to_euler_angles


def clip_depth(depth):
    """
    Parameters
    ----------
    depth: np.ndarray
        depth data

    Returns
    -------
    np.ndarray
    """
    # depth threshold
    clip_thr_depth_min = 400   #  0.4 m
    clip_thr_depth_max = 10000 # 10.0 m

    clip_mask_max = depth > clip_thr_depth_max
    clip_mask_min = depth < clip_thr_depth_min
    clip_mask = np.logical_or(clip_mask_max, clip_mask_min)
    depth[clip_mask] = 0

    return depth


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
