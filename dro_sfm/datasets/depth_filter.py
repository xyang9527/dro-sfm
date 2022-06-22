# -*- coding=utf-8 -*-

import numpy as np

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
