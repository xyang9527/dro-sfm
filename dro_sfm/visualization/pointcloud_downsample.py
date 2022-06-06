# -*- coding=utf-8 -*-

import numpy as np


def generate_pointcloud_NxN(rgb, depth, fx, fy, cx, cy, ply_file, sample_x, sample_y, valid_only, scale=1.0):
    """
    Generate a colored point cloud in PLY format from a color and a depth image.

    Input:
    rgb_file   -- filename of color image
    depth_file -- filename of depth image
    ply_file   -- filename of ply file
    sample_x: int
    sample_y: int
    valid_only: bool

    Reference:
    scripts/infer.py def generate_pointcloud(..)
    """
    # fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    points = []
    h, w = depth.shape
    cloud = np.zeros((h*w, 6), dtype=np.float32)
    n_valid = 0
    for v in range(0, rgb.shape[0], sample_y):
        for u in range(0, rgb.shape[1], sample_x):
            color = rgb[v, u] #rgb.getpixel((u, v))
            Z = depth[v, u] / scale
            if Z <= 0: continue
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            points.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, color[0], color[1], color[2]))
            cloud[n_valid, 0] = X
            cloud[n_valid, 1] = Y
            cloud[n_valid, 2] = Z
            cloud[n_valid, 3] = float(color[0])
            cloud[n_valid, 4] = float(color[1])
            cloud[n_valid, 5] = float(color[2])
            n_valid += 1

    file = open(ply_file, "w")
    file.write('''ply
            format ascii 1.0
            element vertex %d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            property uchar alpha
            end_header
            %s
            ''' % (len(points), "".join(points)))
    file.close()

    if valid_only:
        return cloud[:n_valid, :]

    return cloud
