# -*- coding=utf-8 -*-

from typing import Dict
from collections import OrderedDict
import logging
import os
import os.path as osp
from unittest.loader import VALID_MODULE_NAME

import numpy as np
import cv2

from mpl_toolkits.mplot3d import Axes3D  # implicit used
import matplotlib
import matplotlib.pyplot as plt


def read_obj(filename):
    if not osp.exists(filename):
        print(f'path not exist: {filename}')
        raise ValueError

    coord = []
    with open(filename, 'r') as f_in:
        text = f_in.readlines()
    for line in text:
        line = line.strip()
        if len(line) < 2:
            continue
        if line[0] == 'v':
            _, x, y, z = line.split(' ')
            coord.append([np.float(x), np.float(y), np.float(z)])
    # if len(coord) <= 0:
        # raise ValueError
    return np.array(coord)


class TrajectoryData:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.coord = read_obj(path)

    @property
    def n_vert(self):
        return self.coord.shape[0]

    def get_name(self):
        return self.name

    def get_path(self):
        return self.path


    def coord(self):
        return self.coord

    def scaled_coord(self, scale):
        return self.coord * scale


class VizTraj3D:
    def __init__(self, win_name='VizTraj3D'):
        self.main_fig = plt.figure(win_name, figsize=(16, 8))
        plt.clf()
        elev = -40
        azim = -80
        self.ax_left = self.main_fig.add_subplot(1, 2, 1, projection='3d',
                                            elev=elev, azim=azim)
        self.ax_right = self.main_fig.add_subplot(1, 2, 2, projection='3d',
                                            elev=elev, azim=azim)

    def close(self):
        plt.close()

    def left_draw_lines(self, x_arr, y_arr, z_arr, label, color):
        self.ax_left.set_zticks(np.arange(-3.0, 3.0, step=1.0))
        self.ax_left.set_zbound(-3.0, 3.0)

        self.ax_left.plot(x_arr, y_arr, z_arr, color=color, label=label)
        self.ax_left.set_xlabel('$X$', fontsize=20, color='red')
        self.ax_left.set_ylabel('$Y$', fontsize=20, color='green')
        self.ax_left.set_zlabel('$Z$', fontsize=20, color='blue')
        self.ax_left.set_zlim(-3.0, 3.0)

        self.ax_left.legend()

    def right_draw_lines(self, x_arr, y_arr, z_arr, label, color):
        self.ax_right.set_zticks(np.arange(-3.0, 3.0, step=1.0))
        self.ax_right.set_zbound(-3.0, 3.0)

        self.ax_right.plot(x_arr, y_arr, z_arr, color=color, label=label)
        self.ax_right.set_xlabel('$X$', fontsize=20, color='red')
        self.ax_right.set_ylabel('$Y$', fontsize=20, color='green')
        self.ax_right.set_zlabel('$Z$', fontsize=20, color='blue')
        self.ax_right.set_zlim(-3.0, 3.0)

        self.ax_right.legend()

    @staticmethod
    def show():
        def move_figure(f, x, y):
            """Move figure's upper left corner to pixel (x, y)"""
            backend = matplotlib.get_backend()
            if backend == 'TkAgg':
                f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
            elif backend == 'WXAgg':
                f.canvas.manager.window.SetPosition((x, y))
            else:
                # This works for QT and GTK
                # You can also use window.setGeometry
                f.canvas.manager.window.move(x, y)

        thismanager = plt.get_current_fig_manager()
        move_figure(thismanager, 1920, 0)
        plt.show()


class VizTraj2D:
    def __init__(self, win_name='VizTraj2D'):
        self.bbox_min = [0] * 3
        self.bbox_max = [0] * 3
        self.dim = [0] * 3
        self.dim_xyz = 0
        self.bbox_scale = 1.5

        fig, axs = plt.subplots(1, 3, figsize=(20, 12), dpi=80, constrained_layout=False)
        self.main_fig = fig
        self.subfig_xoy = axs[0]
        self.subfig_yoz = axs[1]
        self.subfig_xoz = axs[2]

        self.main_fig.suptitle(win_name, fontsize=35, color='cyan')

    def close(self):
        plt.close()

    def update_bbox(self, x, y, z):
        xyz = [x, y, z]
        for i in range(3):
            v_min, v_max = min(xyz[i]), max(xyz[i])
            if self.bbox_min[i] > v_min:
                self.bbox_min[i] = v_min
            if self.bbox_max[i] < v_max:
                self.bbox_max[i] = v_max

            if self.dim[i] < np.fabs(v_min):
                self.dim[i] = np.fabs(v_min)
            if self.dim[i] < np.fabs(v_max):
                self.dim[i] = np.fabs(v_max)
            if self.dim_xyz < self.dim[i]:
                self.dim_xyz = self.dim[i]

    def draw_lines(self, x_arr, y_arr, z_arr, label, color):
        self.update_bbox(x_arr, y_arr, z_arr)

        # xoy
        self.subfig_xoy.plot(x_arr, y_arr, label=label, color=color)
        self.subfig_xoy.set_xlabel('X', fontsize=18, color='red')
        self.subfig_xoy.set_ylabel('Y', fontsize=18, color='green')
        self.subfig_xoy.set_xlim(-self.dim[0] * self.bbox_scale, self.dim[0] * self.bbox_scale)
        self.subfig_xoy.set_ylim(-self.dim[1] * self.bbox_scale, self.dim[1] * self.bbox_scale)
        self.subfig_xoy.legend(loc='upper left')
        self.subfig_xoy.set_title('XOY Projeciton', fontsize=25, color='blue')

        # yoz
        self.subfig_yoz.plot(y_arr, z_arr, label=label, color=color)
        self.subfig_yoz.set_xlabel('Y', fontsize=18, color='green')
        self.subfig_yoz.set_ylabel('Z', fontsize=18, color='blue')
        self.subfig_yoz.set_xlim(-self.dim[1] * self.bbox_scale, self.dim[1] * self.bbox_scale)
        self.subfig_yoz.set_ylim(-self.dim[2] * self.bbox_scale, self.dim[2] * self.bbox_scale)
        self.subfig_yoz.legend(loc='upper left')
        self.subfig_yoz.set_title('YOZ Projeciton', fontsize=25, color='red')

        # xoz
        self.subfig_xoz.plot(x_arr, z_arr, label=label, color=color)
        self.subfig_xoz.set_xlabel('X', fontsize=18, color='red')
        self.subfig_xoz.set_ylabel('Z', fontsize=18, color='blue')
        self.subfig_xoz.set_xlim(-self.dim[0] * self.bbox_scale, self.dim[0] * self.bbox_scale)
        self.subfig_xoz.set_ylim(-self.dim[2] * self.bbox_scale, self.dim[2] * self.bbox_scale)
        self.subfig_xoz.legend(loc='upper left')
        self.subfig_xoz.set_title('XOZ Projeciton', fontsize=25, color='green')

    @staticmethod
    def show():
        # plt.savefig(fig_path, dpi=100)
        plt.show()


class VizTrajectory:
    def __init__(self, name, obj_gt, obj_info):
        self.name = name
        self.data_gt = TrajectoryData('GroundTruth', obj_gt)
        self.datas_pred = []
        for k, v in obj_info.items():
            self.datas_pred.append(TrajectoryData(k, v))

        if len(self.datas_pred) <= 0:
            raise ValueError

        self.scale_bbox = None  # bbox based scale
        self.scale_length = None  # length based scale

        self.check_data()
        self.calc_scale()
        self.debug()

    def debug(self):
        print(f'{self.name}:')
        print(f'  n_vert: {self.data_gt.n_vert}')
        n_pred = len(self.datas_pred)

        assert len(self.scale_bbox) == n_pred
        assert len(self.scale_length) == n_pred

        for i in range(n_pred):
            print(f'  {self.datas_pred[i].name:15s} {self.scale_bbox[i]:.6f} {self.scale_length[i]:.6f}')


    def check_data(self):
        n_vert = self.data_gt.n_vert
        for item in self.datas_pred:
            if n_vert != item.n_vert:
                raise ValueError

    def calc_scale(self):
        self.calc_bbox_based_scale()
        self.calc_length_based_scale()

    def calc_bbox_based_scale(self):
        bbox_gt = np.zeros((3, 2), dtype=np.float)
        coord = self.data_gt.coord
        for i in range(3):
            bbox_gt[i, 0] = np.amax(coord[:, i])
            bbox_gt[i, 1] = np.amin(coord[:, i])
        dim_gt = np.linalg.norm(coord[:, 0] - coord[:, 1])

        n_pred = len(self.datas_pred)
        self.scale_bbox = [0.] * n_pred
        for idx in range(n_pred):
            coord = self.datas_pred[idx].coord
            bbox_pred = np.zeros((3, 2), dtype=np.float)
            for i in range(3):
                bbox_pred[i, 0] = np.amax(coord[:, i])
                bbox_pred[i, 1] = np.amin(coord[:, i])
            dim_pred = np.linalg.norm(coord[:, 0] - coord[:, 1])
            self.scale_bbox[idx] = dim_gt / dim_pred

    def calc_length_based_scale(self):
        n_vert = self.data_gt.n_vert
        n_pred = len(self.datas_pred)
        length_gt = 0.
        lengths_pred = [0.] * n_pred

        coord_gt = self.data_gt.coord
        print(f'  coord_gt: {coord_gt.shape}')
        for i in range(1, n_vert):
            length_gt += np.linalg.norm(coord_gt[i, :] - coord_gt[i-1, :])
            for j in range(n_pred):
                lengths_pred[j] += np.linalg.norm(self.datas_pred[j].coord[i, :] - self.datas_pred[j].coord[i-1, :])
        self.scale_length = [0.] * n_pred
        for j in range(n_pred):
            self.scale_length[j] = length_gt / lengths_pred[j]

    def show(self):
        datasets_traj = []
        datasets_traj.append(self.data_gt)
        datasets_traj.extend(self.datas_pred)
        colors = ['red', 'green', 'blue']

        assert len(datasets_traj) == 3
        bbox_scales = [1.]
        bbox_scales.extend(self.scale_bbox)
        length_scales = [1.]
        length_scales.extend(self.scale_length)
        for scale_type, scales in zip(['bbox_based', 'length_based'], [bbox_scales, length_scales]):
            viz_3d = VizTraj3D(f'VizTraj3D: ({scale_type}) {self.name}')
            for i in range(3):
                coord = datasets_traj[i].scaled_coord(scales[i])
                label = datasets_traj[i].name
                viz_3d.left_draw_lines(coord[:, 0], coord[:, 1], coord[:, 2], label, colors[i])
            viz_3d.show()
            viz_3d.close()

            viz_2d = VizTraj2D(f'VizTraj2D: ({scale_type}) {self.name}')
            for i in range(3):
                coord = datasets_traj[i].scaled_coord(scales[i])
                label = datasets_traj[i].name
                viz_2d.draw_lines(coord[:, 0], coord[:, 1], coord[:, 2], label, colors[i])
            viz_2d.show()
            viz_2d.close()


if __name__ == '__main__':
    np.set_printoptions(precision=6, suppress=True)
    root_dir = '/home/sigma/slam'
    datasets = [
        # 'matterport0614/test/matterport014_000_0516',
        # 'matterport0614/test/matterport014_001_0516',
        # 'matterport0614/train_val_test/matterport005_000_0516',
        # 'matterport0614/train_val_test/matterport005_001_0516',
        # 'matterport0614/train_val_test/matterport010_000_0516',
        'matterport0614/train_val_test/matterport010_001_0516',
        ]
    scannet_pred = 'indoor_scannet.ckpt_sample_rate-3_max_frames_450'
    matterport_pred = [
        # 'SupModelMF_DepthPoseNet_it12-h-out_epoch=52_matterport0516_ex-val_all_list-groundtruth-abs_rel_pp_gt=0.069.ckpt_sample_rate-3_max_frames_450',
        # 'SupModelMF_DepthPoseNet_it12-h-out_epoch=173_matterport0516-val_all_list-groundtruth-abs_rel_pp_gt=0.067.ckpt_sample_rate-3_max_frames_450',
        'SupModelMF_DepthPoseNet_it12-h-out_epoch=201_matterport0516_ex-val_all_list-groundtruth-abs_rel_pp_gt=0.064.ckpt_sample_rate-3_max_frames_450',
        ]

    name_gt = 'depths_vis_depth-GT_pose-GT_pose.obj'
    name_pred = 'depths_vis_depth-pred_pose-pred_pose.obj'
    for item_ds in datasets:
        obj_gt = osp.join(root_dir, item_ds, 'infer_video', scannet_pred, name_gt)
        obj_scannet_pred = osp.join(root_dir, item_ds, 'infer_video', scannet_pred, name_pred)
        for item_matterport in matterport_pred:
            obj_matterport_pred = osp.join(root_dir, item_ds, 'infer_video', item_matterport, name_pred)
            info = OrderedDict()
            info['Scannet'] = obj_scannet_pred
            info['Matterport'] = obj_matterport_pred
            viz = VizTrajectory(item_ds, obj_gt, info)
            viz.show()
