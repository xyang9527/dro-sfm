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
    if len(coord) <= 0:
        raise ValueError
    return np.array(coord)


def write_obj(filename, coord):
    with open(filename, 'w') as f_ou:
        n_vert = coord.shape[0]
        for i in range(n_vert):
            x, y, z = coord[i, :]
            f_ou.write(f'v {x} {y} {z}\n')
        for i in range(1, n_vert-1, 2):
            f_ou.write(f'f {i} {i+1} {i+2}\n')


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
        elev = 0  # -40
        azim = 0  # -80
        self.ax_left = self.main_fig.add_subplot(1, 2, 1, projection='3d',
                                            elev=elev, azim=azim)
        self.ax_right = self.main_fig.add_subplot(1, 2, 2, projection='3d',
                                            elev=elev, azim=azim)
        self.bbox_min_left = [0.] * 3
        self.bbox_max_left = [0.] * 3
        self.dim_left = [0.] * 3
        self.dim_xyz_left = 0.

        self.bbox_min_right = [0.] * 3
        self.bbox_max_right = [0.] * 3
        self.dim_right = [0.] * 3
        self.dim_xyz_right = 0.

    def close(self):
        plt.close()

    def update_bbox(self, is_left, x, y, z):
        if is_left:
            bbox_min = self.bbox_min_left
            bbox_max = self.bbox_max_left
            dim = self.dim_left
            dim_xyz = self.dim_xyz_left
        else:
            bbox_min = self.bbox_min_right
            bbox_max = self.bbox_max_right
            dim = self.dim_right
            dim_xyz = self.dim_xyz_right

        xyz = [x, y, z]

        for i in range(3):
            v_min, v_max = min(xyz[i]), max(xyz[i])
            if bbox_min[i] > v_min:
                bbox_min[i] = v_min
            if bbox_max[i] < v_max:
                bbox_max[i] = v_max

            if dim[i] < np.fabs(v_min):
                dim[i] = np.fabs(v_min)
            if dim[i] < np.fabs(v_max):
                dim[i] = np.fabs(v_max)
            if dim_xyz < dim[i]:
                dim_xyz = dim[i]

        if is_left:
            self.bbox_min_left = bbox_min
            self.bbox_max_left = bbox_max
            self.dim_left = dim
            self.dim_xyz_left = dim_xyz
        else:
            self.bbox_min_right = bbox_min
            self.bbox_max_right = bbox_max
            self.dim_right = dim
            self.dim_xyz_right = dim_xyz


    def draw_lines(self, is_left, x_arr, y_arr, z_arr, line_label, line_color, sub_win_name):
        self.update_bbox(is_left, x_arr, y_arr, z_arr)
        if is_left:
            win = self.ax_left
        else:
            win = self.ax_right
        # win.set_zticks(np.arange(-3.0, 3.0, step=1.0))
        # win.set_zbound(-3.0, 3.0)

        win.plot(x_arr, y_arr, z_arr, color=line_color, label=line_label)

        # win.set_zlim(-3.0, 3.0)
        win.set_title(sub_win_name, fontsize=30, color='cyan')


    def decorate(self):
        for dim, win in zip([self.dim_xyz_left, self.dim_xyz_right],
                            [self.ax_left, self.ax_right]):
            win.set_xlabel('$X$', fontsize=20, color='red')
            win.set_ylabel('$Y$', fontsize=20, color='green')
            win.set_zlabel('$Z$', fontsize=20, color='blue')
            win.set_xlim(-dim, dim)
            win.set_ylim(-dim, dim)
            win.set_zlim(0, dim)
            win.legend(loc='upper left')

    def show(self, save_name):
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

        self.decorate()

        thismanager = plt.get_current_fig_manager()
        move_figure(thismanager, 1920, 0)
        print(f'saving: {save_name}')
        plt.savefig(save_name)
        plt.show()


class VizTraj2D:
    def __init__(self, win_name='VizTraj2D'):
        self.bbox_min = [0.] * 3
        self.bbox_max = [0.] * 3
        self.dim = [0.] * 3
        self.dim_xyz = 0.

        fig, axs = plt.subplots(2, 3, figsize=(20, 12), dpi=80, constrained_layout=False)
        self.main_fig = fig
        self.subfig_xoy_top = axs[0, 0]
        self.subfig_yoz_top = axs[0, 1]
        self.subfig_xoz_top = axs[0, 2]
        self.subfig_xoy_bottom = axs[1, 0]
        self.subfig_yoz_bottom = axs[1, 1]
        self.subfig_xoz_bottom = axs[1, 2]

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

    def draw_lines(self, is_top, x_arr, y_arr, z_arr, label, color, sub_win_name):
        self.update_bbox(x_arr, y_arr, z_arr)
        if is_top:
            xoy = self.subfig_xoy_top
            yoz = self.subfig_yoz_top
            xoz = self.subfig_xoz_top
        else:
            xoy = self.subfig_xoy_bottom
            yoz = self.subfig_yoz_bottom
            xoz = self.subfig_xoz_bottom

        # xoy
        xoy.plot(x_arr, y_arr, label=label, color=color)
        xoy.set_xlabel('X', fontsize=18, color='red')
        xoy.set_ylabel('Y', fontsize=18, color='green')
        xoy.set_title(f'XOY Projeciton ({sub_win_name})', fontsize=25, color='blue')

        # yoz
        yoz.plot(y_arr, z_arr, label=label, color=color)
        yoz.set_xlabel('Y', fontsize=18, color='green')
        yoz.set_ylabel('Z', fontsize=18, color='blue')
        yoz.set_title(f'YOZ Projeciton ({sub_win_name})', fontsize=25, color='red')

        # xoz
        xoz.plot(x_arr, z_arr, label=label, color=color)
        xoz.set_xlabel('X', fontsize=18, color='red')
        xoz.set_ylabel('Z', fontsize=18, color='blue')
        xoz.set_title(f'XOZ Projeciton ({sub_win_name})', fontsize=25, color='green')

    def decorate(self):
        for win in [self.subfig_xoy_top, self.subfig_yoz_top, self.subfig_xoz_top]:
            win.set_xlim(-self.dim_xyz, self.dim_xyz)
            win.set_ylim(-self.dim_xyz, self.dim_xyz)
            win.legend(loc='upper left')

    def show(self, save_name):
        self.decorate()
        plt.savefig(save_name, dpi=100)
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

        # save scaled obj
        obj_name = datasets_traj[-1].path

        obj_name_tags = osp.splitext(obj_name)
        bbox_scaled_name = f'{obj_name_tags[0]}_bbox_based_scale{obj_name_tags[1]}'
        length_scaled_name = f'{obj_name_tags[0]}_length_based_scale{obj_name_tags[1]}'
        write_obj(bbox_scaled_name, datasets_traj[-1].scaled_coord(bbox_scales[-1]))
        write_obj(length_scaled_name, datasets_traj[-1].scaled_coord(length_scales[-1]))

        viz_3d = VizTraj3D(f'VizTraj3D: {self.name}')
        for i in range(3):
            label = datasets_traj[i].name

            coord_bbox = datasets_traj[i].scaled_coord(bbox_scales[i])
            viz_3d.draw_lines(True, coord_bbox[:, 0], coord_bbox[:, 1], coord_bbox[:, 2], label, colors[i], 'bbox based scale')

            coord_length = datasets_traj[i].scaled_coord(length_scales[i])
            viz_3d.draw_lines(False, coord_length[:, 0], coord_length[:, 1], coord_length[:, 2], label, colors[i], 'length based scale')

        # save figure
        save_name = obj_name.replace('.obj', '_3D.png')
        viz_3d.show(save_name)
        viz_3d.close()

        viz_2d = VizTraj2D(f'VizTraj2D: {self.name}')
        for i in range(3):
            label = datasets_traj[i].name
            coord_bbox = datasets_traj[i].scaled_coord(bbox_scales[i])
            viz_2d.draw_lines(True, coord_bbox[:, 0], coord_bbox[:, 1], coord_bbox[:, 2], label, colors[i], 'bbox')

            coord_length = datasets_traj[i].scaled_coord(length_scales[i])
            viz_2d.draw_lines(False, coord_length[:, 0], coord_length[:, 1], coord_length[:, 2], label, colors[i], 'length')

        save_name = obj_name.replace('.obj', '_2D.png')
        viz_2d.show(save_name)
        viz_2d.close()


if __name__ == '__main__':
    np.set_printoptions(precision=6, suppress=True)
    root_dir = '/home/sigma/slam'
    datasets = [
        # 'matterport0614/test/matterport014_000_0516',
        # 'matterport0614/test/matterport014_001_0516',
        # 'matterport0614/train_val_test/matterport005_000_0516',  # only good for scannet
        # 'matterport0614/train_val_test/matterport005_001_0516',
        'matterport0614/train_val_test/matterport010_000_0516',  # good for scannet and matterport (epoch=201)
        # 'matterport0614/train_val_test/matterport010_001_0516',  # bad for matterport (epoch=201 -> sudden jump)
        ]
    scannet_pred = 'indoor_scannet.ckpt_sample_rate-3_max_frames_450'
    matterport_pred = [
        # 'SupModelMF_DepthPoseNet_it12-h-out_epoch=52_matterport0516_ex-val_all_list-groundtruth-abs_rel_pp_gt=0.069.ckpt_sample_rate-3_max_frames_450',  # bad
        # 'SupModelMF_DepthPoseNet_it12-h-out_epoch=173_matterport0516-val_all_list-groundtruth-abs_rel_pp_gt=0.067.ckpt_sample_rate-3_max_frames_450',  # bad
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
            viz = VizTrajectory(item_ds + ' : ' + item_matterport, obj_gt, info)
            viz.show()
