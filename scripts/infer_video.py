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


class VideoInfo:
    def __init__(self):
        self.hostname = subprocess.check_output(['hostname']).decode('UTF-8')[:-1]
        self.pwd = subprocess.check_output(['pwd']).decode('UTF-8')[:-1]

        # https://docs.python.org/3/library/datetime.html#datetime-objects
        dt_now = datetime.datetime.now()
        self.datetime = f'{dt_now.year:04d}-{dt_now.month:02d}-{dt_now.day:02d}'
        minute_ex = (dt_now.minute // 10) * 10
        self.datetime_ex = f'{dt_now.year:04d}-{dt_now.month:02d}-{dt_now.day:02d} {dt_now.hour:02d}:{minute_ex:02d}'

        _, hexsha, is_dirty = git_info()
        self.git_hexsha = hexsha
        self.git_is_dirty = is_dirty

        self.path_model = ''
        self.path_data = ''

        self.header_height = 180
        self.footer_height = 150

        self.sample_rate = 1
        self.max_frames = 5

    def print_info(self):
        print(f'  hostname:      {self.hostname}')
        print(f'  pwd:           {self.pwd}')

        print(f'  datetime:      {self.datetime}')
        print(f'  datetime_ex:   {self.datetime_ex}')

        print(f'  git_hexsha:    {self.git_hexsha}')
        print(f'  git_is_dirty:  {self.git_is_dirty}')


g_video_info = VideoInfo()


def parse_args():
    parser = argparse.ArgumentParser(description='dro-sfm inference of depth maps from images')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint (.ckpt)', required=True)
    parser.add_argument('--input', type=str, help='Input folder or video', required=True)
    parser.add_argument('--output', type=str, help='Output folder', required=True)
    parser.add_argument('--data_type', type=str, choices=['kitti', 'indoor', 'scannet', 'matterport', 'general'], required=True)
    parser.add_argument('--sample_rate', type=int, default=10, help='sample rate', required=True)
    parser.add_argument('--max_frames', type=int, default=120, help='max frames to test')
    parser.add_argument('--ply_mode', action="store_true", help='vis point cloud')
    parser.add_argument('--use_depth_gt', action="store_true", help='use GT depth for vis')
    parser.add_argument('--use_pose_gt', action="store_true", help='use GT pose for vis')
    parser.add_argument('--mix_video_mode', action="store_true", help="create video with 4 modes")
    parser.add_argument('--archive_video', type=str, help='folder to archive videos', default='/tmp/dro-sfm-demo')

    parser.add_argument('--image_shape', type=int, nargs='+', default=None,
                        help='Input and output image shape '
                             '(default: checkpoint\'s config.datasets.augmentation.image_shape)')

    args = parser.parse_args()
    assert args.checkpoint.endswith('.ckpt'), \
        'You need to provide a .ckpt file as checkpoint'
    assert args.image_shape is None or len(args.image_shape) == 2, \
        'You need to provide a 2-dimensional tuple as shape (H,W)'
    return args


def get_intrinsics(image_shape_raw, image_shape, data_type):
    # logging.warning(f'get_intrinsics({image_shape_raw}, {image_shape}, {data_type})')
    if data_type == "kitti":
        intr = np.array([7.215376999999999725e+02, 0.000000000000000000e+00, 6.095593000000000075e+02,
                         0.000000000000000000e+00, 7.215376999999999725e+02, 1.728540000000000134e+02,
                         0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00], dtype=np.float32).reshape(3, 3)
    elif data_type == "scannet":
        intr = np.array([1169.621094, 0.000000, 646.295044, 
                         0.000000, 1167.105103, 489.927032,
                         0.000000, 0.000000, 1.000000], dtype=np.float32).reshape(3, 3)
    elif data_type == "indoor":
        intr = np.array([1170.187988, 0.000000, 647.750000, 
                         0.000000, 1170.187988, 483.750000,
                         0.000000, 0.000000, 1.000000], dtype=np.float32).reshape(3, 3)
    elif data_type == "matterport":
        intr = np.array([530.4669406576809,   0.0,             320.5,
                         0.0,               530.4669406576809, 240.5,
                         0.0,                 0.0,               1.0], dtype=np.float32).reshape(3, 3)
    else:
        # print("fake intrinsics")
        w, h = image_shape_raw
        fx = w * 1.2
        fy = w * 1.2
        cx = w / 2.0
        cy = h / 2.0
        intr = np.array([[fx, 0., cx],
                         [0., fy, cy],
                         [0., 0., 1.]])

    orig_w, orig_h = image_shape_raw
    out_h, out_w = image_shape

    # Scale intrinsics
    intr[0] *= out_w / orig_w
    intr[1] *= out_h / orig_h

    return intr


@torch.no_grad()
def infer_and_save_pose(input_file_refs, input_file, model_wrapper, image_shape, data_type,
                        save_depth_root, save_vis_root, save_model=False):
    """
    Process a single input file to produce and save visualization

    Parameters
    ----------
    input_file_refs : list(str)
        Reference image file paths
    input_file : str
        Image file for pose estimation
    model_wrapper : nn.Module
        Model wrapper used for inference
    image_shape : Image shape
        Input image shape
    half: bool
        use half precision (fp16)
    save: str
        Save format (npz or png)
    """
    logging.warning(f'infer_and_save_pose(..)')
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    image_raw_wh = load_image(input_file).size

    # Load image
    def process_image(filename):
        image = load_image(filename)
        logging.info(f'  process_image({filename})')

        # Resize and to tensor
        intr = get_intrinsics(image.size, image_shape, data_type) #(3, 3)
        image = resize_image(image, image_shape)
        image = to_tensor(image).unsqueeze(0)
        intr = torch.from_numpy(intr).unsqueeze(0) #(1, 3, 3)

        # Send image to GPU if available
        if torch.cuda.is_available():
            image = image.to('cuda')
            intr = intr.to('cuda')
        return image, intr

    image_ref = [process_image(input_file_ref)[0] for input_file_ref in input_file_refs]
    image, intrinsics = process_image(input_file)

    batch = {'rgb': image, 'rgb_context': image_ref, "intrinsics": intrinsics}

    output = model_wrapper(batch)
    if save_model:
        # torch.onnx.export(model_wrapper, batch, "/home/sigma/slam/matterport/test/matterport014_000/infer_video/model.onnx")
        pass

    inv_depth = output['inv_depths'][0] #(1, 1, h, w)
    depth = inv2depth(inv_depth)[0, 0].detach().cpu().numpy() #(h, w)

    pose21 = output['poses'][0].mat[0].detach().cpu().numpy() #(4, 4)  #TODO check: targe -> ref[0]
    pose23 = output['poses'][1].mat[0].detach().cpu().numpy() #(4, 4)  #TODO check: targe -> ref[0]

    vis_depth = viz_inv_depth(inv_depth[0]) * 255

    vis_depth_upsample = cv2.resize(vis_depth, image_raw_wh, interpolation=cv2.INTER_LINEAR)
    write_image(os.path.join(save_vis_root, f"{base_name}.jpg"), vis_depth_upsample)

    depth_upsample = cv2.resize(depth, image_raw_wh, interpolation=cv2.INTER_NEAREST)
    np.save(os.path.join(save_depth_root, f"{base_name}.npy"), depth_upsample)

    if data_type == 'matterport' or data_type == 'scannet':
        # ground truth depth
        if data_type == 'matterport':
            depth_gt_int = np.array(Image.open(input_file.replace('cam_left', 'depth').replace('jpg', 'png')), dtype=int)
        else:
            depth_gt_int = np.array(Image.open(input_file.replace('color', 'depth').replace('jpg', 'png')), dtype=int)

        mask_invalid = depth_gt_int == 0
        depth_gt_float = depth_gt_int.astype(np.float) / 1000.0
        vis_depth_gt = viz_inv_depth(depth_gt_float) * 255

        vis_depth_gt[mask_invalid, :] = 0

        save_vis_root_gt = save_vis_root.replace('depth_vis', 'depth_gt_vis')
        if not os.path.exists(save_vis_root_gt):
            os.makedirs(save_vis_root_gt)

        vis_depth_gt_upsample = cv2.resize(vis_depth_gt, image_raw_wh, interpolation=cv2.INTER_LINEAR)
        write_image(os.path.join(save_vis_root_gt, f"{base_name}.jpg"), vis_depth_gt_upsample)

    return depth, pose21, pose23, intrinsics[0].detach().cpu().numpy(), image[0].permute(1, 2, 0).detach().cpu().numpy() * 255


def start_visualization(queue_g, win_size, cinematic=False, render_path=None, clear_points=False, is_kitti=True):
    """ Start interactive slam visualization in seperate process """
    logging.warning(f'start_visualization(..)')

    # visualization is a Process Object
    viz = vis.InteractiveViz(queue_g, cinematic, render_path, clear_points, win_size, is_kitti=is_kitti)
    viz.start()

    return viz


def get_coordinate_xy(coord_shape, device):
    """get meshgride coordinate of x, y and the shape is (B, H, W)"""
    logging.warning(f'get_coordinate_xy({coord_shape}, {device})')
    bs, height, width = coord_shape
    # https://github.com/pytorch/pytorch/issues/50276#issuecomment-945860735
    y_coord, x_coord = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=device),\
                                       torch.arange(0, width, dtype=torch.float32, device=device)], indexing='ij')
    y_coord, x_coord = y_coord.contiguous(), x_coord.contiguous()
    y_coord, x_coord = y_coord.unsqueeze(0).repeat(bs, 1, 1), \
                       x_coord.unsqueeze(0).repeat(bs, 1, 1)

    return x_coord, y_coord


def reproject_with_depth_batch(depth_ref, depth_src, ref_pose, src_pose, xy_coords):
    """project the reference point cloud into the source view, then project back"""
    logging.warning(f'reproject_with_depth_batch({depth_ref.shape}, {depth_src.shape}, {len(ref_pose)}, {len(src_pose)}, {len(xy_coords)})')
    intrinsics_ref, extrinsics_ref = ref_pose["intr"], ref_pose["extr"]
    intrinsics_src, extrinsics_src = src_pose["intr"], src_pose["extr"]

    bs, height, width = depth_ref.shape[:3]

    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = xy_coords  # (B, H, W)
    x_ref, y_ref = x_ref.view([bs, 1, -1]), y_ref.view([bs, 1, -1])  # (B, 1, H*W)

    # reference 3D space
    xyz_ref = torch.matmul(torch.inverse(intrinsics_ref), \
                           torch.cat([x_ref, y_ref, torch.ones_like(x_ref)], dim=1) * \
                           depth_ref.view([bs, 1, -1]))  # (B, 3, H*W)
    # source 3D space
    xyz_src = torch.matmul(torch.matmul(extrinsics_src, torch.inverse(extrinsics_ref)), \
                           torch.cat([xyz_ref, torch.ones_like(x_ref)], dim=1))[:, :3]

    # source view x, y
    k_xyz_src = torch.matmul(intrinsics_src, xyz_src)
    xy_src = k_xyz_src[:, :2] / (k_xyz_src[:, 2:3].clamp(min=1e-10))  # (B, 2, H*W)

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[:, 0].view([bs, height, width]).float()
    y_src = xy_src[:, 1].view([bs, height, width]).float()

    x_src_norm = x_src / ((width - 1) / 2) - 1
    y_src_norm = y_src / ((height - 1) / 2) - 1
    xy_src_norm = torch.stack([x_src_norm, y_src_norm], dim=3)
    sampled_depth_src = torch.nn.functional.grid_sample(depth_src.unsqueeze(1), xy_src_norm, \
                                                        mode="nearest", padding_mode="zeros", align_corners=True)
    sampled_depth_src = sampled_depth_src.squeeze(1)

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = torch.matmul(torch.inverse(intrinsics_src), \
                           torch.cat([xy_src, torch.ones_like(x_ref)], dim=1) * \
                           sampled_depth_src.view([bs, 1, -1]))

    # reference 3D space:#(B, 3, H, W)
    xyz_reprojected = torch.matmul(torch.matmul(extrinsics_ref, torch.inverse(extrinsics_src)), \
                                   torch.cat([xyz_src, torch.ones_like(x_ref)], dim=1))[:, :3]

    # source view x, y, depth
    depth_reprojected = xyz_reprojected[:, 2].view([bs, height, width]).float()
    depth_reprojected = depth_reprojected * (sampled_depth_src > 0)
    k_xyz_reprojected = torch.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = k_xyz_reprojected[:, :2] / (k_xyz_reprojected[:, 2:3].clamp(min=1e-10))
    x_reprojected = xy_reprojected[:, 0].view([bs, height, width]).float()
    y_reprojected = xy_reprojected[:, 1].view([bs, height, width]).float()

    return depth_reprojected, x_reprojected, y_reprojected


def check_geometric_consistency_batch(depth_ref, depth_src, ref_pose, src_pose, xy_coords,\
                                      thres_p_dist=1, thres_d_diff=0.01):
    """check geometric consistency
    consider two factor:
    1.disparity < 1
    2.relative depth differ ratio < 0.001
    """
    logging.warning(f'check_geometric_consistency_batch(..)')
    x_ref, y_ref = xy_coords  # (B, H, W)
    depth_reprojected, x2d_reprojected, y2d_reprojected = \
        reproject_with_depth_batch(depth_ref, depth_src, ref_pose, src_pose, xy_coords)

    # check |p_reproj-p_1| < p_dist
    dist = torch.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < d_diff
    depth_diff = torch.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / (depth_ref.clamp(min=1e-10))

    mask = (dist < thres_p_dist) & (relative_depth_diff < thres_d_diff)
    # mask = (dist < thres_p_dist) & (depth_diff < 0.1)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected


def gemo_filter_fusion(depth_ref, depth_srcs, ref_pose, src_poses, intr, thres_view):
    logging.warning(f'gemo_filter_fusion(..)')
    depth_ref = torch.from_numpy(depth_ref).unsqueeze(0).to("cuda") #(1, H, W)
    ref_pose = torch.from_numpy(ref_pose).unsqueeze(0).to("cuda") #(1, 4, 4)
    intr = torch.from_numpy(intr).unsqueeze(0).to("cuda").float() #(1, 3, 3)

    depth_srcs = [torch.from_numpy(depth_src).unsqueeze(0).to("cuda") for depth_src in depth_srcs]
    src_poses = [torch.from_numpy(src_pose).unsqueeze(0).to("cuda") for src_pose in src_poses]

    xy_coords = get_coordinate_xy(depth_ref.shape, device=depth_ref.device)

    params = {"thres_p_dist": 1, "thres_d_diff": 0.001}

    geo_mask_sum = torch.zeros_like(depth_ref)
    all_srcview_depth_ests = torch.zeros_like(depth_ref)

    for depth_src, src_pose in zip(depth_srcs, src_poses):
        geo_mask, depth_reprojected = check_geometric_consistency_batch( \
            depth_ref, depth_src, \
            ref_pose={"intr": intr, "extr": ref_pose}, \
            src_pose={"intr": intr, "extr": src_pose}, \
            xy_coords=xy_coords, thres_p_dist=params["thres_p_dist"],\
            thres_d_diff=params["thres_d_diff"])
        geo_mask_sum += geo_mask.float()
        all_srcview_depth_ests += depth_reprojected

     # fusion
    geo_mask = (geo_mask_sum - thres_view) >= 0
    depth_ests_averaged = (all_srcview_depth_ests + depth_ref) / (geo_mask_sum + 1)
    depth_ests_averaged = depth_ests_averaged * geo_mask

    return depth_ests_averaged[0].detach().cpu().numpy()


def parse_video(video_file, save_root, sample_rate=10):
    logging.warning(f'parse_video({video_file}, {save_root}, {sample_rate})')

    os.makedirs(save_root, exist_ok=True)
    cap = cv2.VideoCapture(video_file)

    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
    count = 0
    sample_count = 0

    while cap.isOpened():
        ret, img = cap.read()
        if ret:
            if count % sample_rate == 0:
                save_path = os.path.join(save_root, f"{sample_count}".zfill(6) + ".jpg")
                cv2.imwrite(save_path, img)
                sample_count += 1
            count += 1
        else:
            break
    print0(pcolor(f'  video total frames num: {count},  sampled frames num:{sample_count}', 'yellow'))


def init_model(args):
    logging.warning(f'init_model(..)')

    print0(pcolor(f'init model start .....................', 'green'))
    hvd_disable()

    # Parse arguments
    config, state_dict = parse_test_file(args.checkpoint)

    # If no image shape is provided, use the checkpoint one
    image_shape = args.image_shape
    if image_shape is None:
        image_shape = config.datasets.augmentation.image_shape
    print0(pcolor(f'  input image shape:{image_shape}', 'yellow'))

    # Set debug if requested
    # set_debug(config.debug)
    set_debug(True)

    # Initialize model wrapper from checkpoint arguments
    model_wrapper = ModelWrapper(config, load_datasets=False)
    # Restore monodepth_model state
    model_wrapper.load_state_dict(state_dict)

    # Send model to GPU if available
    if torch.cuda.is_available():
        model_wrapper = model_wrapper.to('cuda')
    else:
        raise RuntimeError("cuda is not available")

    # Set to eval mode
    model_wrapper.eval()

    print0(pcolor(f'init model finish .....................', 'blue'))
    return model_wrapper, image_shape


def get_gt_pose(file, data_type):
    pose_file = None

    if data_type == 'scannet':
        pose_file = file.replace('color', 'pose').replace('jpg', 'txt')
    elif data_type == 'matterport':
        pose_file = file.replace('cam_left', 'pose').replace('jpg', 'txt')
    else:
        pose_file = file.replace('color', 'pose').replace('jpg', 'txt')

    if osp.exists(pose_file):
        return True, np.genfromtxt(pose_file)
    return False, None


def get_gt_depth(file, data_type):
    depth_file = None

    if data_type == 'scannet':
        depth_file = file.replace('color', 'depth').replace('jpg', 'png')
    elif data_type == 'matterport':
        depth_file = file.replace('cam_left', 'depth').replace('jpg', 'png')
    else:
        depth_file = file.replace('color', 'depth').replace('jpg', 'png')

    if not osp.exists(depth_file):
        return False, None

    depth_gt_int = np.array(Image.open(depth_file), dtype=int)
    depth_gt_float = depth_gt_int.astype(np.float) / 1000.0

    return True, depth_gt_float


def inference(model_wrapper, image_shape, input, sample_rate, max_frames,
              output_depths_npy, output_vis_video, output_tmp_dir,
              data_type="general", ply_mode=False, use_depth_gt=False, use_pose_gt=False, sfm_params=None):
    logging.warning(f'inference(\n'
                    f'  model_wrapper={type(model_wrapper)},\n'
                    f'  image_shape={image_shape},\n'
                    f'  input={input},\n'
                    f'  sample_rate={sample_rate},\n'
                    f'  output_depths_npy={output_depths_npy},\n'
                    f'  output_vis_video={output_vis_video},\n'
                    f'  output_tmp_dir={output_tmp_dir},\n'
                    f'  data_type={data_type},\n'
                    f'  ply_mode={ply_mode},\n'
                    f'  use_depth_gt={use_depth_gt},\n'
                    f'  use_pose_gt={use_pose_gt},\n'
                    f'  sfm_params={sfm_params})')

    assert os.path.exists(input)
    assert os.path.exists(output_tmp_dir)

    save_depth_root = os.path.join(output_tmp_dir, "depth")
    save_vis_root = os.path.join(output_tmp_dir, "depth_vis")
    save_color_root = osp.join(output_tmp_dir, 'color')

    os.makedirs(save_depth_root, exist_ok=True)
    os.makedirs(save_vis_root, exist_ok=True)
    os.makedirs(save_color_root, exist_ok=True)

    input_type = "folder"

    # processs input data
    if not os.path.isdir(input):
        print0(pcolor(f'processing video input: .........', 'blue'))

        input_type = "video"
        assert os.path.splitext(input)[1] in [".mp4", ".avi", ".mov", ".mpeg", ".flv", ".wmv"]
        input_video_images = os.path.join(output_tmp_dir, "input_video_images")
        parse_video(input, input_video_images, sample_rate)

        # update input
        input = input_video_images

    files = []
    for ext in ['png', 'jpg', 'bmp']:
        files.extend(glob((os.path.join(input, '*.{}'.format(ext)))))

    if input_type == "folder":
        print0(pcolor(f'processing folder input: ...........', 'blue'))
        print0(pcolor(f'  folder total frames num: {len(files)}', 'yellow'))
        files = files[::sample_rate]

    files.sort()
    skip_first_n = 30
    if len(files) > max_frames + skip_first_n:
        files = files[skip_first_n:max_frames+skip_first_n]
    print0(pcolor(f'  Found total {len(files)} files', 'yellow'))
    assert len(files) > 2

    # Process each file
    list_of_files = list(zip(files[:-2],
                              files[1:-1],
                              files[2:]))

    render_desc = ''
    if use_depth_gt:
        render_desc = f'depth-GT'
    else:
        render_desc = f'depth-pred'

    if use_pose_gt:
        render_desc = f'{render_desc}_pose-GT'
    else:
        render_desc = f'{render_desc}_pose-pred'
    render_tag = 'renders'

    traj_modes = ['depth-GT_pose-GT', 'depth-GT_pose-pred', 'depth-pred_pose-GT', 'depth-pred_pose-pred']

    print0(pcolor(f'  use_depth_gt: {use_depth_gt}', 'yellow'))
    print0(pcolor(f'  use_pose_gt:  {use_pose_gt}', 'yellow'))

    if ply_mode:
        logging.info(f'  ply_mode')

        # visulation
        # new points and poses get added to the queue
        queue_g = Queue()
        vis_counter = 0

        render_path=os.path.join(output_tmp_dir, f'{render_tag}_{render_desc}')
        os.makedirs(render_path, exist_ok=True)

        img_sample = cv2.imread(files[0])
        start_visualization(
            queue_g, cinematic=True, render_path=render_path,
            clear_points=False, win_size=img_sample.shape[:2], is_kitti= data_type=="kitti")

    pose_prev = None
    pose_23_prev = None
    depth_list = []
    pose_list = []

    print0(pcolor(f'data_type: {data_type}', 'yellow'))
    print0(pcolor(f'inference start .....................', 'green'))

    for idx_frame, fns in enumerate(list_of_files):
        fn1, fn2, fn3 = fns
        logging.info(f'  frame {idx_frame:4d}\n    fn1={fn1},\n    fn2={fn2},\n    fn3={fn3}')

        has_1, gt_pose_1 = get_gt_pose(fn1, data_type)
        has_2, gt_pose_2 = get_gt_pose(fn2, data_type)
        has_3, gt_pose_3 = get_gt_pose(fn3, data_type)
        if not has_1 or not has_2 or not has_3:
            logging.warning(f'skip frame {idx_frame:4d}')
            continue

        has_depth, depth_gt = get_gt_depth(fn2, data_type)
        depth, pose21, pose23, intr, rgb = infer_and_save_pose([fn1, fn3], fn2, model_wrapper, 
                                                                image_shape, data_type,
                                                                save_depth_root, save_vis_root, idx_frame==0)
        depth_list.append(depth)
        logging.info(f'  idx_frame: {idx_frame:6d}')
        logging.info(f'    depth:  {type(depth)}  {depth.shape}  {depth.dtype}')
        logging.info(f'    pose21: {type(pose21)}  {pose21.shape}  {pose21.dtype}')
        logging.info(f'    pose23: {type(pose23)}  {pose23.shape}  {pose23.dtype}')
        logging.info(f'    intr:   {type(intr)}  {intr.shape}  {intr.dtype}')
        logging.info(f'    rgb:    {type(rgb)}  {rgb.shape}  {rgb.dtype}')

        if use_pose_gt:
            # for idx_t in range(3):
            #    gt_pose_1[idx_t, 3] = -gt_pose_1[idx_t, 3]
            #    gt_pose_2[idx_t, 3] = -gt_pose_2[idx_t, 3]
            #    gt_pose_3[idx_t, 3] = -gt_pose_3[idx_t, 3]

            pose21 = np.matmul(np.linalg.inv(gt_pose_1), gt_pose_2).astype(np.float32)
            pose23 = np.matmul(np.linalg.inv(gt_pose_3), gt_pose_2).astype(np.float32)

            # pose21 = np.linalg.inv(pose21)
            # pose23 = np.linalg.inv(pose23)
            # for idx_t in range(3):
            #    pose21[idx_t, 3] = pose21[idx_t, 3]
            #    pose23[idx_t, 3] = pose23[idx_t, 3]

        else:
            is_matterport_model=True
            if is_matterport_model:
                case_inv = False
                if case_inv:
                    pose21 = np.linalg.inv(pose21)
                    pose23 = np.linalg.inv(pose23)
                else:
                    for idx_t in range(3):
                        pose21[idx_t, 3] = -pose21[idx_t, 3]
                        pose23[idx_t, 3] = -pose23[idx_t, 3]
                        pass

        if use_depth_gt:
            # https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/
            depth_gt_resize = cv2.resize(depth_gt, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
            depth = depth_gt_resize

        '''
        pose21[0][3] = -pose21[0][3]
        pose21[1][3] = -pose21[1][3]
        pose21[2][3] = -pose21[2][3]
        pose23[0][3] = -pose23[0][3]
        pose23[1][3] = -pose23[1][3]
        pose23[2][3] = -pose23[2][3]
        '''

        logging.info(f'frame {idx_frame:6d}\npose21:\n{pose21}\npose23:\n{pose23}\n')

        if ply_mode:
            if pose_23_prev is not None:
                s = np.linalg.norm(np.linalg.norm(pose_23_prev[:3, 3])) / np.linalg.norm(pose21[:3, 3])
                pose21[:3, 3] = pose21[:3, 3] * s

            pose_23_prev = pose23
            pose = pose21

            depth_pad = np.pad(depth, [(0, 1), (0, 1)], "constant")

            depth_grad = (depth_pad[1:, :-1] - depth_pad[:-1, :-1])**2 + (depth_pad[:-1, 1:] - depth_pad[:-1, :-1])**2
            depth[depth_grad > sfm_params["filer_depth_grad_max"]] = 0
            depth[depth > sfm_params["filer_depth_max"]] = 0

            crop_h = sfm_params["depth_crop_h"]
            crop_w = sfm_params["depth_crop_w"]

            depth[:crop_h, :crop_w] = 0
            depth[-crop_h:, -crop_w:] = 0

            logging.info(f"  depth median:{np.median(depth)}")

            if pose_prev is not None:
                pose = np.matmul(pose_prev, pose)
            pose_prev = pose

            pose_list.append(pose)
            num_view = sfm_params["fusion_view_num"]

            if len(pose_list) >= num_view and not use_depth_gt and False:
                depth = gemo_filter_fusion(depth_list[-1], depth_list[-num_view:-1], pose_list[-1],
                                        pose_list[-num_view:-1], intr, thres_view=sfm_params["fusion_thres_view"])

            pcd_colors = rgb.reshape(-1, 3) #TODO rgb or bgr

            h, w = depth.shape[:2]
            x_m, y_m = np.meshgrid(np.arange(0, w), np.arange(0, h))
            xy_m = np.stack([x_m, y_m, np.ones_like(x_m)], axis=2).reshape(-1, 3) #(h*w, 3)
            depth = depth.reshape(-1)[np.newaxis, :] #(1, N)
            p3d = np.multiply(depth, np.matmul(np.linalg.inv(intr), xy_m.transpose(1, 0))) #(3, N)

            p3d_trans = np.matmul(pose[:3], np.concatenate([p3d, np.ones((1, p3d.shape[1]))], axis=0)) #(3, N)

            pcd_coords = p3d_trans.transpose(1, 0)
            pointcloud = (pcd_coords, pcd_colors)
            logging.info(f'  pcd_coords: {pcd_coords.shape}')
            logging.info(f'  pcd_colors: {pcd_colors.shape}')

            vis_counter += 1
            pose = np.linalg.inv(pose)
            queue_g.put((pointcloud, pose))

    # save all depths and vis depths
    depth_npy_list = []
    for file in sorted(glob(os.path.join(save_depth_root, "*.npy"))):
        depth_npy_list.append(np.load(file))

    print(f'writing {output_depths_npy}')
    logging.info(f'writing {output_depths_npy}')
    np.save(output_depths_npy, np.stack(depth_npy_list, axis=0))

    # ======================================================================== #
    # Write Video
    # ======================================================================== #
    files = sorted(glob(os.path.join(save_vis_root, "*.jpg")))
    image_hw = cv2.imread(files[0]).shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 4.0
    gap_size = 40
    tags = osp.splitext(output_vis_video)
    video_name = f'{tags[0]}_{render_desc}{tags[1]}'

    # pose list to obj
    obj_name = video_name.replace('.avi', '_pose.obj')
    with open(obj_name, 'w') as f_ou:
        for item in pose_list:
            f_ou.write(f'v {item[0, 3]} {item[1, 3]} {item[2, 3]}\n')

        n_pose = len(pose_list)
        for idx_p in range(1, n_pose-1, 2):
            f_ou.write(f'f {idx_p} {idx_p+1} {idx_p+2}\n')

    h_header = g_video_info.header_height
    h_footer = g_video_info.footer_height

    video_writer = cv2.VideoWriter(video_name, fourcc, fps, (image_hw[1]*4+gap_size*3, image_hw[0]*2+gap_size*2+h_header+h_footer))
    canvas = np.full((image_hw[0]*2+gap_size*2+h_header+h_footer, image_hw[1]*4+gap_size*3, 3), 64, np.uint8)
    canvas[:h_header, :] = 32
    canvas[image_hw[0]*2+gap_size*2+h_header:image_hw[0]*2+gap_size*2+h_header+h_footer, :] = 128

    cv2.putText(img=canvas, text='(a) Left Camera', org=(150, h_header+image_hw[0]+30), fontScale=1, color=(255, 0, 0), thickness=2, fontFace=cv2.LINE_AA)
    # cv2.putText(canvas, f'(b) Traj-Vis {render_desc}', org=(image_hw[1]+gap_size+50, h_header+image_hw[0]+30), fontScale=1, color=(0, 0, 255), thickness=2, fontFace=cv2.LINE_AA)
    cv2.putText(canvas, '(b) Left Camera + Mask', org=(image_hw[1]+gap_size+150, h_header+image_hw[0]+30), fontScale=1, color=(255, 0, 0), thickness=2, fontFace=cv2.LINE_AA)
    cv2.putText(canvas, '(e) Predicted Depth', org=(150, h_header+image_hw[0]*2+gap_size+30), fontScale=1, color=(0, 0, 255), thickness=2, fontFace=cv2.LINE_AA)
    cv2.putText(canvas, '(f) Groundtruth Depth', org=(image_hw[1]+gap_size+150, h_header+image_hw[0]*2+gap_size+30), fontScale=1, color=(255, 0, 0), thickness=2, fontFace=cv2.LINE_AA)

    # header section
    cv2.putText(img=canvas, text=f'{g_video_info.datetime_ex} @ {g_video_info.hostname} @ {g_video_info.git_hexsha} @ {g_video_info.git_is_dirty}',
        org=(30, 35), fontScale=1, color=(0, 0, 255), thickness=2, fontFace=cv2.LINE_AA)
    cv2.putText(img=canvas, text=f'data:  {g_video_info.path_data}',
        org=(30, 70), fontScale=1, color=(255, 255, 0), thickness=2, fontFace=cv2.LINE_AA)
    cv2.putText(img=canvas, text=f'model: {osp.dirname(g_video_info.path_model)}',
        org=(30, 105), fontScale=1, color=(0, 255, 255), thickness=2, fontFace=cv2.LINE_AA)
    cv2.putText(img=canvas, text=f'       {osp.basename(g_video_info.path_model)}',
        org=(30, 140), fontScale=1, color=(0, 255, 255), thickness=2, fontFace=cv2.LINE_AA)

    # footer section
    cv2.putText(img=canvas, text=f'sample_rate:  {g_video_info.sample_rate:4d}',
        org=(30, image_hw[0]*2+gap_size*2+h_header+35), fontScale=1, color=(255, 0, 255), thickness=2, fontFace=cv2.LINE_AA)
    cv2.putText(img=canvas, text=f'max_frames:  {g_video_info.max_frames:4d}',
        org=(30, image_hw[0]*2+gap_size*2+h_header+70), fontScale=1, color=(255, 255, 0), thickness=2, fontFace=cv2.LINE_AA)
    cv2.putText(img=canvas, text=f'fps:             {fps:4.1f}',
        org=(30, image_hw[0]*2+gap_size*2+h_header+105), fontScale=1, color=(0, 255, 255), thickness=2, fontFace=cv2.LINE_AA)

    n_traj_modes = len(traj_modes)
    color_wrong_pose = (0, 255, 255)
    color_right_pose = (0, 0, 255)
    for idx_mode in range(n_traj_modes):
        mode_text = traj_modes[idx_mode]
        idx_row = idx_mode % 2
        idx_col = 2 + idx_mode // 2

        color_tag = color_wrong_pose
        if 'pose-GT' in mode_text:
            color_tag = color_right_pose

        if idx_mode % 2 == 0:
            cv2.putText(canvas, f'({chr(ord("a") + idx_row * 4 + idx_col)}) Traj-Vis {mode_text}', org=((image_hw[1]+gap_size)*idx_col+50, h_header+image_hw[0]+30), fontScale=1, color=color_tag, thickness=2, fontFace=cv2.LINE_AA)
        else:
            cv2.putText(canvas, f'({chr(ord("a") + idx_row * 4 + idx_col)}) Traj-Vis {mode_text}', org=((image_hw[1]+gap_size)*idx_col+50, h_header+image_hw[0]*2+gap_size+30), fontScale=1, color=color_tag, thickness=2, fontFace=cv2.LINE_AA)

    print(f'writing {output_vis_video}')
    logging.info(f'writing {output_vis_video}')

    if len(files) > max_frames:
        files = files[:max_frames]

    for idx_f, file in enumerate(files):
        name_dir, name_base = osp.split(file)
        if data_type == 'matterport':
            color_file = osp.join(name_dir, f'../../../../cam_left/{name_base}')
        else:
            color_file = osp.join(name_dir, f'../../../../color/{name_base}')

        # subfig(0, 0)
        data_color = cv2.imread(color_file)
        base_name = osp.splitext(osp.basename(file))[0]
        frame_text = f'[{idx_f:4d}] - {base_name}'
        cv2.putText(data_color, frame_text, org=(gap_size, h_header+gap_size), fontScale=1, color=(0, 0, 255), thickness=3, fontFace=cv2.LINE_AA)

        if idx_f > 0:
            dt_curr = np.int64(osp.splitext(osp.basename(files[idx_f]))[0])
            dt_prev = np.int64(osp.splitext(osp.basename(files[idx_f-1]))[0])
            ts_diff = int((dt_curr - dt_prev) / 1e6)
            cv2.putText(data_color, f'  time-diff prev: {ts_diff:4d} ms', org=(gap_size, h_header+gap_size+35), fontScale=1, color=(0, 255, 255), thickness=2, fontFace=cv2.LINE_AA)
        if idx_f < len(files) - 1:
            dt_curr = np.int64(osp.splitext(osp.basename(files[idx_f]))[0])
            dt_next = np.int64(osp.splitext(osp.basename(files[idx_f+1]))[0])
            ts_diff = int((dt_next - dt_curr) / 1e6)
            cv2.putText(data_color, f'  time-diff next: {ts_diff:4d} ms', org=(gap_size, h_header+gap_size+70), fontScale=1, color=(0, 255, 255), thickness=2, fontFace=cv2.LINE_AA)

        cv2.putText(data_color, frame_text, org=(gap_size, h_header+gap_size), fontScale=1, color=(0, 0, 255), thickness=3, fontFace=cv2.LINE_AA)
        canvas[h_header:h_header+image_hw[0], 0:image_hw[1], :] = data_color

        # subfig(1, 0)
        data_depth = cv2.imread(file)
        canvas[h_header+image_hw[0]+gap_size:h_header+image_hw[0]*2+gap_size, 0:image_hw[1], :] = data_depth

        # subfig(1, 1)
        depth_gt_file = file.replace('depth_vis', 'depth_gt_vis')
        if osp.exists(depth_gt_file):
            data_depth_gt = cv2.imread(depth_gt_file)
            data_depth_gt_resize = cv2.resize(data_depth_gt, (image_hw[1], image_hw[0]), interpolation=cv2.INTER_LINEAR)
            canvas[h_header+image_hw[0]+gap_size:h_header+image_hw[0]*2+gap_size, image_hw[1]+gap_size:image_hw[1]*2+gap_size, :] = data_depth_gt_resize
        else:
            logging.info(f'  missing {depth_gt_file}')

        # subfig(0, 1)
        '''
        traj_file = osp.join(osp.dirname(osp.dirname(file)), f'{render_tag}_{render_desc}/{idx_f:06d}.png')
        if osp.exists(traj_file):
            logging.info(f'  load {traj_file}')
            data_traj = cv2.imread(traj_file)
            if data_traj is not None:
                canvas[h_header:h_header+image_hw[0], image_hw[1]+gap_size:image_hw[1]*2+gap_size, :] = data_traj
        else:
            logging.info(f'  missing {traj_file}')
        '''
        if data_type == 'matterport':
            file_color_mask = color_file.replace('/cam_left/', '/cam_left_vis/')
            if osp.exists(file_color_mask):
                data_color_mask = cv2.imread(file_color_mask)
                canvas[h_header:h_header+image_hw[0], image_hw[1]+gap_size:image_hw[1]*2+gap_size, :] = data_color_mask

        for idx_mode in range(n_traj_modes):
            mode_text = traj_modes[idx_mode]
            idx_row = idx_mode % 2
            idx_col = 2 + idx_mode // 2

            traj_file = osp.join(osp.dirname(osp.dirname(file)), f'{render_tag}_{mode_text}/{idx_f:06d}.png')
            if osp.exists(traj_file):
                logging.info(f'  load {traj_file}')
                data_traj = cv2.imread(traj_file)
                if data_traj is not None:
                    data_traj_resize = cv2.resize(data_traj, (image_hw[1], image_hw[0]), interpolation=cv2.INTER_LINEAR)
                    canvas[h_header+(image_hw[0]+gap_size)*idx_row:h_header+(image_hw[0]+gap_size)*idx_row+image_hw[0], (image_hw[1]+gap_size)*idx_col:(image_hw[1]+gap_size)*idx_col+image_hw[1], :] = data_traj_resize
            else:
                logging.info(f'  missing {traj_file}')

        video_writer.write(canvas)

    video_writer.release()
    print0(pcolor(f'inference finish .....................', 'blue'))
    return render_desc


def main():
    logging.warning(f'main()')
    args = parse_args()
    logging.info(f'  args: {args}')

    sfm_params = {
        "filer_depth_grad_max": 0.05,
        "filer_depth_max": 6 if not args.data_type=='kitti' else 15,
        "depth_crop_h": 32,
        "depth_crop_w": 32,
        "depth_crop_w": 32,
        "fusion_view_num": 5,
        "fusion_thres_view": 1,
    }

    model_wrapper, image_shape = init_model(args)

    input = args.input
    output_depths_npy = os.path.join(args.output, "depths.npy")
    output_vis_video = os.path.join(args.output, "depths_vis.avi")
    output_tmp_dir = os.path.join(args.output, "tmp")
    sample_rate = args.sample_rate
    data_type = args.data_type 
    ply_mode = args.ply_mode
    max_frames = args.max_frames
    use_depth_gt = args.use_depth_gt
    use_pose_gt = args.use_pose_gt
    mix_video_mode = args.mix_video_mode
    archive_video = args.archive_video

    # ======================================================================== #
    # VideoInfo
    g_video_info.path_model = args.checkpoint
    g_video_info.path_data = input

    g_video_info.sample_rate = sample_rate
    g_video_info.max_frames = max_frames

    g_video_info.print_info()
    # ======================================================================== #

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(output_tmp_dir, exist_ok=True)

    if not mix_video_mode:
        print0(pcolor(f'  model: {args.checkpoint}', 'magenta'))
        print0(pcolor(f'  data:  {input}', 'magenta'))
        desc = inference(model_wrapper, image_shape, input, sample_rate=sample_rate, max_frames=max_frames,
            output_depths_npy=output_depths_npy, output_vis_video=output_vis_video, 
            output_tmp_dir=output_tmp_dir, data_type=data_type,
            ply_mode=ply_mode, use_depth_gt=use_depth_gt, use_pose_gt=use_pose_gt, sfm_params=sfm_params)
    else:
        modes = [(False, False), (False, True), (True, False), (True, True)]
        for use_depth_gt, use_pose_gt in modes:
            # color: red, blue, green, cyan, magenta, yellow
            print0(pcolor(f'  model: {args.checkpoint}', 'magenta'))
            print0(pcolor(f'  data:  {input}', 'magenta'))
            inference(model_wrapper, image_shape, input, sample_rate=sample_rate, max_frames=max_frames,
                output_depths_npy=output_depths_npy, output_vis_video=output_vis_video, 
                output_tmp_dir=output_tmp_dir, data_type=data_type,
                ply_mode=ply_mode, use_depth_gt=use_depth_gt, use_pose_gt=use_pose_gt, sfm_params=sfm_params)

        desc = 'depth-GT_pose-GT'

    # archive final video
    if not osp.exists(archive_video):
        os.makedirs(archive_video)

    tmp_tags = osp.splitext(output_vis_video)
    src_name = f'{tmp_tags[0]}_{desc}{tmp_tags[1]}'
    data_case_name = input.split('/')[-2]
    dst_name = osp.join(archive_video,
        f'{data_case_name}_SR{sample_rate:03d}_MF{max_frames:04d}'
        f'_{g_video_info.datetime}@{g_video_info.hostname}@{g_video_info.git_hexsha[:8]}@{g_video_info.git_is_dirty}'
        f'_{osp.basename(g_video_info.path_model)}.avi')
    subprocess.call(['cp', src_name, dst_name])

    # clean tmp dir
    # import shutil
    # shutil.rmtree(output_tmp_dir)


if __name__ == '__main__':
    setup_log('kneron_infer_video.log')
    time_beg_infer_video = time.time()

    np.set_printoptions(precision=6, suppress=True)
    main()
    # g_video_info.print_info()

    time_end_infer_video = time.time()
    logging.warning(f'elapsed {time_end_infer_video - time_beg_infer_video:.6f} seconds.')
    print0(pcolor(f'\nelapsed {time_end_infer_video - time_beg_infer_video:.6f} seconds.\n', 'red'))
