# -*- coding=utf8 -*-

'''
References
----------
https://pythonprogramming.altervista.org/png-to-gif/
    Png to animated Gif with Python

https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#saving
    PIL GIF Saving

https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.save
    PIL Image save

https://www.wfonts.com/font/microsoft-sans-serif
    Free Font Microsoft Sans Serif
'''

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import PIL
import glob
import numpy as np
import logging
import time
import cv2
import numpy as np

import os
import os.path as osp
import sys
lib_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(lib_dir)
from dro_sfm.utils.setup_log import setup_log


def load_images():
    logging.warning(f'load_images()')

    # http://code.nabla.net/doc/PIL/api/PIL/PngImagePlugin/PIL.PngImagePlugin.PngImageFile.html
    #     PngImageFile Class

    # Create the frames
    frames = []
    imgs = sorted(glob.glob("/home/sigma/vgithub-xyang9527/kneron_figure/scannet/*.png"))
    for idx_f, item in enumerate(imgs):
        new_frame = Image.open(item)
        logging.info(f'  {idx_f:4d} {osp.basename(item)}')
        logging.info(f'  {type(new_frame)} {new_frame.size}')
        # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.crop
        #     (left, upper, right, lower)
        frames.append(new_frame.crop((60, 0, 3280, 1080)))

    # frames = add_text_with_pil(frames)
    # frames = add_text_with_opencv(frames)
    return frames


def add_text_with_pil(images):
    logging.warning(f'add_text_with_pil(..)')
    # https://www.geeksforgeeks.org/adding-text-on-image-using-python-pil/
    #     Adding Text on Image using Python – PIL
    # https://stackoverflow.com/a/16377244

    font = ImageFont.truetype("/home/sigma/vgithub-xyang9527/kneron_figure/ttf/micro_se.ttf", 16)
    font = ImageFont.truetype("/home/sigma/vgithub-xyang9527/kneron_figure/ttf/CHGENE1.ttf", 64)
    # git@github.com:sonatype/maven-guide-zh.git
    font = ImageFont.truetype("/home/sigma/vgithub-xyang9527/kneron_figure/ttf/simsun.ttc", 64)
    for idx, img in enumerate(images):
        draw = ImageDraw.Draw(img)
        # https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html#PIL.ImageDraw.ImageDraw.text
        # draw.text((100, 300), f'Frame {idx:4d}', fill=(255, 0, 0), font=font, stroke_width=0)
        draw.text((100, 300), f'帧序号 {idx:4d}', fill=(255, 0, 0), font=font, stroke_width=0)

    return images


def add_text_with_opencv(images):
    logging.warning(f'add_text_with_opencv(..)')
    # OpenCV putText is only able to support a small ascii subset of characters
    #     and does not support unicode characters which are other symboles
    #     like chinese and arabic characters.
    for idx, img in enumerate(images):
        nimg = np.array(img)
        ocvim = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

        org = (100, 300)
        font_scale = 1
        color = (0, 0, 255)
        thickness = 2
        text=f'Frame {idx:4d}'
        ocvim = cv2.putText(img=ocvim, text=text, org=org, fontScale=font_scale, color=color, thickness=thickness, fontFace=cv2.LINE_AA)
        ocvim_back = cv2.cvtColor(ocvim, cv2.COLOR_BGR2RGB)
        img_back = Image.fromarray(ocvim_back.astype('uint8'), 'RGB')
        images[idx] = img_back
    return images


def write_gif(frames):
    logging.warning(f'write_gif()')
    '''
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#saving
    im.save(out, save_all=True, append_images=[im1, im2, ...])

        save_all
        append_images
        include_color_table
        interlace
        disposal
        palette
        optimize
        transparency
        duration
        loop
        comment
    '''
    # Save into a GIF file that loops forever
    frames[0].save('/home/sigma/vgithub-xyang9527/kneron_figure/png_to_gif.gif', format='GIF',
                  append_images=frames[1:],
                  save_all=True,
                  duration=1500, loop=0, comment=b'Aligned Scannet Point Cloud')


def process_scannet():
    input_dir = '/home/sigma/vgithub-xyang9527/kneron_figure/scannet'
    input_names = sorted(glob.glob("/home/sigma/vgithub-xyang9527/kneron_figure/scannet/*.png"))
    info = [
    {'name': 'A000_scannet_local_coord_000000.png', 'left_caption': '单帧 000 相机局部坐标点云', 'right_caption': '帧 000 图像', 'bottom_caption': 'Scannet 单帧点云'},
    {'name': 'A090_scannet_local_coord_000090.png', 'left_caption': '单帧 090 相机局部坐标点云', 'right_caption': '帧 090 图像', 'bottom_caption': 'Scannet 单帧点云'},
    {'name': 'A180_scannet_local_coord_000180.png', 'left_caption': '单帧 180 相机局部坐标点云', 'right_caption': '帧 180 图像', 'bottom_caption': 'Scannet 单帧点云'},
    {'name': 'A270_scannet_local_coord_000270.png', 'left_caption': '单帧 270 相机局部坐标点云', 'right_caption': '帧 270 图像', 'bottom_caption': 'Scannet 单帧点云'},
    {'name': 'A360_scannet_local_coord_000360.png', 'left_caption': '单帧 360 相机局部坐标点云', 'right_caption': '帧 360 图像', 'bottom_caption': 'Scannet 单帧点云'},
    {'name': 'A450_scannet_local_coord_000450.png', 'left_caption': '单帧 450 相机局部坐标点云', 'right_caption': '帧 450 图像', 'bottom_caption': 'Scannet 单帧点云'},
    {'name': 'A540_scannet_local_coord_000540.png', 'left_caption': '单帧 540 相机局部坐标点云', 'right_caption': '帧 540 图像', 'bottom_caption': 'Scannet 单帧点云'},
    {'name': 'A630_scannet_local_coord_000630.png', 'left_caption': '单帧 630 相机局部坐标点云', 'right_caption': '帧 630 图像', 'bottom_caption': 'Scannet 单帧点云'},
    {'name': 'A720_scannet_local_coord_000720.png', 'left_caption': '单帧 720 相机局部坐标点云', 'right_caption': '帧 720 图像', 'bottom_caption': 'Scannet 单帧点云'},
    {'name': 'A810_scannet_local_coord_000810.png', 'left_caption': '单帧 810 相机局部坐标点云', 'right_caption': '帧 810 图像', 'bottom_caption': 'Scannet 单帧点云'},
    {'name': 'A900_scannet_local_coord_000900.png', 'left_caption': '单帧 900 相机局部坐标点云', 'right_caption': '帧 900 图像', 'bottom_caption': 'Scannet 单帧点云'},
    {'name': 'A990_scannet_local_coord_000990.png', 'left_caption': '单帧 990 相机局部坐标点云', 'right_caption': '帧 990 图像', 'bottom_caption': 'Scannet 单帧点云'},
    {'name': 'B180_scannet_local_coord_000000_000180.png', 'left_caption': '多帧 000-180 相机局部坐标点云', 'right_caption': '帧 180 图像', 'bottom_caption': 'Scannet 未对齐的多帧点云'},
    {'name': 'B450_scannet_local_coord_000000_000450.png', 'left_caption': '多帧 000-450 相机局部坐标点云', 'right_caption': '帧 450 图像', 'bottom_caption': 'Scannet 未对齐的多帧点云'},
    {'name': 'B720_scannet_local_coord_000000_000720.png', 'left_caption': '多帧 000-720 相机局部坐标点云', 'right_caption': '帧 720 图像', 'bottom_caption': 'Scannet 未对齐的多帧点云'},
    {'name': 'B990_scannet_local_coord_000000_000990.png', 'left_caption': '多帧 000-990 相机局部坐标点云', 'right_caption': '帧 990 图像', 'bottom_caption': 'Scannet 未对齐的多帧点云'},
    {'name': 'C000_scannet_global_coord_000000.png', 'left_caption': '首帧坐标下新增帧 000 点云', 'right_caption': '帧 000 图像', 'bottom_caption': 'Scannet 使用 pose 对齐点云'},
    {'name': 'C090_scannet_global_coord_000090.png', 'left_caption': '首帧坐标下新增帧 090 点云', 'right_caption': '帧 090 图像', 'bottom_caption': 'Scannet 使用 pose 对齐点云'},
    {'name': 'C180_scannet_global_coord_000180.png', 'left_caption': '首帧坐标下新增帧 180 点云', 'right_caption': '帧 180 图像', 'bottom_caption': 'Scannet 使用 pose 对齐点云'},
    {'name': 'C270_scannet_global_coord_000270.png', 'left_caption': '首帧坐标下新增帧 270 点云', 'right_caption': '帧 270 图像', 'bottom_caption': 'Scannet 使用 pose 对齐点云'},
    {'name': 'C360_scannet_global_coord_000360.png', 'left_caption': '首帧坐标下新增帧 360 点云', 'right_caption': '帧 360 图像', 'bottom_caption': 'Scannet 使用 pose 对齐点云'},
    {'name': 'C450_scannet_global_coord_000450.png', 'left_caption': '首帧坐标下新增帧 450 点云', 'right_caption': '帧 450 图像', 'bottom_caption': 'Scannet 使用 pose 对齐点云'},
    {'name': 'C540_scannet_global_coord_000540.png', 'left_caption': '首帧坐标下新增帧 540 点云', 'right_caption': '帧 540 图像', 'bottom_caption': 'Scannet 使用 pose 对齐点云'},
    {'name': 'C630_scannet_global_coord_000630.png', 'left_caption': '首帧坐标下新增帧 630 点云', 'right_caption': '帧 630 图像', 'bottom_caption': 'Scannet 使用 pose 对齐点云'},
    {'name': 'C720_scannet_global_coord_000720.png', 'left_caption': '首帧坐标下新增帧 720 点云', 'right_caption': '帧 720 图像', 'bottom_caption': 'Scannet 使用 pose 对齐点云'},
    {'name': 'C810_scannet_global_coord_000810.png', 'left_caption': '首帧坐标下新增帧 810 点云', 'right_caption': '帧 810 图像', 'bottom_caption': 'Scannet 使用 pose 对齐点云'},
    {'name': 'C900_scannet_global_coord_000900.png', 'left_caption': '首帧坐标下新增帧 900 点云', 'right_caption': '帧 900 图像', 'bottom_caption': 'Scannet 使用 pose 对齐点云'},
    {'name': 'C990_scannet_global_coord_000990.png', 'left_caption': '首帧坐标下新增帧 990 点云', 'right_caption': '帧 990 图像', 'bottom_caption': 'Scannet 使用 pose 对齐点云'},
    {'name': 'D990A_scannet_global_coord_000990_A.png', 'left_caption': '对齐后点云', 'right_caption': '帧 990 图像', 'bottom_caption': 'Scannet 使用 pose 对齐点云'},
    {'name': 'D990B_scannet_global_coord_000990_B.png', 'left_caption': '对齐后点云', 'right_caption': '帧 990 图像', 'bottom_caption': 'Scannet 使用 pose 对齐点云'},
    {'name': 'D990C_scannet_global_coord_000990_C.png', 'left_caption': '对齐后点云', 'right_caption': '帧 990 图像', 'bottom_caption': 'Scannet 使用 pose 对齐点云'}]

    font32 = ImageFont.truetype("/home/sigma/vgithub-xyang9527/kneron_figure/ttf/simsun.ttc", 32)
    font96 = ImageFont.truetype("/home/sigma/vgithub-xyang9527/kneron_figure/ttf/simsun.ttc", 96)
    # view pixel coord with "GNU Image Manipulation Program"
    pos_left_caption = (340, 180)
    pos_right_caption = (2400, 180)
    pos_bottom_caption = (1000, 800)

    frames = []
    for item in info:
        str_name = item['name']
        str_left_caption = item['left_caption']
        str_right_caption = item['right_caption']
        str_bottom_caption = item['bottom_caption']

        new_frame = Image.open(osp.join(input_dir, str_name))
        new_frame = new_frame.crop((60, 0, 3280, 1080))
        draw = ImageDraw.Draw(new_frame)
        draw.text(xy=pos_left_caption, text=str_left_caption, fill=(255, 0, 0), font=font32)
        draw.text(xy=pos_right_caption, text=str_right_caption, fill=(255, 0, 0), font=font32)
        draw.text(xy=pos_bottom_caption, text=str_bottom_caption, fill=(255, 0, 0), font=font96)
        frames.append(new_frame)

    # Write GIF
    frames[0].save('/home/sigma/vgithub-xyang9527/kneron_figure/gif/scannet_pointcloud_unaligned.gif', format='GIF',
                  append_images=frames[1:16],
                  save_all=True,
                  duration=1200, loop=0)

    frames[16].save('/home/sigma/vgithub-xyang9527/kneron_figure/gif/scannet_pointcloud_aligned.gif', format='GIF',
                  append_images=frames[17:],
                  save_all=True,
                  duration=1200, loop=0)

    # Write video
    video = cv2.VideoWriter('/home/sigma/vgithub-xyang9527/kneron_figure/avi/scannet_pointcloud_unaligned.avi', cv2.VideoWriter_fourcc(*'XVID'), 1, frames[0].size)
    for f in frames[:16]:
        video.write(cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR))

    video = cv2.VideoWriter('/home/sigma/vgithub-xyang9527/kneron_figure/avi/scannet_pointcloud_aligned.avi', cv2.VideoWriter_fourcc(*'XVID'), 1, frames[16].size)
    for f in frames[16:]:
        video.write(cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR))
    pass


if __name__ == '__main__':
    setup_log('kneron_img2gif.log')
    logging.info(f'  PIL.__version__:    {PIL.__version__}')
    logging.info(f'  cv2.__version__:    {cv2.__version__}')

    time_beg_img2gif = time.time()

    np.set_printoptions(precision=6, suppress=True)
    # images = load_images()
    # write_gif(images)
    process_scannet()

    time_end_img2gif = time.time()

    print(f'img2gif.py elapsed {time_end_img2gif - time_beg_img2gif:.6f} seconds.')
