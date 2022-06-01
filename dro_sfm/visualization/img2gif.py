# -*- coding=utf8 -*-

"""
References
----------
  https://pythonprogramming.altervista.org/png-to-gif/
    Png to animated Gif with Python

  https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#saving
    PIL GIF Saving

  https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.save
    PIL Image save
"""

from PIL import Image
import glob
import numpy as np
import logging
import time

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
        print(f'{idx_f:4d} {osp.basename(item)}')
        print(f'  {type(new_frame)} {new_frame.size}')
        # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.crop
        #     (left, upper, right, lower)
        frames.append(new_frame.crop((60, 0, 3280, 1080)))

    return frames

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


if __name__ == '__main__':
    setup_log('kneron_img2gif.log')
    time_beg_img2gif = time.time()

    np.set_printoptions(precision=6, suppress=True)
    images = load_images()
    write_gif(images)

    time_end_img2gif = time.time()
    print(f'img2gif.py elapsed {time_end_img2gif - time_beg_img2gif:.6f} seconds.')
