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
import os.path as osp

# Create the frames
frames = []
imgs = sorted(glob.glob("/home/sigma/vgithub-xyang9527/kneron_figure/scannet/*.png"))
for idx_f, item in enumerate(imgs):
    new_frame = Image.open(item)
    frames.append(new_frame)
    print(f'{idx_f:4d} {osp.basename(item)}')

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
