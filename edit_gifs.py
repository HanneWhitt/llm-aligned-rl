from PIL import Image
from PIL import ImageDraw
import numpy as np
import os
from shutil import rmtree

ep = 0
folder_list = []

ep_boundaries = [0, 12, 25, 36, 43, 54, 67, 76, 87, 98]
# ep_boundaries = [0, 11, 22, 33, 42, 58, 102, 115, 127, 141]

# LLM_feedback

gif = Image.open('naive_policy.gif')
for i in range(gif.n_frames):
    if i in ep_boundaries:
        ep += 1
        frame_count = 0
        folder = f'../gif_edits/naive_edits/episode_{ep}'
        folder_list.append(folder)
        if os.path.isdir(folder):
            rmtree(folder)
        os.mkdir(folder)
    frame_count += 1
    gif.seek(i)
    gif.save(f'{folder}/{frame_count}.png')

