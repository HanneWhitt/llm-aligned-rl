from PIL import Image
from PIL import ImageDraw
import sys
import os
from shutil import rmtree
    

def edit_gif(frame_folders, out, text):
    ep = 0
    for fold in frame_folders:
        ep += 1
        print(fold)
        f = [im for im in os.listdir(fold) if im.endswith('.png')]
        n_images = len(f)
        print(n_images)
        f = [f"{fold}{i + 1}.png" for i in range(n_images)]
        outfolder = f'{out}episode_{ep}/'
        if os.path.isdir(outfolder):
            rmtree(outfolder)
        os.mkdir(outfolder)
        for i, image in enumerate(f):
            im = Image.open(image)
            I1 = ImageDraw.Draw(im)
            I1.fontmode = "L"
            I1.text((12, 12), text, fill=(0, 0, 0), font_size=18)
            I1.text((12, 50+20), f"Episode {ep} / 10,000", fill=(0, 0, 0), font_size=16)
            I1.text((12, 86+20), f"t = {i + 1}", fill=(0, 0, 0), font_size=16)
            im.save(f'{outfolder}{i + 1}.png')



def make_gif(frame_folders, out):
    frames = []
    ep = 0
    for fold in frame_folders:
        ep += 1
        print(fold)
        f = [im for im in os.listdir(fold) if im.endswith('.png')]
        n_images = len(f)
        print(n_images)
        f = [f"{fold}{i + 1}.png" for i in range(n_images)]
        frames += [Image.open(im) for im in f]
    frame_one = frames[0]
    frame_one.save(out, format="GIF", append_images=frames,
        save_all=True, duration=500, loop=0)
    print('Saved to ', out)
    

if __name__ == "__main__":
    # 
    # frame_folders = sys.argv[1:]

    frame_folders = [f'../gif_edits/naive_eps/episode_{i}/' for i in range(1,11)]

    edit_gif(frame_folders, out='../gif_edits/naive_edits/', text='Naive policy')

    # frame_folders = [f'../gif_edits/naive_edits/episode_{i}/' for i in range(1,11)]

    # make_gif(frame_folders, out='../gif_edits/naive_policy.gif')

    frame_folders = [f'../gif_edits/naive_edits/episode_{i}/' for i in range(1,11)]

    make_gif(frame_folders, out='../gif_edits/naive_policy.gif')
