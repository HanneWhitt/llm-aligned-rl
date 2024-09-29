from PIL import Image
import sys
import os

def make_gif(frame_folders, out='../figures/test.gif'):
    frames = []
    for fold in frame_folders:
        print(fold)
        f = [im for im in os.listdir(fold) if im.endswith('.png')]
        n_images = len(f)
        print(n_images)
        f = [f"{fold}{i}.png" for i in range(n_images)]
        frames += [Image.open(image) for image in f]
        print(len(frames))
    frame_one = frames[0]
    frame_one.save(out, format="GIF", append_images=frames,
        save_all=True, duration=500, loop=0)
    print('Saved to ', out)
    

if __name__ == "__main__":
    # 
    # frame_folders = sys.argv[1:]

    frame_folders = [f'../final_analysis_set/episode_{i}/' for i in range(10)]

    make_gif(frame_folders, out='../figures/LLM_feedback_policy.gif')