
import numpy as np
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
import homegrid
import torch



class RandomDataset(Dataset):

    def __init__(self, return_reduced=False):
        self.env = gym.make("homegrid-cat")
        self.env.reset()
        self.env_terminated = False
        self.return_reduced = return_reduced
    
    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        if self.env_terminated:
            obs, info = self.env.reset()
            self.env_terminated = False
        else:
            a = np.random.choice([0, 1, 2, 3])
            obs, reward, terminated, truncated, info = self.env.step(a)
            if terminated or truncated:
                self.env_terminated = True
        if self.return_reduced:
            im = self.env.binary_grid
        else:
            im = self.env.render()
        im = im.astype('float32')
        im = im/255
        im = torch.from_numpy(im)
        im = im.permute(2, 0, 1)
        return im
    

if __name__ == "__main__":

    from PIL import Image

    ds = RandomDataset()



    new_item = ds.__getitem__(1).permute(1, 2, 0)

    print(new_item.shape)

    new_item = new_item.numpy()

    def save_top_example(img, savefile):
        imsave = Image.fromarray(img)
        imsave.save(savefile)
        print(f"IMAGE SAVED TO {savefile}")

    save_top_example(new_item, '../autoencoder_samples/first_attempt/test.png')
    
