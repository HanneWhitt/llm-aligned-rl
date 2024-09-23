import gymnasium as gym
import homegrid
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt 
from PIL import Image
import numpy as np
from autoencoder_dataloader import RandomDataset
from torch.optim.lr_scheduler import MultiStepLR
from piqa import SSIM
import os



# Define the autoencoder architecture
#  defining encoder
class Encoder(nn.Module):
  def __init__(self, in_channels=3, out_channels=16, latent_dim=200, act_fn=nn.ReLU()):
    super().__init__()

    self.act_fn = act_fn

    self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1) # (448, 384)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.conv2 = nn.Conv2d(out_channels + in_channels, out_channels, 32, stride=32) #  (14, 12)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1) #  (14, 12)
    self.bn3 = nn.BatchNorm2d(out_channels)

    self.conv1d = nn.Conv2d(2*out_channels, out_channels, 3, padding=1)
    self.bn4 = nn.BatchNorm2d(out_channels)
    
    self.flat = nn.Flatten()
    
    self.linear = nn.Linear(out_channels*14*12, latent_dim)

  def forward(self, x):
    
    c1 = self.conv1(x)
    c1 = self.bn1(c1)
    c1 = self.act_fn(c1)
    c1 = torch.cat((c1, x), 1)

    c1 = self.conv2(c1)
    c1 = self.bn2(c1)
    c1 = self.act_fn(c1)
    
    c2 = self.conv3(c1)
    c2 = self.bn3(c2)
    c2 = self.act_fn(c2)
    c2 = torch.cat((c2, c1), 1)

    c2 = self.conv1d(c2)
    c2 = self.bn4(c2)
    
    c2 = self.flat(c2)
    c2 = self.linear(c2)

    return c2


#  defining decoder
class Decoder(nn.Module):
  def __init__(self, in_channels=3, out_channels=16, latent_dim=200, act_fn=nn.ReLU()):
    super().__init__()

    self.out_channels = out_channels
    self.act_fn = act_fn

    self.linear = nn.Sequential(
        nn.Linear(latent_dim, 2*out_channels*14*12),
        act_fn,
        nn.BatchNorm1d(2*out_channels*14*12)
    )

    self.convt1 = nn.ConvTranspose2d(2*out_channels, out_channels, 3, padding=1) #  (14, 12)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.convt2 = nn.ConvTranspose2d(3*out_channels, out_channels, 32, stride=32) # (448, 384)
    self.bn2 = nn.BatchNorm2d(out_channels)
    self.convt3 = nn.ConvTranspose2d(out_channels, out_channels, 3, padding=1)
    self.bn3 = nn.BatchNorm2d(out_channels)
    self.convt1x1 = nn.ConvTranspose2d(2*out_channels, in_channels, 1)
    self.sig = nn.Sigmoid()

  def forward(self, x):
    output = self.linear(x)
    output = output.view(-1, 2*self.out_channels, 12, 14)

    ct1 = self.convt1(output)
    ct1 = self.bn1(ct1)
    ct1 = self.act_fn(ct1)
    ct1 = torch.cat((ct1, output), 1)

    ct1 = self.convt2(ct1)
    ct1 = self.bn1(ct1)
    ct1 = self.act_fn(ct1)
    
    ct2 = self.convt3(ct1)
    ct2 = self.bn3(ct2)
    ct2 = self.act_fn(ct2)
    ct2 = torch.cat((ct2, ct1), 1)

    ct2 = self.convt1x1(ct2)
    ct2 = self.sig(ct2)
        
    return ct2


#  defining autoencoder
class Autoencoder(nn.Module):
  def __init__(self, encoder, decoder, device):
    super().__init__()
    self.encoder = encoder
    self.encoder.to(device)

    self.decoder = decoder
    self.decoder.to(device)

  def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
 
 

# # Define transform
# transform = transforms.Compose([
#     transforms.Resize((64, 64)),
#     transforms.ToTensor(),
# ])
 
# Load dataset
train_dataset = RandomDataset()
test_dataset = RandomDataset()

# Define the dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=32, 
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=32)
 
# Move the model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


FOLDER = '../autoencoder_samples/32_filter_SSIM_skip/'

LOAD_MODEL_FROM = None#'../autoencoder_samples/32_filter/conv_autoencoder.pth'


if os.path.isdir(FOLDER):
    print('FOLDER EXISTS -', FOLDER)
    input('Sure you want to overwrite?')
    input('Sure you sure?')
else:
    os.mkdir(FOLDER)

# Initialize the autoencoder
model = Autoencoder(Encoder(), Decoder(), device)

if LOAD_MODEL_FROM:
    model.load_state_dict(torch.load(LOAD_MODEL_FROM, weights_only=True))
    print('LOADED MODEL FROM ', LOAD_MODEL_FROM, ', resuming training')


# Define the loss function and optimizer

class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)

criterion = SSIMLoss().cuda()
 # if you need GPU support
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = MultiStepLR(optimizer, milestones=[100, 200, 300, 400, 500, 600], gamma=0.5)


def save_top_example(img, savefile):
    saveexample  = (img[0].permute(1, 2, 0).detach().cpu().numpy()*255).astype('uint8')
    imsave = Image.fromarray(saveexample)
    imsave.save(savefile)
    print(f"IMAGE SAVED TO {savefile}")


min_loss = 1000
lses = []



# Train the autoencoder
num_epochs = 1000000
for epoch in range(num_epochs):
    epoch_losses = []
    for data in train_loader:
        img = data

        # print(img[0])
        # print(type(img))
        # print(img.shape)
        # print(img.dtype)
        # input()

        img = img.to(device)
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, img)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    ls = np.mean(epoch_losses)
    lses.append(ls)
    np.savetxt(FOLDER + "loss_record.csv", lses, delimiter=",")
    print('Epoch [{}/{}], Loss: {:.6f}'.format(epoch+1, num_epochs, ls))
    
    if ls < min_loss:
        print('NEW MIN LOSS')
        min_loss = ls
        torch.save(model.state_dict(), FOLDER + 'conv_autoencoder.pth')
        if epoch % 5== 0 or epoch > 50:
            save_top_example(img, FOLDER + f'epoch_{epoch}_original.png')
            save_top_example(output, FOLDER + f'epoch_{epoch}_reconstructed.png')
        
        
    scheduler.step()

# # Save the model
# torch.save(model.state_dict(), '../autoencoder_samples/32_filter/conv_autoencoder.pth')









# env = gym.make("homegrid-cat")


# for i in range(1000):
#     s = time.time()
#     obs, info = env.reset()
#     for st in range(100):
#         #a = random.choice([0, 1, 2, 3])
#         for layer in [3, 2, 1, 0]:
#             print(obs[:, :, layer])
#         a = int(input('action -->')) - 1
#         obs, reward, terminated, truncated, info = env.step(a)

#         if terminated or truncated:
#             break
#     print('Run ', i, time.time() - s, 'seconds')
