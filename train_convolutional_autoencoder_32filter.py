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


# Define the autoencoder architecture
#  defining encoder
class Encoder(nn.Module):
  def __init__(self, in_channels=3, out_channels=16, latent_dim=200, act_fn=nn.ReLU()):
    super().__init__()

    self.net = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), # (448, 384)
        act_fn,
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, 32, stride=32), #  (14, 12)
        act_fn,
        nn.BatchNorm2d(out_channels),
        # nn.Conv2d(out_channels, 2*out_channels, 3, padding=1), #  (14, 12)
        # act_fn,
        # nn.BatchNorm2d(2*out_channels),
        nn.Flatten(),
        nn.Linear(out_channels*14*12, latent_dim),
        act_fn
    )

  def forward(self, x):
    #x = x.view(-1, 3, 32, 32)
    output = self.net(x)
    return output


#  defining decoder
class Decoder(nn.Module):
  def __init__(self, in_channels=3, out_channels=16, latent_dim=200, act_fn=nn.ReLU()):
    super().__init__()

    self.out_channels = out_channels

    self.linear = nn.Sequential(
        nn.Linear(latent_dim, out_channels*14*12),
        act_fn,
        nn.BatchNorm1d(out_channels*14*12)
    )

    self.conv = nn.Sequential(
        # nn.ConvTranspose2d(2*out_channels, out_channels, 3, padding=1), #  (14, 12)
        # act_fn,
        # nn.BatchNorm2d(out_channels),
        nn.ConvTranspose2d(out_channels, out_channels, 32, stride=32), # (448, 384)
        act_fn,
        nn.BatchNorm2d(out_channels),
        nn.ConvTranspose2d(out_channels, in_channels, 3, padding=1), 
        nn.Sigmoid()
    )

  def forward(self, x):
    output = self.linear(x)
    output = output.view(-1, self.out_channels, 12, 14)
    output = self.conv(output)
    return output


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
                                           batch_size=64, 
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=64)
 
# Move the model to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Initialize the autoencoder
model = Autoencoder(Encoder(), Decoder(), device)
  
# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = MultiStepLR(optimizer, milestones=[400], gamma=0.1)


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
    epoch_lses = []
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
        ls = loss.item()
        epoch_lses.append(ls)
    ls = np.mean(epoch_lses[-int(1000/64):])
    lses.append(ls)
    np.savetxt("../autoencoder_samples/32_filter_higher_lr/loss_record.csv", lses, delimiter=",")
    print('Epoch [{}/{}], Loss: {:.6f}'.format(epoch+1, num_epochs, ls))
    
    if ls < min_loss:
        print('NEW MIN LOSS')
        min_loss = ls
        torch.save(model.state_dict(), '../autoencoder_samples/32_filter_higher_lr/conv_autoencoder.pth')
        if epoch > 50:
           save_top_example(img, f'../autoencoder_samples/32_filter_higher_lr/epoch_{epoch}_original.png')
           save_top_example(output, f'../autoencoder_samples/32_filter_higher_lr/epoch_{epoch}_reconstructed.png')
    if epoch % 5== 0:
        save_top_example(img, f'../autoencoder_samples/32_filter_higher_lr/epoch_{epoch}_original.png')
        save_top_example(output, f'../autoencoder_samples/32_filter_higher_lr/epoch_{epoch}_reconstructed.png')
    scheduler.step()

# # Save the model
# torch.save(model.state_dict(), '../autoencoder_samples/32_filter_higher_lr/conv_autoencoder.pth')









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
