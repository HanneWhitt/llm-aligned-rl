import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from shutil import rmtree



def get_judgement(json_content):
    return json_content['feedback'][0]['question_3']


# agent_indexes = \
#     [[i, j] for j in range(8, 14) for i in [2, 4, 5, 6]] + \
#     [[3, 8], [3, 11], [3, 12]] + \
#     [[7, i] for i in range(1, 14)] + \
    

# def get_flattened_reduced(im):
#     assert im.shape == (12, 14, 4)
#     # agent layer
    

def get_sequence(json_content, reduce=True, flatten=False):
    reduced_format_file = json_content["reduced_format_file"]
    sequence = np.load(reduced_format_file)
    if reduce:
        sequence = sequence[:, :, 1:-1, :-1]
    if flatten:
        n_images, image_i, image_j, image_channels = sequence.shape
        sequence = sequence.reshape(n_images, image_i*image_j*image_channels)
    return sequence


def zero_pad(sequence, pad):
    n_images, image_i, image_j, image_channels = sequence.shape
    zeros = np.zeros((pad, image_i, image_j, image_channels))
    zeros[-n_images:, :, :, :] = sequence
    return zeros
    


def load_example(dataset_folder, ep_name, reduce=True, flatten=False):
    json_file = f'{dataset_folder}{ep_name}/{ep_name}.json'
    with open(json_file) as f:
        json_content = json.load(f)
    judgement = get_judgement(json_content)
    sequence = get_sequence(json_content, reduce=reduce, flatten=flatten)
    return judgement, sequence


# Define Convolutional LSTM classifier model
class ConvLSTMClassifier(nn.Module):
    def __init__(
        self,
        batch_size = 32,
        sequence_length = 30,
        input_size = 12,
        input_channels = 3,
        filters_mid = 16,
        filters_final = 4,
        kernel_size = 3,
        lstm_hidden_size = 64,
        lstm_n_layers = 2
    ):
        super(ConvLSTMClassifier, self).__init__()

        self.batch_size = batch_size
        self.input_size = input_size
        self.input_channels = input_channels
        self.scaled_input_size = input_size//4
        self.sequence_length = sequence_length
        self.filters_final = filters_final
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_n_layers = lstm_n_layers

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(
            input_channels,
            filters_mid,
            kernel_size,
            stride=1,
            padding=1
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # (6, 6)
        self.conv2 = nn.Conv2d(
            filters_mid,
            filters_final,
            kernel_size,
            stride=1,
            padding=1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # (3,3)
    
        lstm_input_size = filters_final*self.scaled_input_size**2
        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, lstm_n_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        x = x.view(self.batch_size*self.sequence_length, self.input_channels, self.input_size, self.input_size)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = x.view(self.batch_size, self.sequence_length, self.filters_final, self.scaled_input_size, self.scaled_input_size)
        x = x.reshape(self.batch_size, self.sequence_length, self.filters_final*self.scaled_input_size*self.scaled_input_size)

        h0 = torch.zeros(self.lstm_n_layers, x.size(0), self.lstm_hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm_n_layers, x.size(0), self.lstm_hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = torch.flatten(self.sigmoid(out))
        return out

min_loss = 1000

if __name__ == '__main__':

    DATASET_FOLDER = '../trajectory_dataset_3/'
    truncation = 30
    train_examples = 8000
    batch_size = 128
    RESULTS_FOLDER = '../sequence_model_results/first_attempt/'

    if os.path.isdir(RESULTS_FOLDER):
        input('SURE YOU WANT TO DELETE ' + RESULTS_FOLDER)
        input("SURE YOU'RE SURE?")
        rmtree(RESULTS_FOLDER)
    os.mkdir(RESULTS_FOLDER)


    episodes = os.listdir(DATASET_FOLDER)

    episodes = [load_example(DATASET_FOLDER, ep, reduce=True) for ep in episodes]

    episodes = [(judge, seq) for judge, seq in episodes if seq.shape[0] <= truncation]

    episodes = [(judge, zero_pad(seq, truncation)) for judge, seq in episodes]

    episodes = [(judge, seq) for judge, seq in episodes]

    train = episodes[:train_examples]
    test = episodes[train_examples:]


    # Move the model to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)


    train_input = torch.from_numpy(np.array([seq for judge, seq in train]).astype('float32'))
    train_target =  torch.from_numpy(np.stack([int(judge) for judge, seq in train]).astype('float32'))
    test_input = torch.from_numpy(np.array([seq for judge, seq in test]).astype('float32'))
    test_target = torch.from_numpy(np.stack([int(judge) for judge, seq in test]).astype('float32'))

    train_dataset = TensorDataset(train_input, train_target)
    test_dataset = TensorDataset(test_input, test_target)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)


    model = ConvLSTMClassifier(batch_size=batch_size).to(device)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    #scheduler = MultiStepLR(optimizer, milestones=[400, 800, 1200], gamma=0.5)

    num_epochs = 100
    for epoch in range(num_epochs):
        lses = []
        for inpt, target in train_loader:
            inpt = torch.permute(inpt, (0, 1, 4, 2, 3))
            inpt = inpt.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(inpt)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            ls = loss.item()
            lses.append(ls)

        ls = np.mean(lses)
        print('Epoch [{}/{}], Loss: {:.6f}'.format(epoch+1, num_epochs, ls))
        np.savetxt(f"{RESULTS_FOLDER}loss_record.csv", lses, delimiter=",")


        if ls < min_loss:
            print('NEW MIN LOSS')
            min_loss = ls
            torch.save(model.state_dict(), '{RESULTS_FOLDER}sequence_model.pth')

