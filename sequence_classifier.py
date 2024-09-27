import os
import json
import numpy as np
import torch
from torch import nn


def get_judgement(json_content):
    return json_content['feedback'][0]['question_3']


# agent_indexes = \
#     [[i, j] for j in range(8, 14) for i in [2, 4, 5, 6]] + \
#     [[3, 8], [3, 11], [3, 12]] + \
#     [[7, i] for i in range(1, 14)] + \
    

# def get_flattened_reduced(im):
#     assert im.shape == (12, 14, 4)
#     # agent layer
    

def get_sequence(json_content, reduce=True, flatten=False, pad=100):
    reduced_format_file = json_content["reduced_format_file"]
    sequence = np.load(reduced_format_file)
    if reduce:
        sequence = sequence[:, :, 1:-1, :-1]
    if flatten:
        n_images, image_i, image_j, image_channels = sequence.shape
        sequence = sequence.reshape(n_images, image_i*image_j*image_channels)
        return sequence


def load_example(ep_name, reduce=True, flatten=True):
    json_file = f'{SAVE_FOLDER}{ep_name}/{ep_name}.json'
    with open(json_file) as f:
        json_content = json.load(f)
    judgement = get_judgement(json_content)
    sequence = get_sequence(json_content, reduce=reduce, flatten=flatten)
    return judgement, sequence






# Define Convolutional LSTM classifier model
class ConvLSTMClassifier(nn.Module):
    def __init__(
        self,
        sequence_length = 100,
        input_size = 12,
        input_channels = 3,
        filters_mid = 16,
        filters_final = 4,
        kernel_size = 3,
        lstm_hidden_size = 64,
        lstm_n_layers = 2
    ):
        super(ConvLSTMClassifier, self).__init__()

        self.scaled_input_size = input_size//4
        self.sequence_length = sequence_length
        self.filters_final = filters_final
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_n_layers = lstm_n_layers

        self.relu = self.nn.ReLU()

        self.conv1 = nn.Conv2d(
            input_channels,
            filters_mid,
            kernel_size,
            stride=1,
            padding=1
        )
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2), # (6, 6)
        self.conv2 = nn.Conv2d(
            filters_mid,
            filters_final,
            kernel_size,
            stride=1,
            padding=1
        )
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2), # (3,3)
    
        lstm_input_size = filters_final*self.scaled_input_size**2
        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, lstm_n_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()




    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = x.view(-1, self.sequence_length, self.scaled_input_size, self.scaled_input_size, self.filters_final)

        h0 = torch.zeros(self.lstm_n_layers, x.size(0), self.lstm_hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm_n_layers, x.size(0), self.lstm_hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


if __name__ == '__main__':
    SAVE_FOLDER = '../trajectory_dataset_3/'
    episodes = os.listdir(SAVE_FOLDER)

    ep_name = episodes[0]

    judgement, sequence = load_example(ep_name)

    print(sequence.shape)