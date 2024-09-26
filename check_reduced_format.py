import numpy as np
import os


SAVE_FOLDER = '../trajectory_dataset/'

for episode in os.listdir(SAVE_FOLDER):
    reduced_format_file = f'{SAVE_FOLDER}{episode}/{episode}_reduced_format.npy'
    a = np.load(reduced_format_file)
    assert a.shape[1:] == (12, 14, 4)

    print(f'\n\n{episode}\n\n')

    for layer in range(a.shape[-1]):
        print(f'\n\nlayer {layer}\n\n')
        for step in a:
            print(step[:, :, layer])
            input()