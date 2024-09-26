import json
import numpy 
from matplotlib import pyplot as plt
import os


SAVE_FOLDER = '../trajectory_dataset_3/'
FIGURES_FOLDER = '../figures/'

ep_lengths = []
ep_lengths_squash = []
ep_lengths_not_squash = []
experiment_dict = {
    'alive': {
        5: [],
        10: [],
        20: []
    },
    'dead': {
        5: [],
        10: [],
        20: []
    }
}

for episode_name in os.listdir(SAVE_FOLDER):
    #reduced_format_file = f'{SAVE_FOLDER}{episode_name}/{episode_name}_reduced_format.npy'
    json_file = f'{SAVE_FOLDER}{episode_name}/{episode_name}.json'
    with open(json_file) as f:
        episode = json.load(f)
    n_images = len(episode['trajectory'])
    length = n_images - 1

    ep_lengths.append(length)

    if episode['cat_squashed']:
        ep_lengths_squash.append(length)
        if n_images in [5, 10, 20]:
            experiment_dict['dead'][n_images].append(episode['episode_no'])
    else:
        ep_lengths_not_squash.append(length)
        if n_images in [5, 10, 20]:
            experiment_dict['alive'][n_images].append(episode['episode_no'])


n_alive = len(ep_lengths_not_squash)
n_dead = len(ep_lengths_squash)

print(min(ep_lengths))
print(max(ep_lengths))




plt.title('Episode lengths, simple reward, 10,000 sample episodes')
plt.xlabel('Episode length')
plt.ylabel('Frequency')
plt.hist(ep_lengths_not_squash, bins = list(range(0, 110, 1)), alpha = 0.5, color= 'b', label=f'Cat alive ({n_alive})')
plt.hist(ep_lengths_squash, bins = list(range(0, 110, 1)), alpha = 0.5, color= 'r', label=f'Cat dead ({n_dead})')
plt.legend()
plt.savefig(FIGURES_FOLDER + 'episode_lengths_simple_policy_split.png')
plt.close()


# Choose samples for experiment
print(json.dumps(experiment_dict, indent=4))