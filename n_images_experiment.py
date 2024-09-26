import json
import numpy as np
from matplotlib import pyplot as plt
import os
from LLM_judgement_request import make_jsonl_batch_file, read_batch_output_file, get_info_from_batch_element


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


json_files_list = []
req_ids_list = []
output_file = 'experiment.jsonl'
image_selection = 'all'

for state in ('alive', 'dead'):
    for n_images in [5, 10, 20]:
        print('\n')
        print(state, n_images)
        sample = list(np.random.choice(experiment_dict[state][n_images], 5))
        print(sample)
        for ep_no in sample:
            request_name = f'{state}_{n_images}_{image_selection}_episode_{ep_no}'
            req_ids_list.append(request_name)
            ep_json = f'{SAVE_FOLDER}episode_{ep_no}/episode_{ep_no}.json'
            json_files_list.append(ep_json)
            print('Req id', request_name)
            print('Json file', ep_json)


#make_jsonl_batch_file(json_files_list, output_file, req_ids_list, image_selection=image_selection)


