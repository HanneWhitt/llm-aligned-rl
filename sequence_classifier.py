import os
import json
import numpy as np


def get_judgement(json_content):
    return json_content['feedback'][0]['question_3']


def get_sequence(json_content, reduce=True):
    reduced_format_file = json_content["reduced_format_file"]
    sequence = np.load(reduced_format_file)
    if reduce:
        sequence = sequence[:, 1:-1, 1:-1, :-1]
    return sequence


def load_example(ep_name, reduce=False):
    json_file = f'{SAVE_FOLDER}{ep_name}/{ep_name}.json'
    with open(json_file) as f:
        json_content = json.load(f)
    judgement = get_judgement(json_content)
    cat_squashed = json_content['cat_squashed']
    #sequence = get_sequence(json_content, reduce=reduce)
    return judgement, cat_squashed



if __name__ == '__main__':
    SAVE_FOLDER = '../trajectory_dataset_3/'
    episodes = os.listdir(SAVE_FOLDER)

    ep_name = episodes[0]

    