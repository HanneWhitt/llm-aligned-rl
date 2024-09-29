import json
import numpy 
from matplotlib import pyplot as plt
import os


SAVE_FOLDER = '../trajectory_dataset_3/'
FIGURES_FOLDER = '../figures/'


ep_lengths = []
ep_lengths_squash = []
ep_lengths_not_squash = []


alive_llm_ok = 0
alive_llm_not_ok = 0
dead_llm_ok = 0
dead_llm_not_ok = 0



episode_names_list = [f'episode_{i}' for i in range(10000)]

for episode_name in episode_names_list:
    #reduced_format_file = f'{SAVE_FOLDER}{episode_name}/{episode_name}_reduced_format.npy'
    json_file = f'{SAVE_FOLDER}{episode_name}/{episode_name}.json'
    with open(json_file) as f:
        episode = json.load(f)
    n_images = len(episode['trajectory'])
    length = n_images - 1

    ep_lengths.append(length)

    judgement = episode['feedback'][0]['question_3']

    # print(type(judgement))
    # input()

    # CAT DEAD 
    if episode['cat_squashed']:
        ep_lengths_squash.append(length)

        # LLM OK 
        if judgement:
            dead_llm_ok += 1

        # LLM NOT OK
        else:
            dead_llm_not_ok += 1

    # CAT ALIVE
    else:

        # LLM OK
        if judgement:
            alive_llm_ok += 1

        # LLM NOT OK 
        else:
            alive_llm_not_ok += 1
            print(episode_name)
            print('Ep lenght', length)

        ep_lengths_not_squash.append(length)



n_alive = len(ep_lengths_not_squash)
n_dead = len(ep_lengths_squash)

# print(min(ep_lengths))
# print(max(ep_lengths))




# print(alive_llm_ok)
# print(alive_llm_not_ok)
# print(dead_llm_ok)
# print(dead_llm_not_ok)


# plt.title('Episode lengths, simple reward, 10,000 sample episodes')
# plt.xlabel('Episode length')
# plt.ylabel('Frequency')
# plt.hist(ep_lengths_not_squash, bins = list(range(0, 110, 1)), alpha = 0.5, color= 'b', label=f'Cat alive ({n_alive})')
# plt.hist(ep_lengths_squash, bins = list(range(0, 110, 1)), alpha = 0.5, color= 'r', label=f'Cat dead ({n_dead})')
# plt.legend()
# plt.savefig(FIGURES_FOLDER + 'episode_lengths_simple_policy_split.png')
# plt.close()


# Choose samples for experiment
