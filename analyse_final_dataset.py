import json
import numpy as np
from matplotlib import pyplot as plt
import os


FIGURES_FOLDER = '../figures/'


save_folders = [
    '../trajectory_dataset_3/',
    #'../third_attempt_analysis_set/',
    '../final_analysis_set/'
]

for SAVE_FOLDER in save_folders:

    ep_lengths = []
    ep_lengths_squash = []
    ep_lengths_not_squash = []

    found_fruit_total = 0


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

        #judgement = episode['feedback'][0]['question_3']

        found_fruit = episode['trajectory'][-1]['reward_this_state'] > 0.1
        if found_fruit:
            found_fruit_total += 1
        
        # print(type(judgement))
        # input()

        # CAT DEAD 
        if episode['cat_squashed']:
            ep_lengths_squash.append(length)

            # # LLM OK 
            # if judgement:
            #     dead_llm_ok += 1

            # # LLM NOT OK
            # else:
            #     dead_llm_not_ok += 1

        # CAT ALIVE
        else:

            # LLM OK
            # if judgement:
            #     alive_llm_ok += 1

            # # LLM NOT OK 
            # else:
            #     alive_llm_not_ok += 1
            #     print(episode_name)
            #     print('Ep lenght', length)

            ep_lengths_not_squash.append(length)



    n_alive = len(ep_lengths_not_squash)
    n_dead = len(ep_lengths_squash)

    print(SAVE_FOLDER)
    print('cat dead ', n_dead)
    print('cat alive ', n_alive)
    print('Found fruit', found_fruit_total)
    print('Mean ep legnth', np.mean(ep_lengths))


    # print(min(ep_lengths))
    # print(max(ep_lengths))




    # print(alive_llm_ok)
    # print(alive_llm_not_ok)
    # print(dead_llm_ok)
    # print(dead_llm_not_ok)

    if '_3' in SAVE_FOLDER:
        plt.title('Episode lengths, naive policy')
        plt.xlabel('Steps')
        plt.ylabel('Frequency')
        plt.hist(ep_lengths_not_squash, bins = list(range(0, 110, 1)), alpha = 0.5, color= 'b', label=f'Cat alive ({n_alive})')
        plt.hist(ep_lengths_squash, bins = list(range(0, 110, 1)), alpha = 0.5, color= 'r', label=f'Cat dead ({n_dead})')
        plt.legend()
        plt.savefig(FIGURES_FOLDER + 'episode_lengths_naive_policy.png')
        plt.close()

    else:
        plt.title('Episode lengths, policy with LLM feedback')
        plt.xlabel('Steps')
        plt.ylabel('Frequency')
        plt.hist(ep_lengths_not_squash, bins = list(range(0, 110, 1)), alpha = 0.5, color= 'b', label=f'Cat alive ({n_alive})')
        plt.hist(ep_lengths_squash, bins = list(range(0, 110, 1)), alpha = 0.5, color= 'r', label=f'Cat dead ({n_dead})')
        plt.legend()
        plt.savefig(FIGURES_FOLDER + 'episode_lengths_LLM_feedback_policy.png')
        plt.close()


    # Choose samples for experiment
