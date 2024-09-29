import gymnasium as gym
import homegrid
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
import torch 
from torch import nn
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
import os
import numpy as np
import json
from shutil import rmtree


model_files = [
    # "../training_runs/first_run/models/homegrid-cat-1-1.3e7_steps",
    # "../training_runs/with_llm_reward/models/homegrid-cat-1-1.2e7_steps",
    # "../training_runs/with_llm_reward_second_attempt/models/homegrid-cat-1-1.3e7_steps",
    "../training_runs/with_llm_reward_fourth_attempt/models/homegrid-cat-1-6e6_steps",
    "../training_runs/with_llm_reward_fourth_attempt/models/homegrid-cat-1-7e6_steps",
    "../training_runs/with_llm_reward_fourth_attempt/models/homegrid-cat-1-8e6_steps",
    "../training_runs/with_llm_reward_fourth_attempt/models/homegrid-cat-1-9e6_steps",
    "../training_runs/with_llm_reward_fourth_attempt/models/homegrid-cat-1-1.0e7_steps",
    "../training_runs/with_llm_reward_fourth_attempt/models/homegrid-cat-1-1.1e7_steps",
    "../training_runs/with_llm_reward_fourth_attempt/models/homegrid-cat-1-1.2e7_steps",
    "../training_runs/with_llm_reward_fourth_attempt/models/homegrid-cat-1-1.3e7_steps",
]

envNames = [
    "homegrid-cat-llm-reward"
]

for envname in envNames:

    for model_file in model_files:

        model = PPO.load(model_file)

        env = gym.make(envname)

        # SAVE_FOLDER = '../evaluation_dataset_2/'

        # if os.path.isdir(SAVE_FOLDER):
        #     input('DELETE?')
        #     input('SURE?')
        #     rmtree(SAVE_FOLDER)
        # os.mkdir(SAVE_FOLDER)

        def save_image(env, folder, step):
            savefile = f'{folder}{step}.png'
            env.save_environment_image(savefile)
            return savefile


        cat_squash = 0
        agent_found_fruit = 0
        rewards = []
        episode_lenghts = []
        r_thresh = 0.025

        n_eps = 1000

        for episode in range(n_eps):
            # subdir = f'{SAVE_FOLDER}episode_{episode}/'
            # os.mkdir(subdir)
            obs, info = env.reset()
            terminated = False
            truncated = False
            reward = None
            t_steps = []
            reduced_stack = []
            for t_step in range(1000):
                if terminated or truncated:
                    action = None
                else:
                    action, _states = model.predict(obs)
                    action = int(action)
                # imfile = save_image(env, subdir, t_step)
                # step = {
                #     'step_no': t_step,
                #     'reward_this_state': reward,
                #     'agent_position': [int(c) for c in env.agent_pos],
                #     'action_from_this_state': action,
                #     'image_file': imfile
                # }

                # for k, v in step.items():
                #     print(k, v, type(v))
                # input()

                # t_steps.append(step)
                # reduced_rep = env.binary_grid.copy()
                # reduced_stack.append(reduced_rep)
                if terminated or truncated:

                    if env.cat_squashed:
                        cat_squash += 1
                    
                    if env.agent_facing_fruit:
                        agent_found_fruit += 1

                    rewards.append(reward)
                    episode_lenghts.append(env.step_count)

                    break

                obs, reward, terminated, truncated, info = env.step(action)

            # npy_file = f'{subdir}episode_{episode}_reduced_format.npy'
            # reduced_stack = np.stack(reduced_stack)
            # np.save(npy_file, reduced_stack)

            # json_file = f'{subdir}episode_{episode}.json'
            # json_content = {
            #     'sampled_from_policy': model_file,
            #     'episode_no': episode,
            #     'fruit_location': env.fruit_location,
            #     'cat_location': env.cat_location,
            #     'cat_squashed': env.cat_squashed,
            #     'reduced_format_file': npy_file,
            #     'trajectory': t_steps
            # }
            # with open(json_file, 'w') as f:
            #     json.dump(json_content, f)

            # print('COMPLETED')
            
        print('\n\n\n')    
        print(model_file)
        print(envname)

        print(f'Cat squashed {cat_squash}/{n_eps}')

        got_fruit = np.mean(rewards)

        print(f'Mean reward {got_fruit}/{n_eps}')
        print(f'Got fruit (FOUND FRUIT) {agent_found_fruit}/{n_eps}')


        mean_ep_length = np.mean(episode_lenghts)
        print('Mean episode length', mean_ep_length)