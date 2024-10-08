import gymnasium as gym
import homegrid
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
import torch 
from torch import nn
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
import time

model = PPO.load("../training_runs/with_llm_reward_third_attempt/models/homegrid-cat-1-1.1e7_steps")

env = make_vec_env("homegrid-cat-llm-reward", n_envs=4)

obs = env.reset()
while True:
    time.sleep(0.5)
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render("human")