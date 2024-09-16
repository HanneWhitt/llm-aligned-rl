import gymnasium as gym
import homegrid
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
import torch 
from torch import nn
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO

model = PPO.load("homegrid-cat-1")

env = gym.make("homegrid-cat")
obs, info = env.reset()
while True:
    action, _states = model.predict(obs)
    action = int(action)

    print(action)

    obs, reward, terminated, truncated, info = env.step(action)
    env.render()