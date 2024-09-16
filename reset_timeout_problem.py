import gymnasium as gym
import homegrid
import time
import random


env = gym.make("homegrid-cat")


for i in range(1000):
    s = time.time()
    obs, info = env.reset()
    for st in range(100):
        a = random.choice([0, 1, 2, 3])
        obs, reward, terminated, truncated, info = env.step(a)
        if terminated or truncated:
            break
    print('Run ', i, time.time() - s, 'seconds')
