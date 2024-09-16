import gymnasium as gym
import homegrid
import time


env = gym.make("homegrid-cat")


for i in range(100000):
    s = time.time()
    obs, info = env.reset()
    print('Call ', i, time.time() - s, 'seconds')