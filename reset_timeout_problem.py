import gymnasium as gym
import homegrid
import time


env = gym.make("homegrid-cat")


for i in range(100):
    print('Call ', i)
    s = time.time()
    obs, info = env.reset()
    print(time.time() - s, 'seconds')