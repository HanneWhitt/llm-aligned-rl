import gymnasium as gym
import numpy as np
import homegrid

env = gym.make("homegrid-cat")
obs, info = env.reset()

val = env.layout.valid_poss['agent_start']


random_point = np.array([4.5, 3.8])

val = np.array(val)

diff = val - random_point

diff_sq = diff**2

sum_sq_dist = np.sum(diff_sq, axis=1)

print(val)
print(diff)
print(diff_sq)
print(sum_sq_dist)

sq_dist = np.sum((val - random_point)**2, axis=1)
print(sq_dist)
sort_idxs = sq_dist.argsort()
print(sort_idxs)

