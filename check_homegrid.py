from stable_baselines3.common.env_checker import check_env
from matplotlib import pyplot as plt

import gymnasium as gym

import homegrid

# from homegrid.window import Window
# import matplotlib.pyplot as plt
# from tokenizers import Tokenizer


env = gym.make("homegrid-cat")
obs, info = env.reset()

for layer in range(obs.shape[-1]):
    print(obs[:, :, layer])

print(obs.shape)
print(env.agent_pos)

av = env.agent_view_binary_grid()

for layer in range(obs.shape[-1]):
    print(av[:, :, layer])

rend = env.render()
# input('\n.render() output \n\n')
plt.imshow(rend)
plt.show()

exit()


# check_env(env)
# print("CHECK DONE")


input('GENERAL CHEKS')


print(obs)
print(obs.shape)
input("\n.reset() obs output\n\n")
print(info)
input("\n.reset() info output\n\n")



print('STEP')
obs, reward, terminated, truncated, info = env.step(0)
print('step over')

print(obs)
print(obs.shape)
input("\n.step() obs output\n\n")
print(reward)
input("\n.step() reward output\n\n")
print(terminated)
input("\n.step() terminated output\n\n")
print(truncated)
input("\n.step() truncated output\n\n")
print(info)
input("\n.step() info output\n\n")




print(env.action_space)
print('env.action_space')
input()



print(env.observation_space)
print('env.observation_space')
input()



print(env.metadata)
print('env.metadata')
input()


print(env.render_mode)
print('env.render_mode')
input()

print(env.reward_range)
print('env.reward_range')
input()

print(env.spec)
print('env.spec')


rend = env.render()
print(rend)
input('\n.render() output \n\n')
plt.imshow(rend)
plt.show()
