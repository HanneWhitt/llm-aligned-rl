import gym
from matplotlib import pyplot as plt
import homegrid

# from homegrid.window import Window
# import matplotlib.pyplot as plt
# from tokenizers import Tokenizer


env = gym.make("homegrid-cat")
obs, info = env.reset()


print(obs.shape)
print(env.binary_grid.shape)
input()


for layer in range(0, 4):
    print(env.binary_grid[:,:,layer])
    input()


rend = env.render()
print(rend)
input('\n.render() output \n\n')
plt.imshow(rend)
plt.show()




print(env.binary_grid[:,:,0])





input(0)
#position_funcs = [env.agent_pos, env.cat_location, env.fruit_location, env.overlap_check]
# bg = env.create_binary_grid(position_funcs)


env.step(1)
print(env.binary_grid[:,:,0])

print(obs.shape)
print(env.binary_grid.shape)

input(1)

env.step(2)
print(env.binary_grid[:,:,0])
input(2)

env.step(1)
print(env.binary_grid[:,:,0])
input(3)

env.step(2)
print(env.binary_grid[:,:,0])
input(4)



# grid, vis_mask = env.gen_obs_grid()

# print(grid)

# print(grid.grid)

# print(vis_mask)