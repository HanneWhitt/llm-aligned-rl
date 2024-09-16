import gymnasium as gym
import homegrid

env = gym.make("CartPole-v1")

obs, info = env.reset()


print(obs)
print(type(obs))

print(env.observation_space)



env = gym.make("homegrid-cat")

obs, info = env.reset()


print(obs)
print(type(obs))

print(env.observation_space)
