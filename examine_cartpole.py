import gymnasium as gym

env = gym.make("CartPole-v1")

obs, info = env.reset()


print(obs)
print(type(obs))


