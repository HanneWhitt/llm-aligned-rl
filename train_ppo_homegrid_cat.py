import gymnasium as gym
import homegrid
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


# Parallel environments
# env = make_vec_env("homegrid-cat", n_envs=4)

env = gym.make("homegrid-cat")

print('Defining model')
model = PPO("MlpPolicy", env, verbose=2)
print('Model defined\n')

print('\nStarting learning...')
model.learn(total_timesteps=25000, log_interval=1000, progress_bar=True)


model.save("homegrid_cat_1")

del model # remove to demonstrate saving and loading

model = PPO.load("homegrid_cat_1")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render("human")