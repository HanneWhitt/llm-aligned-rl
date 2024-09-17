import gymnasium as gym
import homegrid
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import torch 
from torch import nn
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback




class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
    

policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

#env = gym.make("homegrid-cat")
env = make_vec_env("homegrid-cat", n_envs=4)

model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log="./tensorboard_log_2/")


for n in range(1, 11):
    sts = f"{n}e5_steps" if n < 10 else '1e6_steps'
    print(sts)
    model.learn(1e5, progress_bar=True, tb_log_name='continuous_log', reset_num_timesteps=False)
    model.save(f"homegrid-cat-1-{sts}")

for n in range(2, 10000):
    print(sts)
    sts = f"{n}e6_steps" if n < 10 else f'{n/10}e7_steps'
    model.learn(1e6, progress_bar=True, tb_log_name='continuous_log', reset_num_timesteps=False)
    model.save(f"homegrid-cat-1-{sts}")


del model # remove to demonstrate saving and loading

model = PPO.load("homegrid-cat-1")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render("human")