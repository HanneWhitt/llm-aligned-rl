import gym
from matplotlib import pyplot as plt
import homegrid

# from homegrid.window import Window
# import matplotlib.pyplot as plt
# from tokenizers import Tokenizer


env = gym.make("homegrid-cat")
obs, info = env.reset()