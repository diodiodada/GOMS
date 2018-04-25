import pickle
import gym
import numpy as np

env = gym.make('FetchPickAndPlace-v0')

observation = env.reset()
print(observation)

# while True:
#     env.render()
#     action = env.action_space.sample()
#
#     observation, reward, done, info = env.step(action)
#     # print(observation['achieved_goal'])
#     # next_state = observation["observation"]



