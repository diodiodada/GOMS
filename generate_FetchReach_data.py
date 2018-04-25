
import pickle
import gym
import numpy as np
env = gym.make('FetchReach-v0')

# state, action, next_state
#    10,      4,         10
data = np.zeros((50000, 16))


observation = env.reset()
state = observation["observation"]

for i in range(50000):
    # env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    goal = observation["achieved_goal"]
    next_state = observation["observation"]

    data[i, 0:10] = state
    data[i, 10:13] = action[0:3]
    data[i, 13:16] = goal

    state = next_state

pickle.dump(data, open("FetchReach-v0.p", "wb"))
