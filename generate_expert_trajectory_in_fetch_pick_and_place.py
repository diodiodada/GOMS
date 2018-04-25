import pickle
import gym
import numpy as np


def policy(observation):
    global stage
    global open_counter
    global close_counter
    action = np.zeros((4,))
    if stage == "reach_object":
        # move towards the object
        # if distance > 0.03 of distance < -0.03, using 1/-1
        # else using distance exactly
        action_3 = observation["observation"][6:9]
        action_3[2] = action_3[2] + 0.07

        if action_3[0] < 0.001 and action_3[1] < 0.001 and action_3[2] < 0.001 and action_3[0] > -0.001 and action_3[1] > -0.001 and action_3[2] > -0.001:
            # print("reach the target !!")
            stage = stage_set[1]
        else:
            for i in range(3):
                if action_3[i] > 0.03:
                    action_3[i] = 1
                elif action_3[i] < -0.03:
                    action_3[i] = -1
                else:
                    action_3[i] = action_3[i] / 0.03
            action[0:3] = action_3

    elif stage == "open":
        # open the claw !!
        if open_counter < 3:
            action = [0.0, 0.0, 0.0, 1]
            open_counter = open_counter + 1
        else:
            # print("open the claw !!")
            stage = stage_set[2]
            open_counter = 0

    elif stage == "go_down":

        action_3 = observation["observation"][6:9]
        action_3[2] = action_3[2] + 0.0

        if action_3[0] < 0.001 and action_3[1] < 0.001 and action_3[2] < 0.001 and action_3[0] > -0.001 and action_3[
            1] > -0.001 and action_3[2] > -0.001:
            # print("go down already !!")
            stage = stage_set[3]
        else:
            for i in range(3):
                if action_3[i] > 0.03:
                    action_3[i] = 1
                elif action_3[i] < -0.03:
                    action_3[i] = -1
                else:
                    action_3[i] = action_3[i] / 0.03
            action[0:3] = action_3

    elif stage == "close":
        # close the claw !!
        if close_counter < 3:
            action = [0.0, 0.0, 0.0, -1.0]
            close_counter = close_counter + 1
        else:
            # print("close the claw !!")
            stage = stage_set[4]
            close_counter = 0

    elif stage == "reach":

        desired_goal = observation["desired_goal"]
        achieved_goal = observation["achieved_goal"]
        action_3 = desired_goal - achieved_goal

        if action_3[0] < 0.001 and action_3[1] < 0.001 and action_3[2] < 0.001 and action_3[0] > -0.001 and action_3[
            1] > -0.001 and action_3[2] > -0.001:
            # print("reach already !!")
            pass
        else:
            for i in range(3):
                if action_3[i] > 0.03:
                    action_3[i] = 1
                elif action_3[i] < -0.03:
                    action_3[i] = -1
                else:
                    action_3[i] = action_3[i] / 0.03
            action[0:3] = action_3
        action[3] = -1

    return action


env = gym.make('FetchPickAndPlace-v0')

stage_set = ["reach_object", "open", "go_down", "close", "reach"]
open_counter = 0
close_counter = 0

data = []

for trajectory_num in range(5000):
    stage = stage_set[0]

    observation = env.reset()
    done = False
    # print(observation)

    while not done:
        # env.render()

        action = policy(observation)

        previous_observation = observation
        observation, reward, done, info = env.step(action)

        record = []
        record.extend(previous_observation["observation"])
        record.extend(action)
        record.extend(observation["observation"])
        record.extend(observation["desired_goal"])
        record.extend([float(done)])

        data.append(record)

    print(trajectory_num)

data = np.array(data)
pickle.dump(data, open("FetchPickAndPlace-v0.p", "wb"))


