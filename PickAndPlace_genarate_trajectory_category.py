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

        min = -0.015
        max = 0.015

        if action_3[0] < max and action_3[1] < max and action_3[2] < max and action_3[0] > min and action_3[1] > min and action_3[2] > min:
            # print("reach the target !!")
            stage = stage_set[1]
        else:
            for i in range(3):
                if action_3[i] > max:
                    action_3[i] = 0.5
                elif action_3[i] < min:
                    action_3[i] = -0.5
                else:
                    action_3[i] = 0.0
            action[0:3] = action_3
        action[3] = 1.0

    elif stage == "go_down":

        action_3 = observation["observation"][6:9]
        action_3[2] = action_3[2] + 0.0

        min = -0.015
        max = 0.015

        if action_3[0] < max and action_3[1] < max and action_3[2] < max and action_3[0] > min and action_3[
            1] > min and action_3[2] > min:
            # print("go down already !!")
            stage = stage_set[2]
        else:
            for i in range(3):
                if action_3[i] > max:
                    action_3[i] = 0.5
                elif action_3[i] < min:
                    action_3[i] = -0.5
                else:
                    action_3[i] = 0.0
            action[0:3] = action_3
        action[3] = 1.0

    elif stage == "close":
        # close the claw !!
        if close_counter < 3:
            action = [0.0, 0.0, 0.0, -1.0]
            close_counter = close_counter + 1
        else:
            # print("close the claw !!")
            stage = stage_set[3]
            close_counter = 0
        action[3] = -1.0

    elif stage == "reach":

        desired_goal = observation["desired_goal"]
        achieved_goal = observation["achieved_goal"]
        action_3 = desired_goal - achieved_goal

        min = -0.015
        max = 0.015

        if action_3[0] < max and action_3[1] < max and action_3[2] < max and action_3[0] > min and action_3[
            1] > min and action_3[2] > min:
            # print("reach already !!")
            pass
        else:
            for i in range(3):
                if action_3[i] > max:
                    action_3[i] = 0.5
                elif action_3[i] < min:
                    action_3[i] = -0.5
                else:
                    action_3[i] = 0
            action[0:3] = action_3
        action[3] = -1.0

    return action


env = gym.make('FetchPickAndPlace-v0')

stage_set = ["reach_object", "go_down", "close", "reach"]
close_counter = 0

data = []

for trajectory_num in range(50000):
    stage = stage_set[0]

    observation = env.reset()
    done = False
    # print(observation)

    while not done:
        env.render()

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

# data = np.array(data)
# pickle.dump(data, open("FetchPickAndPlace-50000.p", "wb"))


