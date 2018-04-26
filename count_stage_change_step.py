import pickle
import gym
import numpy as np


def policy(observation):
    global stage
    global open_counter
    global close_counter
    action = np.zeros((4,))
    stage_change = "not-change"

    if stage == "reach_object":
        # move towards the object
        # if distance > 0.03 of distance < -0.03, using 1/-1
        # else using distance exactly
        action_3 = observation["observation"][6:9]
        action_3[2] = action_3[2] + 0.07

        if action_3[0] < 0.001 and action_3[1] < 0.001 and action_3[2] < 0.001 and action_3[0] > -0.001 and action_3[1] > -0.001 and action_3[2] > -0.001:
            # print("reach the target !!")
            stage = stage_set[1]
            stage_change = "from-reach-to-open"
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
            stage_change = "from-open-to-go-down"

    elif stage == "go_down":

        action_3 = observation["observation"][6:9]
        action_3[2] = action_3[2] + 0.0

        if action_3[0] < 0.001 and action_3[1] < 0.001 and action_3[2] < 0.001 and action_3[0] > -0.001 and action_3[
            1] > -0.001 and action_3[2] > -0.001:
            # print("go down already !!")
            stage = stage_set[3]
            stage_change = "from-go-down-to-close"
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
            stage_change = "from-close-to-reach"

    elif stage == "reach":

        desired_goal = observation["desired_goal"]
        achieved_goal = observation["achieved_goal"]
        action_3 = desired_goal - achieved_goal

        min = -0.001
        max = 0.001
        if action_3[0] < max and action_3[1] < max and action_3[2] < max and action_3[0] > min and action_3[
            1] > min and action_3[2] > min:
            # print("reach already !!")
            stage = stage_set[5]
            stage_change = "already-to-goal"
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

    elif stage == "hold":

        action[3] = -1

    return action, stage_change


env = gym.make('FetchPickAndPlace-v0')

stage_set = ["reach_object", "open", "go_down", "close", "reach", "hold"]
open_counter = 0
close_counter = 0

from_reach_to_open = []
from_open_to_go_down = []
from_go_down_to_close = []
from_close_to_reach = []
already_to_goal = []

for trajectory_num in range(1000):
    stage = stage_set[0]

    observation = env.reset()
    done = False

    step = 1
    while not done:
        # env.render()

        action, stage_change = policy(observation)

        if stage_change == "from-reach-to-open":
            from_reach_to_open.append(step)
        elif stage_change == "from-open-to-go-down":
            from_open_to_go_down.append(step)
        elif stage_change == "from-go-down-to-close":
            from_go_down_to_close.append(step)
        elif stage_change == "from-close-to-reach":
            from_close_to_reach.append(step)
        elif stage_change == "already-to-goal":
            already_to_goal.append(step)

        # print(stage_change)

        observation, reward, done, info = env.step(action)

        step = step + 1

    print(trajectory_num)

print("==========================")


from_reach_to_open = np.array(from_reach_to_open)
from_open_to_go_down = np.array(from_open_to_go_down)
from_go_down_to_close = np.array(from_go_down_to_close)
from_close_to_reach = np.array(from_close_to_reach)
already_to_goal = np.array(already_to_goal)

print(from_reach_to_open.mean(), end=" ")
print(from_reach_to_open.std(), end=" ")
print(from_reach_to_open.min(), end=" ")
print(from_reach_to_open.max())

print(from_open_to_go_down.mean(), end=" ")
print(from_open_to_go_down.std(), end=" ")
print(from_open_to_go_down.min(), end=" ")
print(from_open_to_go_down.max())

print(from_go_down_to_close.mean(), end=" ")
print(from_go_down_to_close.std(), end=" ")
print(from_go_down_to_close.min(), end=" ")
print(from_go_down_to_close.max())

print(from_close_to_reach.mean(), end=" ")
print(from_close_to_reach.std(), end=" ")
print(from_close_to_reach.min(), end=" ")
print(from_close_to_reach.max())

print(already_to_goal.mean(), end=" ")
print(already_to_goal.std(), end=" ")
print(already_to_goal.min(), end=" ")
print(already_to_goal.max())

