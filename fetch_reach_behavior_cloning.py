from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Average
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import *
from keras.layers.core import Dropout
from keras.utils.vis_utils import plot_model
import pickle
import gym
import numpy as np


def behavior_cloning_0():

    state = Input(shape=(10,))
    goal = Input(shape=(3,))

    fc1_s = Dense(50, activation='relu')(state)
    fc2_s = Dense(50, activation='relu')(fc1_s)
    fc3_s = Dense(50, activation='relu')(fc2_s)

    fc1_g = Dense(50, activation='relu')(goal)
    fc2_g = Dense(50, activation='relu')(fc1_g)
    fc3_g = Dense(50, activation='relu')(fc2_g)

    concat_0 = Concatenate(axis=-1)([fc3_s, fc3_g])

    concat_1 = Dense(50, activation='relu')(concat_0)
    concat_2 = Dense(50, activation='relu')(concat_1)
    concat_3 = Dense(50, activation='relu')(concat_2)
    concat_4 = Dense(50, activation='relu')(concat_3)
    concat_5 = Dense(50, activation='relu')(concat_4)
    concat_6 = Dense(50, activation='relu')(concat_5)
    output = Dense(20)(concat_6)

    model = Model(inputs=[state, goal], outputs=output, name='behavior_cloning')

    return model


def behavior_cloning_1():

    state = Input(shape=(10,))
    goal = Input(shape=(3,))

    concat_0 = Concatenate(axis=-1)([state, goal])

    concat_1 = Dense(50, activation='relu')(concat_0)
    concat_2 = Dense(50, activation='relu')(concat_1)
    concat_3 = Dense(50, activation='relu')(concat_2)
    concat_4 = Dense(50, activation='relu')(concat_3)
    drop_1 = Dropout(0.5)(concat_4)
    concat_5 = Dense(50, activation='relu')(drop_1)
    concat_6 = Dense(50, activation='relu')(concat_5)
    concat_7 = Dense(50, activation='relu')(concat_6)
    concat_8 = Dense(50, activation='relu')(concat_7)
    drop_2 = Dropout(0.5)(concat_8)
    concat_9 = Dense(50, activation='relu')(drop_2)

    output = Dense(3)(concat_9)

    model = Model(inputs=[state, goal], outputs=output, name='behavior_cloning')

    return model


def train(model):

    # (state,action,next_state,reward,done)
    #    17     6          17      1    1
    data = pickle.load(open('FetchReach-v0.p', 'rb'))

    state_feed = data[:, 0:10]
    action_feed = data[:, 10:13]
    goal_feed = data[:, 13:16]

    model.compile(optimizer=Adam(lr=1e-4),
                  loss='mean_squared_error',
                  # metrics=['mse']
                  )

    tf_board = TensorBoard(log_dir='./logs',
                           histogram_freq=0,
                           write_graph=True,
                           write_images=False,
                           embeddings_freq=0,
                           embeddings_layer_names=None,
                           embeddings_metadata=None)

    early_stop = EarlyStopping(monitor='val_loss',
                               patience=2,
                               verbose=0,
                               mode='auto')

    model_checkpoint = ModelCheckpoint('one-step-backward.{epoch:02d}-{val_loss:.2f}.hdf5',
                                       monitor='val_loss',                    # here 'val_loss' and 'loss' are the same
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True)

    model.fit([state_feed, goal_feed],
              action_feed,
              batch_size=50,
              epochs=100,
              verbose=1,
              validation_split=0.2,
              shuffle=True,
              callbacks=[tf_board, model_checkpoint])


def test(model_for_25_nets):
    env = gym.make('FetchReach-v0')

    model_for_25_nets.compile(optimizer=Adam(lr=1e-4),
                              loss='mean_squared_error',
                              metrics=['mse'])

    model_for_25_nets.load_weights('one-step-backward.60-0.06.hdf5', by_name=True)

    while True:

        two_state = np.zeros((2, 10))
        two_goal = np.zeros((2, 3))
        observation = env.reset()
        two_state[0, :] = observation["observation"]
        two_goal[0, :] = observation["desired_goal"]

        action = np.zeros((4,))
        done = False

        while not done:

            env.render()
            action_two = model_for_25_nets.predict_on_batch([two_state, two_goal])
            action[0:3] = action_two[0, :]

            observation, reward, done, info = env.step(action)
            two_state[0, :] = observation["observation"]
            two_goal[0, :] = observation["desired_goal"]

            if done:
                print(True)


if __name__ == '__main__':

    model = behavior_cloning_1()
    # train(model)
    test(model)
