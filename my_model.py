from keras.models import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import *
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
import pickle
import gym
import numpy as np


def metric_net():
    state_a = Input(shape=(11,))
    state_b = Input(shape=(11,))
    goal = Input(shape=(3,))

    m_concat_0 = Concatenate(axis=-1)([state_a, state_b, goal])

    m_concat_1 = Dense(50, activation='relu')(m_concat_0)
    m_concat_2 = Dense(50, activation='relu')(m_concat_1)

    m_concat_3 = Dense(50, activation='relu')(m_concat_2)
    m_concat_4 = Dense(50, activation='relu')(m_concat_3)

    m_concat_5 = Dense(30, activation='relu')(m_concat_4)
    m_concat_6 = Dense(10, activation='relu')(m_concat_5)

    m_output_score = Dense(1, activation='sigmoid', name="m_output_score")(m_concat_6)

    model = Model(inputs=[state_a, state_b, goal], outputs=m_output_score, name='metric_net')

    return model


def forward_net():
    state = Input(shape=(11,))
    action = Input(shape=(11,))

    f_concat_0 = Concatenate(axis=-1)([state, action])

    f_concat_1 = Dense(50, activation='relu')(f_concat_0)
    f_concat_2 = Dense(50, activation='relu')(f_concat_1)
    f_concat_3 = Dense(50, activation='relu')(f_concat_2)
    f_concat_4 = Dense(50, activation='relu')(f_concat_3)
    f_output_state = Dense(11, name="f_output_state")(f_concat_4)

    model = Model(inputs=[state, action], outputs=f_output_state, name='forward_net')

    return model


def our_model():
    state = Input(shape=(200, 11))
    goal = Input(shape=(200, 3))

    # =========== backward_model ===========
    b_concat_0 = Concatenate(axis=-1)([state, goal])

    b_concat_1 = Dense(50, activation='relu', name="b_concat_1")(b_concat_0)
    b_concat_2 = Dense(50, activation='relu', name="b_concat_2")(b_concat_1)

    b_lstm_1 = LSTM(100, return_sequences=True, return_state=False, stateful=False, name="b_lstm_1")(b_concat_2)

    b_concat_3 = Dense(50, activation='relu', name="b_concat_3")(b_lstm_1)
    b_concat_4 = Dense(50, activation='relu', name="b_concat_4")(b_concat_3)

    b_output_x = Dense(3, activation='softmax', name="x")(b_concat_4)
    b_output_y = Dense(3, activation='softmax', name="y")(b_concat_4)
    b_output_z = Dense(3, activation='softmax', name="z")(b_concat_4)
    b_output_hand = Dense(2, activation='softmax', name="hand")(b_concat_4)

    action_estimate = Concatenate(axis=-1)([b_output_x, b_output_y, b_output_z, b_output_hand])

    # =========== forward_model ===========
    next_state_estimate = forward_net()([state, action_estimate])

    # =========== metrics_model ===========
    score = metric_net()([state, next_state_estimate, goal])

    print(score.shape)


our_model()



