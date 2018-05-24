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

    bn = BatchNormalization()(m_concat_4)

    m_concat_5 = Dense(30, activation='relu')(bn)
    m_concat_6 = Dense(10, activation='relu')(m_concat_5)

    m_output_score = Dense(1, activation='sigmoid', name="m_output_score")(m_concat_6)
    # m_output_score = Dense(1, name="m_output_score")(m_concat_6)

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
    state = Input(shape=(None, 11))
    goal = Input(shape=(None, 3))

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
    forward = forward_net()
    forward.trainable = False
    next_state_estimate = forward([state, action_estimate])

    # =========== metrics_model ===========
    metrics = metric_net()
    metrics.trainable = False
    score = metrics([state, next_state_estimate, goal])

    model = Model(inputs=[state, goal], outputs=[action_estimate, next_state_estimate, score], name='our_model')
    return model


def train_forward_net(model):
    data = pickle.load(open('Pick-Place-Push-category-4-paths-1000.p', 'rb'))

    # get state feed
    i = np.array([0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16])
    state_feed = data[:, i]

    # get next state feed
    j = i + 36
    next_state_feed = data[:, j]

    # get action feed
    action_feed = data[:, 32:36]
    action_x_feed = action_feed[:, 0]
    action_y_feed = action_feed[:, 1]
    action_z_feed = action_feed[:, 2]
    action_hand_feed = action_feed[:, 3]

    action_x_feed = to_categorical(action_x_feed, 3)
    action_y_feed = to_categorical(action_y_feed, 3)
    action_z_feed = to_categorical(action_z_feed, 3)
    action_hand_feed = to_categorical(action_hand_feed, 2)

    action_feed = np.append(action_x_feed, action_y_feed, axis=-1)
    action_feed = np.append(action_feed, action_z_feed, axis=-1)
    action_feed = np.append(action_feed, action_hand_feed, axis=-1)

    model.compile(optimizer=Adam(lr=1e-4),
                  loss='mse',
                  )

    tf_board = TensorBoard(log_dir='./logs',
                           histogram_freq=0,
                           write_graph=True,
                           write_images=False,
                           embeddings_freq=0,
                           embeddings_layer_names=None,
                           embeddings_metadata=None)

    model_checkpoint = ModelCheckpoint('F.{epoch:d}-{val_loss:.6f}.hdf5',
                                       monitor='val_loss',  # here 'val_loss' and 'loss' are the same
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True)

    model.fit([state_feed, action_feed],
              next_state_feed,
              batch_size=50,
              # initial_epoch=201,
              epochs=1000,
              verbose=1,
              validation_split=0.2,
              shuffle=True,
              callbacks=[tf_board, model_checkpoint])


def train_metrics_net(model):
    data = pickle.load(open('Pick-Place-Push-category-4-paths-1000.p', 'rb'))

    i = np.array([0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16])
    state_feed = data[:, i]

    j = i + 36
    next_state_feed = data[:, j]

    goal_feed = data[:, 23:26]

    true_value_1_feed = np.ones((state_feed.shape[0], 1))
    true_value_0_feed = np.zeros((state_feed.shape[0], 1))

    half_1 = np.concatenate((state_feed, next_state_feed, goal_feed, true_value_1_feed), axis=1)
    half_2 = np.concatenate((next_state_feed, state_feed, goal_feed, true_value_0_feed), axis=1)

    all_data = np.concatenate((half_1, half_2), axis=0)
    np.random.shuffle(all_data)

    # feed data
    state_a_feed = all_data[:, 0:11]
    state_b_feed = all_data[:, 11:22]
    goal_feed = all_data[:, 22:25]
    score_feed = all_data[:, 25]

    # score_feed = to_categorical(score_feed, 2)

    model.compile(optimizer=Adam(lr=1e-4),
                  loss='binary_crossentropy',
                  metrics=['acc']
                  )

    tf_board = TensorBoard(log_dir='./logs',
                           histogram_freq=0,
                           write_graph=True,
                           write_images=False,
                           embeddings_freq=0,
                           embeddings_layer_names=None,
                           embeddings_metadata=None)

    model_checkpoint = ModelCheckpoint('M.{epoch:d}-{val_loss:.6f}.hdf5',
                                       monitor='val_loss',  # here 'val_loss' and 'loss' are the same
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True)

    model.fit([state_a_feed, state_b_feed, goal_feed],
              score_feed,
              batch_size=500,
              # initial_epoch=201,
              epochs=1000,
              verbose=1,
              validation_split=0.2,
              shuffle=True,
              callbacks=[tf_board, model_checkpoint])

# forward_model = forward_net()
# train_forward_net(forward_model)

# metrics_model = metric_net()
# train_metrics_net(metrics_model)


model = forward_net()
print(model.summary())




