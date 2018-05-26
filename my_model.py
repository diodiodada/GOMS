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

    m_concat_1 = Dense(50, activation='relu', name='m_concat_1')(m_concat_0)
    m_concat_2 = Dense(50, activation='relu', name='m_concat_2')(m_concat_1)

    m_concat_3 = Dense(50, activation='relu', name='m_concat_3')(m_concat_2)
    m_concat_4 = Dense(50, activation='relu', name='m_concat_4')(m_concat_3)

    m_bn = BatchNormalization(name='m_bn')(m_concat_4)

    m_concat_5 = Dense(30, activation='relu', name='m_concat_5')(m_bn)
    m_concat_6 = Dense(10, activation='relu', name='m_concat_6')(m_concat_5)

    m_output_score = Dense(1, activation='sigmoid', name="m_output_score")(m_concat_6)

    model = Model(inputs=[state_a, state_b, goal], outputs=m_output_score, name='metric_net')

    return model


def forward_net():
    state = Input(shape=(11,))
    action = Input(shape=(11,))

    f_concat_0 = Concatenate(axis=-1)([state, action])

    f_concat_1 = Dense(50, activation='relu', name='f_concat_1')(f_concat_0)
    f_concat_2 = Dense(50, activation='relu', name='f_concat_2')(f_concat_1)
    f_concat_3 = Dense(50, activation='relu', name='f_concat_3')(f_concat_2)
    f_concat_4 = Dense(50, activation='relu', name='f_concat_4')(f_concat_3)
    f_output_state = Dense(11, name="f_output_state")(f_concat_4)

    model = Model(inputs=[state, action], outputs=f_output_state, name='forward_net')

    return model


def our_model_copy():
    state = Input(shape=(200, 11))
    goal = Input(shape=(200, 3))

    # =========== backward_model ===========
    b_concat_0 = Concatenate(axis=-1)([state, goal])

    masking_1 = Masking(mask_value=0.0)(b_concat_0)

    b_concat_1 = Dense(50, activation='relu', name="b_concat_1")(masking_1)
    b_concat_2 = Dense(50, activation='relu', name="b_concat_2")(b_concat_1)

    b_lstm_1 = LSTM(100, return_sequences=True, return_state=False, stateful=False, name="b_lstm_1")(b_concat_2)

    b_concat_3 = Dense(50, activation='relu', name="b_concat_3")(b_lstm_1)
    b_concat_4 = Dense(50, activation='relu', name="b_concat_4")(b_concat_3)

    b_output_x = Dense(3, activation='softmax', name="b_output_x")(b_concat_4)
    b_output_y = Dense(3, activation='softmax', name="b_output_y")(b_concat_4)
    b_output_z = Dense(3, activation='softmax', name="b_output_z")(b_concat_4)
    b_output_hand = Dense(2, activation='softmax', name="b_output_hand")(b_concat_4)

    action_estimate = Concatenate(axis=-1)([b_output_x, b_output_y, b_output_z, b_output_hand])

    # =========== forward_model ===========
    f_concat_0 = Concatenate(axis=-1)([state, action_estimate])
    f_concat_1 = Dense(50, activation='relu', trainable=False, name='f_concat_1')(f_concat_0)
    f_concat_2 = Dense(50, activation='relu', trainable=False, name='f_concat_2')(f_concat_1)
    f_concat_3 = Dense(50, activation='relu', trainable=False, name='f_concat_3')(f_concat_2)
    f_concat_4 = Dense(50, activation='relu', trainable=False, name='f_concat_4')(f_concat_3)
    f_output_state = Dense(11, name="f_output_state", trainable=False)(f_concat_4)

    # =========== metrics_model ===========
    m_concat_0 = Concatenate(axis=-1)([state, f_output_state, goal])
    m_concat_1 = Dense(50, activation='relu', trainable=False, name='m_concat_1')(m_concat_0)
    m_concat_2 = Dense(50, activation='relu', trainable=False, name='m_concat_2')(m_concat_1)
    m_concat_3 = Dense(50, activation='relu', trainable=False, name='m_concat_3')(m_concat_2)
    m_concat_4 = Dense(50, activation='relu', trainable=False, name='m_concat_4')(m_concat_3)
    m_bn = BatchNormalization(trainable=False, name='m_bn')(m_concat_4)
    m_concat_5 = Dense(30, activation='relu', trainable=False, name='m_concat_5')(m_bn)
    m_concat_6 = Dense(10, activation='relu', trainable=False, name='m_concat_6')(m_concat_5)
    m_output_score = Dense(1, activation='sigmoid', name="m_output_score", trainable=False)(m_concat_6)

    model = Model(inputs=[state, goal],
                  outputs=[b_output_x,
                           b_output_y,
                           b_output_z,
                           b_output_hand,
                           f_output_state,
                           m_output_score],
                  name='our_model')
    return model


def our_model_copy_for_test():

    state = Input(shape=(1, 11), batch_shape=(2, 1, 11))
    goal = Input(shape=(1, 3), batch_shape=(2, 1, 3))

    # =========== backward_model ===========
    b_concat_0 = Concatenate(axis=-1)([state, goal])

    b_concat_1 = Dense(50, activation='relu', name="b_concat_1")(b_concat_0)
    b_concat_2 = Dense(50, activation='relu', name="b_concat_2")(b_concat_1)

    b_lstm_1 = LSTM(100, return_sequences=True, return_state=False, stateful=True, name="b_lstm_1")(b_concat_2)

    b_concat_3 = Dense(50, activation='relu', name="b_concat_3")(b_lstm_1)
    b_concat_4 = Dense(50, activation='relu', name="b_concat_4")(b_concat_3)

    b_output_x = Dense(3, activation='softmax', name="b_output_x")(b_concat_4)
    b_output_y = Dense(3, activation='softmax', name="b_output_y")(b_concat_4)
    b_output_z = Dense(3, activation='softmax', name="b_output_z")(b_concat_4)
    b_output_hand = Dense(2, activation='softmax', name="b_output_hand")(b_concat_4)

    model = Model(inputs=[state, goal], outputs=[b_output_x, b_output_y, b_output_z, b_output_hand])

    return model


# def our_model():
#     state = Input(shape=(None, 11))
#     goal = Input(shape=(None, 3))
#
#     # =========== backward_model ===========
#     b_concat_0 = Concatenate(axis=-1)([state, goal])
#
#     b_concat_1 = Dense(50, activation='relu', name="b_concat_1")(b_concat_0)
#     b_concat_2 = Dense(50, activation='relu', name="b_concat_2")(b_concat_1)
#
#     b_lstm_1 = LSTM(100, return_sequences=True, return_state=False, stateful=False, name="b_lstm_1")(b_concat_2)
#
#     b_concat_3 = Dense(50, activation='relu', name="b_concat_3")(b_lstm_1)
#     b_concat_4 = Dense(50, activation='relu', name="b_concat_4")(b_concat_3)
#
#     b_output_x = Dense(3, activation='softmax', name="x")(b_concat_4)
#     b_output_y = Dense(3, activation='softmax', name="y")(b_concat_4)
#     b_output_z = Dense(3, activation='softmax', name="z")(b_concat_4)
#     b_output_hand = Dense(2, activation='softmax', name="hand")(b_concat_4)
#
#     action_estimate = Concatenate(axis=-1)([b_output_x, b_output_y, b_output_z, b_output_hand])
#
#     # =========== forward_model ===========
#     forward = forward_net()
#     forward.trainable = False
#     next_state_estimate = forward([state, action_estimate])
#
#     # =========== metrics_model ===========
#     metrics = metric_net()
#     metrics.trainable = False
#     score = metrics([state, next_state_estimate, goal])
#
#     model = Model(inputs=[state, goal], outputs=[action_estimate, next_state_estimate, score], name='our_model')
#     return model


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
              batch_size=500,
              # initial_epoch=201,
              epochs=500,
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

    # model.load_weights('M.651-0.141286.hdf5', by_name=True)

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
              # initial_epoch=652,
              epochs=1000,
              verbose=1,
              validation_split=0.2,
              shuffle=True,
              callbacks=[tf_board, model_checkpoint])


def reshape_data(filename):

    data = pickle.load(open(filename, 'rb'))

    num_trajectory = 0
    last_index = 0
    num_length = []
    num_index = []

    for i in range(data.shape[0]):
        if data[i, -1] == 1.0:
            num_trajectory += 1
            length = i - last_index + 1
            num_length.append(length)
            num_index.append(i)
            last_index = i + 1

    num_length = np.array(num_length)
    num_index = np.array(num_index)
    print("mean:", num_length.mean(), "variance:", num_length.var(), "max:", num_length.max(), "min:", num_length.min())

    data_reshape = np.zeros((1000, 200, 69))

    for i in range(1000):
        data_reshape[i, 0:num_length[i], :] = data[num_index[i] - num_length[i] + 1:num_index[i] + 1, :]

    return data_reshape


def train_our_model_copy(model):
    # data = reshape_data('Pick-Place-Push-category-4-paths-1000.p')
    data = reshape_data('Pick-Place-Push-category-1000.p')

    # ==== get state feed ====
    i = np.array([0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16])
    state_feed = data[:, :, i]

    # ==== get goal feed ====
    goal_feed = data[:, :, 23:26]

    # ==== get action feed ====
    action_feed = data[:, :, 32:36]
    action_x_feed = action_feed[:, :, 0]
    action_y_feed = action_feed[:, :, 1]
    action_z_feed = action_feed[:, :, 2]
    action_hand_feed = action_feed[:, :, 3]

    action_x_feed = to_categorical(action_x_feed, 3)
    action_y_feed = to_categorical(action_y_feed, 3)
    action_z_feed = to_categorical(action_z_feed, 3)
    action_hand_feed = to_categorical(action_hand_feed, 2)

    # ==== get next state feed ====
    j = i + 36
    next_state_feed = data[:, :, j]

    # ==== get score feed ====
    score_feed = np.ones((data.shape[0], data.shape[1], 1))

    model.compile(optimizer=Adam(lr=1e-4),
                  loss=['categorical_crossentropy',
                        'categorical_crossentropy',
                        'categorical_crossentropy',
                        'categorical_crossentropy',
                        'mse',
                        'binary_crossentropy'],

                  metrics={'b_output_x': 'acc',
                           'b_output_y': 'acc',
                           'b_output_z': 'acc',
                           'b_output_hand': 'acc',
                           'f_output_state': 'mse',
                           'm_output_score': 'mse'},
                  loss_weights=[1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                  )

    model.load_weights('M.826-0.122378.hdf5', by_name=True)
    model.load_weights('F.490-0.000002.hdf5', by_name=True)

    tf_board = TensorBoard(log_dir='./logs',
                           histogram_freq=0,
                           write_graph=True,
                           write_images=False,
                           embeddings_freq=0,
                           embeddings_layer_names=None,
                           embeddings_metadata=None)

    model_checkpoint = ModelCheckpoint('our-GSP-one.{epoch:d}-{val_loss:.6f}.hdf5',
                                       monitor='val_loss',  # here 'val_loss' and 'loss' are the same
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True)

    model.fit([state_feed,
               goal_feed],
              [action_x_feed,
               action_y_feed,
               action_z_feed,
               action_hand_feed,
               next_state_feed,
               score_feed],
              batch_size=50,
              # initial_epoch=201,
              epochs=1000,
              verbose=1,
              validation_split=0.2,
              shuffle=True,
              callbacks=[tf_board, model_checkpoint])


def test_our_model_copy(model):
    step_size = 0.01

    env = gym.make('FetchPickAndPlace-v0')

    model.compile(optimizer=Adam(lr=1e-4),
                  loss=['categorical_crossentropy',
                        'categorical_crossentropy',
                        'categorical_crossentropy',
                        'categorical_crossentropy',],

                  metrics={'b_output_x': 'acc',
                           'b_output_y': 'acc',
                           'b_output_z': 'acc',
                           'b_output_hand': 'acc'},
                  )

    model.load_weights('our-GSP-one.999-1.265106.hdf5', by_name=True)

    i = np.array([0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16])
    while True:

        two_state = np.zeros((2, 1, 11))
        two_goal = np.zeros((2, 1, 3))
        observation = env.reset()
        two_state[0, 0, :] = observation["my_new_observation"][i]
        two_goal[0, 0, :] = observation["my_new_observation"][23:26]

        done = False

        model.reset_states()

        while not done:
            env.render()
            two_x, two_y, two_z, two_hand = model.predict_on_batch([two_state, two_goal])

            x = two_x[0, 0, :]
            y = two_y[0, 0, :]
            z = two_z[0, 0, :]
            hand = two_hand[0, 0, :]

            action = np.zeros(4, )

            if x.argmax() == 0:
                action[0] = 0
            elif x.argmax() == 1:
                action[0] = -(step_size / 0.03)
            elif x.argmax() == 2:
                action[0] = (step_size / 0.03)

            if y.argmax() == 0:
                action[1] = 0
            elif y.argmax() == 1:
                action[1] = -(step_size / 0.03)
            elif y.argmax() == 2:
                action[1] = (step_size / 0.03)

            if z.argmax() == 0:
                action[2] = 0
            elif z.argmax() == 1:
                action[2] = -(step_size / 0.03)
            elif z.argmax() == 2:
                action[2] = (step_size / 0.03)

            if hand.argmax() == 0:
                action[3] = -1.0
            elif hand.argmax() == 1:
                action[3] = 1.0

            observation, reward, done, info = env.step(action)
            two_state[0, 0, :] = observation["my_new_observation"][i]
            two_goal[0, 0, :] = observation["my_new_observation"][23:26]

            if done:
                print(True)


def check_usage_for_lstm(model):
    # this file is used for check:
    # whether model loading weights correctly
    # whether model using the state from previous time

    data = pickle.load(open('Pick-Place-Push-reshaped-category-1000.p', 'rb'))

    i = np.array([0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16])
    state_feed = data[:, :, i]
    goal_feed = data[:, :, 23:26]

    model.compile(optimizer=Adam(lr=1e-4),
                              loss=['categorical_crossentropy',
                                    'categorical_crossentropy',
                                    'categorical_crossentropy',
                                    'categorical_crossentropy', ],

                              metrics={'b_output_x': 'acc',
                                       'b_output_y': 'acc',
                                       'b_output_z': 'acc',
                                       'b_output_hand': 'acc'},
                              )

    model.load_weights('our.944-4.472422.hdf5', by_name=True)

    two_state = np.zeros((2, 1, 11))
    two_goal = np.zeros((2, 1, 3))

    two_state[0, 0, :] = state_feed[0, 0, :]
    two_state[1, 0, :] = state_feed[0, 0, :]

    two_goal[0, 0, :] = goal_feed[0, 0, :]
    two_goal[1, 0, :] = goal_feed[0, 0, :]

    action_two = model.predict_on_batch([two_state, two_goal])
    print(action_two)

    print("\n")

    model.reset_states()

    action_two = model.predict_on_batch([two_state, two_goal])
    print(action_two)


# forward_model = forward_net()
# train_forward_net(forward_model)


# metrics_model = metric_net()
# train_metrics_net(metrics_model)


# model = our_model_copy()
# train_our_model_copy(model)


model = our_model_copy_for_test()
test_our_model_copy(model)


# model = our_model_copy_for_test()
# check_usage_for_lstm(model)

