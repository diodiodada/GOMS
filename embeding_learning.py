from keras.models import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import *
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
import pickle
# import gym
import numpy as np
from keras import backend as K

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
#
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.visible_device_list = '2'
# set_session(tf.Session(config=config))


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
    print("mean:", num_length.mean(),
          "variance:", num_length.var(),
          "max:", num_length.max(),
          "min:", num_length.min(),
          "length:", num_length.shape[0])

    data_reshape = np.zeros((num_length.shape[0], 400, 51))

    state_0 = np.zeros((num_length.shape[0], 23))
    state_g = np.zeros((num_length.shape[0], 23))

    for i in range(num_length.shape[0]):
        data_reshape[i, 0:num_length[i], :] = data[num_index[i] - num_length[i] + 1:num_index[i] + 1, :]
        state_0[i, :] = data[num_index[i]-num_length[i]+1, 0:23]
        state_g[i, :] = data[num_index[i], 0:23]

    return data_reshape, state_0, state_g


def get_padding(x):
    return x[:, :-1, :]


def trajectory_encoder():

    states = Input(shape=(400, 23))
    state_0 = Input(shape=(23,))
    state_g = Input(shape=(23,))

    states_padding = Lambda(get_padding)(states)

    # =========== encoder ===========

    e_dense_1 = Dense(50, activation='relu', name="e_dense_1")(states)
    e_dense_2 = Dense(50, activation='relu', name="e_dense_2")(e_dense_1)

    e_lstm_1 = LSTM(100, return_sequences=False, stateful=False, name="e_lstm_1")(e_dense_2)

    e_dense_3 = Dense(50, activation='relu', name="e_dense_3")(e_lstm_1)
    e_dense_4 = Dense(50, activation='relu', name="e_dense_4")(e_dense_3)

    # =========== decoder ===========

    d_concat_0 = Concatenate(axis=-1)([e_dense_4, state_0, state_g])
    d_dense_0 = Dense(23, activation='relu', name="d_dense_0")(d_concat_0)
    d_reshape = Reshape((1, 23))(d_dense_0)
    d_concat_1 = Concatenate(axis=1)([d_reshape, states_padding])

    d_dense_1 = Dense(50, activation='relu', name="d_dense_1")(d_concat_1)
    d_dense_2 = Dense(50, activation='relu', name="d_dense_2")(d_dense_1)

    d_lstm_1 = LSTM(100, return_sequences=True, stateful=False, name="d_lstm_1")(d_dense_2)

    d_dense_3 = Dense(50, activation='relu', name="d_dense_3")(d_lstm_1)
    d_dense_4 = Dense(50, activation='relu', name="d_dense_4")(d_dense_3)

    d_output = Dense(23, activation='relu', name="d_output")(d_dense_4)

    model = Model(inputs=[states, state_0, state_g],
                  outputs=d_output,
                  name='encoder_model')

    # print(model.summary())

    return model


def train_trajectory_encoder(model):

    data, state_0_feed, state_g_feed = reshape_data('data/PP-24-paths-2400-0.p')

    # ==== get feed data ====
    states_feed = data[:, :, 0:23]
    output_feed = data[:, :, 0:23]

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

    model_checkpoint = ModelCheckpoint('weights/tra-encoder.{epoch:d}-{val_loss:.6f}.hdf5',
                                       monitor='val_loss',  # here 'val_loss' and 'loss' are the same
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True)

    model.fit([states_feed,
               state_0_feed,
               state_g_feed],
              output_feed,
              batch_size=50,
              # initial_epoch=201,
              epochs=1000,
              verbose=1,
              validation_split=0.2,
              shuffle=True,
              callbacks=[tf_board, model_checkpoint])


model = trajectory_encoder()
train_trajectory_encoder(model)

