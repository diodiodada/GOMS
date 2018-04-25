from keras.models import Model
from keras.layers import Input, Dense, Concatenate,Average
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import *
from keras.utils.vis_utils import plot_model
import pickle



def create_backward_model():

    b_state = Input(shape=(10,))
    b_next_state = Input(shape=(10,))

    B_dense_s_1 = Dense(50, activation='relu')(b_state)
    B_dense_s_2 = Dense(50, activation='relu')(B_dense_s_1)
    B_dense_s_3 = Dense(50, activation='relu')(B_dense_s_2)

    B_dense_a_1 = Dense(50, activation='relu')(b_next_state)
    B_dense_a_2 = Dense(50, activation='relu')(B_dense_a_1)
    B_dense_a_3 = Dense(50, activation='relu')(B_dense_a_2)

    B_input_1 = Concatenate(axis=-1)([B_dense_s_3, B_dense_a_3])

    B_input_2 = Dense(50, activation='relu')(B_input_1)
    B_input_3 = Dense(50, activation='relu')(B_input_2)
    B_input_4 = Dense(50, activation='relu')(B_input_3)
    B_action_output = Dense(4)(B_input_4)

    MODEL_B = Model(inputs=[b_state, b_next_state], outputs=B_action_output, name = 'model_backward')

    return MODEL_B

def train(model):

    # (state,action,next_state,reward,done)
    #    17     6          17      1    1
    data = pickle.load(open('FetchReach-v0-sample.p', 'rb'))

    state_feed = data[:, 0:10]
    action_feed = data[:, 10:14]
    next_state_feed = data[:, 14:24]

    model.compile(optimizer = Adam(lr = 1e-4),
                              loss = 'mae',
                              metrics=['mse'])

    tf_board = TensorBoard(log_dir='./logs',
                           histogram_freq=0,
                           write_graph=True,
                           write_images=False,
                           embeddings_freq=0,
                           embeddings_layer_names=None,
                           embeddings_metadata=None)

    early_stop = EarlyStopping(monitor='val_loss',
                               patience=5,
                               verbose=0,
                               mode='auto')

    model_checkpoint = ModelCheckpoint('backward.{epoch:02d}-{val_loss:.6f}.hdf5',
                                       monitor='val_loss',                        # here 'val_loss' and 'loss' are the same
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True)

    model.fit([state_feed,next_state_feed],
              action_feed,
            batch_size=50,
            epochs=100,
            verbose=1,
            validation_split=0.2,
            shuffle=True,
            callbacks=[tf_board, early_stop , model_checkpoint])


def test(model):
    data = pickle.load(open('FetchReach-v0-sample2.p', 'rb'))

    state_feed = data[:, 0:10]
    action_feed = data[:, 10:14]
    next_state_feed = data[:, 14:24]

    model.compile(optimizer=Adam(lr=1e-4),
                              loss='mae',
                              metrics=['mse'])

    model.load_weights('backward.18-0.162757.hdf5', by_name=True)

    result = model.predict_on_batch([state_feed[0:2],next_state_feed[0:2]])
    print(result - action_feed[0:2])

    # result = model.test_on_batch([state_feed,next_state_feed],action_feed)
    # print(result)



if __name__ == '__main__':

    model = create_backward_model()
    # train(model)
    test(model)