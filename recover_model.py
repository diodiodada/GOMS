from keras.models import Model
from keras.layers import Input, Dense, Concatenate,Average
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import *
from keras.utils.vis_utils import plot_model
import pickle



def create_recover_model():

    r_action = Input(shape=(4,))
    r_state = Input(shape=(10,))

    R_dense_a_1 = Dense(50, activation='relu')(r_action)
    R_dense_a_2 = Dense(50, activation='relu')(R_dense_a_1)
    R_dense_a_3 = Dense(50, activation='relu')(R_dense_a_2)

    R_dense_s_1 = Dense(50, activation='relu')(r_state)
    R_dense_s_2 = Dense(50, activation='relu')(R_dense_s_1)
    R_dense_s_3 = Dense(50, activation='relu')(R_dense_s_2)

    R_input_1 = Concatenate(axis=-1)([R_dense_s_3, R_dense_a_3])

    R_input_2 = Dense(50, activation='relu')(R_input_1)
    R_input_3 = Dense(50, activation='relu')(R_input_2)
    R_input_4 = Dense(50, activation='relu')(R_input_3)
    R_original_state_output = Dense(10)(R_input_4)

    MODEL_R = Model(inputs=[r_action, r_state], outputs=R_original_state_output, name='model_recover')

    return MODEL_R

def train(model):

    # (state,action,next_state,reward,done)
    #    17     6          17      1    1
    data = pickle.load(open('FetchReach-v0-sample.p', 'rb'))

    state_feed = data[:, 0:10]
    action_feed = data[:, 10:14]
    next_state_feed = data[:, 14:24]

    model.compile(optimizer = Adam(lr = 1e-4),
                              loss = 'mean_absolute_error',
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

    model_checkpoint = ModelCheckpoint('recover.{epoch:02d}-{val_loss:.6f}.hdf5',
                                       monitor='val_loss',                        # here 'val_loss' and 'loss' are the same
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True)

    model.fit([action_feed,next_state_feed],
                state_feed,
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
                              loss='mean_absolute_error',
                              metrics=['mse'])

    model.load_weights('recover.21-0.004708.hdf5', by_name=True)

    result = model.predict_on_batch([action_feed[0:2],next_state_feed[0:2]])
    print(result - next_state_feed[0:2])

    # result = model.test_on_batch([action_feed,next_state_feed],state_feed)
    # print(result)



if __name__ == '__main__':

    model = create_recover_model()
    # train(model)
    test(model)