from keras.models import Model
from keras.layers import Input, Dense, Concatenate,Average
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import *
from keras.utils.vis_utils import plot_model
import pickle

def state_feature():
    state = Input(shape=(17,))

    s_1 = Dense(50, activation='relu', name = 's_1')(state)
    s_2 = Dense(50, activation='relu', name = 's_2')(s_1)
    s_3 = Dense(50, activation='relu', name = 's_3')(s_2)

    state_feature = Model(inputs=state, outputs=s_3, name='state_feature')

    return state_feature

def action_feature():
    action = Input(shape=(6,))

    a_1 = Dense(50, activation='relu', name = 'a_1')(action)
    a_2 = Dense(50, activation='relu', name = 'a_2')(a_1)
    a_3 = Dense(50, activation='relu', name = 'a_3')(a_2)

    action_feature = Model(inputs=action, outputs=a_3, name='action_feature')

    return action_feature

# =================== define forward model ===================
# state + action  -> next_state
def create_forward_model():
    global state_feature
    global action_feature

    f_state = Input(shape=(17,))
    f_action = Input(shape=(6,))

    F_dense_s_3 = state_feature(f_state)
    F_dense_a_3 = action_feature(f_action)

    F_concat = Concatenate(axis=-1, name='F_concat')([F_dense_s_3, F_dense_a_3])

    F_concat_1 = Dense(50, activation='relu', name='F_concat_1')(F_concat)
    F_concat_2 = Dense(50, activation='relu', name='F_concat_2')(F_concat_1)
    F_concat_3 = Dense(50, activation='relu', name='F_concat_3')(F_concat_2)
    F_next_state_output = Dense(17, name='F_next_state_output')(F_concat_3)

    MODEL_F = Model(inputs=[f_state, f_action], outputs=F_next_state_output, name = 'forward_model')

    return MODEL_F

# =================== define backward model ===================
# state + next_state -> action
def create_backward_model():
    global state_feature
    global action_feature

    b_state = Input(shape=(17,))
    b_next_state = Input(shape=(17,))

    B_dense_s_3 = state_feature(b_state)
    B_dense_a_3 = state_feature(b_next_state)

    B_concat = Concatenate(axis=-1, name='B_concat')([B_dense_s_3, B_dense_a_3])

    B_concat_1 = Dense(50, activation='relu', name='B_concat_1')(B_concat)
    B_concat_2 = Dense(50, activation='relu', name='B_concat_2')(B_concat_1)
    B_concat_3 = Dense(50, activation='relu', name='B_concat_3')(B_concat_2)
    B_action_output = Dense(6, name='B_action_output')(B_concat_3)

    MODEL_B = Model(inputs=[b_state, b_next_state], outputs=B_action_output, name = 'backward_model')

    return MODEL_B

# =================== define recover model ===================
# action + next_state -> state
def create_recover_model():
    global state_feature
    global action_feature

    r_action = Input(shape=(6,))
    r_state = Input(shape=(17,))

    R_dense_a_3 = action_feature(r_action)
    R_dense_s_3 = state_feature(r_state)

    R_concat = Concatenate(axis=-1, name='R_concat')([R_dense_s_3, R_dense_a_3])

    R_concat_1 = Dense(50, activation='relu', name='R_input_2')(R_concat)
    R_concat_2 = Dense(50, activation='relu', name='R_input_3')(R_concat_1)
    R_concat_3 = Dense(50, activation='relu', name='R_input_4')(R_concat_2)
    R_original_state_output = Dense(17, name='R_original_state_output')(R_concat_3)

    MODEL_R = Model(inputs=[r_action, r_state], outputs=R_original_state_output, name = 'recover_model')

    return MODEL_R

state_feature = state_feature()
action_feature = action_feature()

def test():

    # (state,action,next_state,reward,done)
    #    17     6          17      1    1
    data = pickle.load(open('/home/zj/Desktop/4zj_HalfCheetah-v2_expert_traj.p', 'rb'))

    state_feed = data[:, 0:17]
    action_feed = data[:, 17:23]
    next_state_feed = data[:, 23:40]

    model_f = create_forward_model()

    input_for_net = [Input(shape=(17,)), Input(shape=(6,))]
    output_for_net = model_f(input_for_net)

    model_for_25_nets = Model(inputs=input_for_net, outputs=output_for_net)

    model_for_25_nets.load_weights('weights_average_output.57-0.34.hdf5', by_name=True)

    input1 = state_feed[0:2, :]
    input2 = action_feed[0:2, :]

    result = model_for_25_nets.predict([input1, input2], batch_size=2, verbose=0)

    print(result)
    print("==========================================")
    print(next_state_feed[0:2])
    print("==========================================")
    print(result - next_state_feed[0:2])

def test1():

    # (state,action,next_state,reward,done)
    #    17     6          17      1    1
    data = pickle.load(open('/home/zj/Desktop/4zj_HalfCheetah-v2_expert_traj.p', 'rb'))

    state_feed = data[:, 0:17]
    action_feed = data[:, 17:23]
    next_state_feed = data[:, 23:40]

    model_f = create_forward_model()

    model_f.load_weights('weights_one_model.99-1.24.hdf5', by_name=True)

    input1 = state_feed[0:2, :]
    input2 = action_feed[0:2, :]

    result = model_f.predict([input1, input2], batch_size=2, verbose=0)

    print(result)
    print("==========================================")
    print(next_state_feed[0:2])
    print("==========================================")
    print(result - next_state_feed[0:2])

def train():

    # (state,action,next_state,reward,done)
    #    17     6          17      1    1
    data = pickle.load(open('/home/zj/Desktop/4zj_HalfCheetah-v2_expert_traj.p', 'rb'))

    state_feed = data[:, 0:17]
    action_feed = data[:, 17:23]
    next_state_feed = data[:, 23:40]

    model_f = create_forward_model()

    input_for_net = [Input(shape=(17,)), Input(shape=(6,))]
    output_for_net = model_f(input_for_net)

    model_for_25_nets = Model(inputs=input_for_net, outputs=output_for_net)

    model_for_25_nets.compile(optimizer=Adam(lr=1e-4),
                  loss='mean_squared_error',
                  metrics=['mse'])

    tf_board = TensorBoard(log_dir='./logs',
                           histogram_freq=0,
                           write_graph=True,
                           write_images=False,
                           embeddings_freq=0,
                           embeddings_layer_names=None,
                           embeddings_metadata=None)

    early_stop = EarlyStopping(monitor='val_loss',
                               patience=3,
                               verbose=0,
                               mode='auto')

    model_checkpoint = ModelCheckpoint('weights_one_model.{epoch:02d}-{val_loss:.2f}.hdf5',
                                       monitor='val_loss',  # here 'val_loss' and 'loss' are the same
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True)

    model_for_25_nets.fit([state_feed, action_feed],
              next_state_feed,
              batch_size=50,
              epochs=100,
              verbose=1,
              validation_split=0.2,
              shuffle=True,
              callbacks=[tf_board, early_stop, model_checkpoint])

if __name__ == '__main__':
    # train()
    test()
    # test1()