# state_space Box(17)
# action_space Box(6)

# import gym
# env = gym.make('HalfCheetah-v2')
# env.reset()
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample())

from keras.models import Model
from keras.layers import Input, Dense, Concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import *
import pickle

#(state,action,next_state,reward,done)
#    17     6          17      1    1
data = pickle.load(open('/home/zj/Desktop/4zj_HalfCheetah-v2_expert_traj.p','rb'))

state_feed = data[:,0:17]
action_feed = data[:,17:23]
next_state_feed = data[:,23:40]

state = Input(shape=(17,))
action = Input(shape=(6,))
next_state = Input(shape=(17,))

# Forward network
# state + action  -> next_state

F_STATE_T = state
F_ACTION_T = action

F_dense_s_1 = Dense(50, activation='relu')(F_STATE_T)
F_dense_s_2 = Dense(50, activation='relu')(F_dense_s_1)
F_dense_s_3 = Dense(50, activation='relu')(F_dense_s_2)

F_dense_a_1 = Dense(50, activation='relu')(F_ACTION_T)
F_dense_a_2 = Dense(50, activation='relu')(F_dense_a_1)
F_dense_a_3 = Dense(50, activation='relu')(F_dense_a_2)

F_input_1 = Concatenate(axis=-1)([F_dense_s_3,F_dense_a_3])

F_input_2 = Dense(50, activation='relu')(F_input_1)
F_input_3 = Dense(50, activation='relu')(F_input_2)
F_input_4 = Dense(50, activation='relu')(F_input_3)
next_state_output = Dense(17)(F_input_4)


# Backward network
# state + next_state -> action

B_STATE_T = state
B_ACTION_T = next_state

B_dense_s_1 = Dense(50, activation='relu')(B_STATE_T)
B_dense_s_2 = Dense(50, activation='relu')(B_dense_s_1)
B_dense_s_3 = Dense(50, activation='relu')(B_dense_s_2)

B_dense_a_1 = Dense(50, activation='relu')(B_ACTION_T)
B_dense_a_2 = Dense(50, activation='relu')(B_dense_a_1)
B_dense_a_3 = Dense(50, activation='relu')(B_dense_a_2)

B_input_1 = Concatenate(axis=-1)([B_dense_s_3,B_dense_a_3])

B_input_2 = Dense(50, activation='relu')(B_input_1)
B_input_3 = Dense(50, activation='relu')(B_input_2)
B_input_4 = Dense(50, activation='relu')(B_input_3)
action_output = Dense(6)(B_input_4)


# Recover network
# action + next_state -> state

R_STATE_T = next_state
R_ACTION_T = action

R_dense_s_1 = Dense(50, activation='relu')(R_STATE_T)
R_dense_s_2 = Dense(50, activation='relu')(R_dense_s_1)
R_dense_s_3 = Dense(50, activation='relu')(R_dense_s_2)

R_dense_a_1 = Dense(50, activation='relu')(R_ACTION_T)
R_dense_a_2 = Dense(50, activation='relu')(R_dense_a_1)
R_dense_a_3 = Dense(50, activation='relu')(R_dense_a_2)

R_input_1 = Concatenate(axis=-1)([R_dense_s_3,R_dense_a_3])

R_input_2 = Dense(50, activation='relu')(R_input_1)
R_input_3 = Dense(50, activation='relu')(R_input_2)
R_input_4 = Dense(50, activation='relu')(R_input_3)
state_output = Dense(17)(R_input_4)


#============================================================================

model = Model(inputs=[state,action,next_state], outputs=[state_output,action_output,next_state_output])
model.compile(optimizer = Adam(lr = 1e-4),
              loss = 'mean_squared_error',
              metrics=['mse'])

tf_board = TensorBoard(log_dir='./logs',
                    histogram_freq=0,
                    write_graph=True,
                    write_images=False,
                    embeddings_freq=0,
                    embeddings_layer_names=None,
                    embeddings_metadata=None)

early_stop = EarlyStopping(monitor='val_loss',
                               patience=0,
                               verbose=0,
                               mode='auto')

model_checkpoint = ModelCheckpoint('model_base_average_output.{epoch:02d}-{val_loss:.2f}.hdf5',
                                       monitor='val_loss',                        # here 'val_loss' and 'loss' are the same
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True)

model.fit([state_feed,action_feed,next_state_feed],
                          [state_feed, action_feed, next_state_feed],
                            batch_size=50,
                            epochs=1000,
                            verbose=1,
                            validation_split=0.2,
                            shuffle=True,
                            callbacks=[tf_board, early_stop, model_checkpoint])