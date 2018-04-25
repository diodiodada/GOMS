# state_space Box(17)
# action_space Box(6)

# import gym
# env = gym.make('HalfCheetah-v2')
# env.reset()
# for _ in range(1000):
#     env.render()
#     env.step(env.action_space.sample())

from keras.models import Model
from keras.layers import Input, Dense, Concatenate,Average
from keras.callbacks import ModelCheckpoint
from keras.optimizers import *
from keras.utils.vis_utils import plot_model
import pickle



# =================== define forward model ===================
# state + action  -> next_state
def create_forward_model():

    f_state = Input(shape=(17,))
    f_action = Input(shape=(6,))

    F_dense_s_1 = Dense(50, activation='relu', name='F_dense_s_1')(f_state)
    F_dense_s_2 = Dense(50, activation='relu', name='F_dense_s_2')(F_dense_s_1)
    F_dense_s_3 = Dense(50, activation='relu', name='F_dense_s_3')(F_dense_s_2)

    F_dense_a_1 = Dense(50, activation='relu', name='F_dense_a_1')(f_action)
    F_dense_a_2 = Dense(50, activation='relu', name='F_dense_a_2')(F_dense_a_1)
    F_dense_a_3 = Dense(50, activation='relu', name='F_dense_a_3')(F_dense_a_2)

    F_input_1 = Concatenate(axis=-1, name='F_input_1')([F_dense_s_3, F_dense_a_3])

    F_input_2 = Dense(50, activation='relu', name='F_input_2')(F_input_1)
    F_input_3 = Dense(50, activation='relu', name='F_input_3')(F_input_2)
    F_input_4 = Dense(50, activation='relu', name='F_input_4')(F_input_3)
    F_next_state_output = Dense(17, name='F_next_state_output')(F_input_4)

    MODEL_F = Model(inputs=[f_state, f_action], outputs=F_next_state_output)

    return MODEL_F

# =================== define backward model ===================
# state + next_state -> action
def create_backward_model():

    b_state = Input(shape=(17,))
    b_next_state = Input(shape=(17,))

    B_dense_s_1 = Dense(50, activation='relu', name='B_dense_s_1')(b_state)
    B_dense_s_2 = Dense(50, activation='relu', name='B_dense_s_2')(B_dense_s_1)
    B_dense_s_3 = Dense(50, activation='relu', name='B_dense_s_3')(B_dense_s_2)

    B_dense_a_1 = Dense(50, activation='relu', name='B_dense_a_1')(b_next_state)
    B_dense_a_2 = Dense(50, activation='relu', name='B_dense_a_2')(B_dense_a_1)
    B_dense_a_3 = Dense(50, activation='relu', name='B_dense_a_3')(B_dense_a_2)

    B_input_1 = Concatenate(axis=-1, name='B_input_1')([B_dense_s_3, B_dense_a_3])

    B_input_2 = Dense(50, activation='relu', name='B_input_2')(B_input_1)
    B_input_3 = Dense(50, activation='relu', name='B_input_3')(B_input_2)
    B_input_4 = Dense(50, activation='relu', name='B_input_4')(B_input_3)
    B_action_output = Dense(6, name='B_action_output')(B_input_4)

    MODEL_B = Model(inputs=[b_state, b_next_state], outputs=B_action_output)

    return MODEL_B

# =================== define recover model ===================
# action + next_state -> state
def create_recover_model():

    r_action = Input(shape=(6,))
    r_state = Input(shape=(17,))

    R_dense_a_1 = Dense(50, activation='relu', name='R_dense_a_1')(r_action)
    R_dense_a_2 = Dense(50, activation='relu', name='R_dense_a_2')(R_dense_a_1)
    R_dense_a_3 = Dense(50, activation='relu', name='R_dense_a_3')(R_dense_a_2)

    R_dense_s_1 = Dense(50, activation='relu', name='R_dense_s_1')(r_state)
    R_dense_s_2 = Dense(50, activation='relu', name='R_dense_s_2')(R_dense_s_1)
    R_dense_s_3 = Dense(50, activation='relu', name='R_dense_s_3')(R_dense_s_2)

    R_input_1 = Concatenate(axis=-1, name='R_input_1')([R_dense_s_3, R_dense_a_3])

    R_input_2 = Dense(50, activation='relu', name='R_input_2')(R_input_1)
    R_input_3 = Dense(50, activation='relu', name='R_input_3')(R_input_2)
    R_input_4 = Dense(50, activation='relu', name='R_input_4')(R_input_3)
    R_original_state_output = Dense(17, name='R_original_state_output')(R_input_4)

    MODEL_R = Model(inputs=[r_action, r_state], outputs=R_original_state_output)

    return MODEL_R


# =================== add computation node to the graph ===================
def add_computation_node(type, current_components, need):

    global MODEL_F
    global MODEL_B
    global MODEL_R

    # MODEL_F = create_forward_model()
    # MODEL_B = create_backward_model()
    # MODEL_R = create_recover_model()

    if 'forward' in need:
        # forward-network's type : 0,1,2,3
        if type[0] == 0:
            if ('s_t' in current_components) and ('a_t' in current_components):
                next_state_estimate = MODEL_F([current_components['s_t'], current_components['a_t']])
                return {'s_t+1^': next_state_estimate}, 'forward'
        elif type[0] == 1:
            if ('s_t^' in current_components) and ('a_t' in current_components):
                next_state_estimate = MODEL_F([current_components['s_t^'], current_components['a_t']])
                return {'s_t+1^': next_state_estimate}, 'forward'
        elif type[0] == 2:
            if ('s_t' in current_components) and ('a_t^' in current_components):
                next_state_estimate = MODEL_F([current_components['s_t'], current_components['a_t^']])
                return {'s_t+1^': next_state_estimate}, 'forward'
        elif type[0] == 3:
            if ('s_t^' in current_components) and ('a_t^' in current_components):
                next_state_estimate = MODEL_F([current_components['s_t^'], current_components['a_t^']])
                return {'s_t+1^': next_state_estimate}, 'forward'

    if 'backward' in need:
        # backward-network's type : 0,1,2,3
        if type[1] == 0:
            if ('s_t' in current_components) and ('s_t+1' in current_components):
                action_estimate = MODEL_B([current_components['s_t'], current_components['s_t+1']])
                return {'a_t^': action_estimate}, 'backward'
        elif type[1] == 1:
            if ('s_t^' in current_components) and ('s_t+1' in current_components):
                action_estimate = MODEL_B([current_components['s_t^'], current_components['s_t+1']])
                return {'a_t^': action_estimate}, 'backward'
        elif type[1] == 2:
            if ('s_t' in current_components) and ('s_t+1^' in current_components):
                action_estimate = MODEL_B([current_components['s_t'], current_components['s_t+1^']])
                return {'a_t^': action_estimate}, 'backward'
        elif type[1] == 3:
            if ('s_t^' in current_components) and ('s_t+1^' in current_components):
                action_estimate = MODEL_B([current_components['s_t^'], current_components['s_t+1^']])
                return {'a_t^': action_estimate}, 'backward'

    if 'recover' in need:
        # recover-network's type : 0,1,2,3
        if type[2] == 0:
            if ('a_t' in current_components) and ('s_t+1' in current_components):
                state_estimate = MODEL_R([current_components['a_t'], current_components['s_t+1']])
                return {'s_t^': state_estimate}, 'recover'
        elif type[2] == 1:
            if ('a_t^' in current_components) and ('s_t+1' in current_components):
                state_estimate = MODEL_R([current_components['a_t^'], current_components['s_t+1']])
                return {'s_t^': state_estimate}, 'recover'
        elif type[2] == 2:
            if ('a_t' in current_components) and ('s_t+1^' in current_components):
                state_estimate = MODEL_R([current_components['a_t'], current_components['s_t+1^']])
                return {'s_t^': state_estimate}, 'recover'
        elif type[2] == 3:
            if ('a_t^' in current_components) and ('s_t+1^' in current_components):
                state_estimate = MODEL_R([current_components['a_t^'], current_components['s_t+1^']])
                return {'s_t^': state_estimate}, 'recover'

# =================== add computation node to the graph ===================
def add_one_cell(F, B, R, input_for_25_nets):
    state = input_for_25_nets[0]
    action = input_for_25_nets[1]
    next_state = input_for_25_nets[2]

    components = {'s_t': state, 'a_t': action, 's_t+1': next_state}

    need = ['forward', 'backward', 'recover']

    # this line should be run three times !!!!!!
    # I try to add computation node in "add_computation_node" funtion
    # I am not building a model in "add_computation_node" funtion !!
    # "add_computation_node" funtion should prevent add same repeated node !
    node_return, network_type = add_computation_node([F, B, R], components, need)
    components.update(node_return)
    need.remove(network_type)

    node_return, network_type = add_computation_node([F, B, R], components, need)
    components.update(node_return)
    need.remove(network_type)

    node_return, network_type = add_computation_node([F, B, R], components, need)
    components.update(node_return)
    need.remove(network_type)


    return [components['s_t^'], components['a_t^'], components['s_t+1^']]

# =================== global variable for weight sharing ===================

MODEL_F = create_forward_model()
MODEL_B = create_backward_model()
MODEL_R = create_recover_model()

all_type = [
    '000', '001', '002', '003', '010', '012',
    '020', '021', '022', '023', '030', '032',
    '100', '101', '110',
    '120', '130',
    '200', '201', '202', '203', '210',
    '300', '301', '310',
]

def create_input():
    input_new = [Input(shape=(17,)),
                Input(shape=(6,)),
                Input(shape=(17,))]
    return input_new

# ==== construct the whole big network consisting of 25 networks ====
def construct_the_whole_network():

    input_for_25_nets = []

    outputs_for_25_nets = []

    for i in range(len(all_type)):
        f = int(all_type[i][0])
        b = int(all_type[i][1])
        r = int(all_type[i][2])

        state_input_for_1, action_input_for_1, next_state_input_for_1 = create_input()
        input_for_25_nets.append(state_input_for_1)
        input_for_25_nets.append(action_input_for_1)
        input_for_25_nets.append(next_state_input_for_1)

        outputs_for_25_nets.append( add_one_cell(f, b, r, [state_input_for_1,
                                                           action_input_for_1,
                                                           next_state_input_for_1] ))

        # print( models[i].name, all_type[i] )


    # calculate the average of the output
    state_outputs_for_25_nets=[]
    action_outputs_for_25_nets=[]
    next_state_outputs_for_25_nets=[]

    # separate the action state next_state
    for i in range(len(all_type)):
        state_outputs_for_25_nets.append(outputs_for_25_nets[i][0])
        action_outputs_for_25_nets.append(outputs_for_25_nets[i][1])
        next_state_outputs_for_25_nets.append(outputs_for_25_nets[i][2])

    # calculate the average
    outputs_for_25_nets_average = [Average()(state_outputs_for_25_nets),
                                   Average()(action_outputs_for_25_nets),
                                   Average()(next_state_outputs_for_25_nets)]

    # define the model
    model_for_25_nets = Model(inputs=input_for_25_nets, outputs=outputs_for_25_nets_average)

    # print(model_for_25_nets.layers)
    # yaml_string = model_for_25_nets.to_yaml()
    # print(yaml_string)

    # draw the picture of the network
    # plot_model(model_for_25_nets, to_file='model_for_25_nets_average.png',show_shapes = True, show_layer_names = True)
    # plot_model(model_for_25_nets, to_file='model_for_25_nets_average.png')

    return model_for_25_nets

# === training ===
def train(model_for_25_nets):

    # (state,action,next_state,reward,done)
    #    17     6          17      1    1
    data = pickle.load(open('/home/zj/Desktop/4zj_HalfCheetah-v2_expert_traj.p', 'rb'))

    state_feed = data[:, 0:17]
    action_feed = data[:, 17:23]
    next_state_feed = data[:, 23:40]

    # construct the input feed
    input_feed = []
    for i in range(25):
        input_feed.append(state_feed)
        input_feed.append(action_feed)
        input_feed.append(next_state_feed)

    model_for_25_nets.compile(optimizer = Adam(lr = 1e-4), loss = 'mean_squared_error')
    model_checkpoint = ModelCheckpoint('weights_average_output.{epoch:02d}-{val_loss:.2f}.hdf5',
                                       monitor='val_loss',                        # here 'val_loss' and 'loss' are the same
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True)

    model_for_25_nets.fit(input_feed,
                          [state_feed, action_feed, next_state_feed],
                            batch_size=50,
                            epochs=10,
                            verbose=1,
                            validation_split=0.2,
                            shuffle=True,
                            callbacks=[model_checkpoint])


if __name__ == '__main__':

    model_for_25_nets = construct_the_whole_network()
    train(model_for_25_nets)

