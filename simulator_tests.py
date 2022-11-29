# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 16:29:56 2018
@author: henrique.ferreira
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_acc_sensor_read_x(time_vector, state_vector, add_noise=True):
    tv = time_vector
    size = len(tv)
    for i in range(size):
        ins = tv[i]*1e-1 # time instant
        acc   = 0.683*ins - 1.041*ins**2 + 0.351*ins**3
        if add_noise:
            acc += np.random.uniform(float(-1),float(1))*1e-2
        state_vector[i][2]  = acc
    return state_vector

def calculate_next_state(states, dt):
    p_0, v_0, a_0 = states[0]
    p_1, v_1, a_1 = states[1]
    a_1 = a_1
    v_1 = v_0+a_0*dt
    p_1 = p_0+v_0*dt
    next_state = [p_1, v_1, a_1]
    return next_state

def simulate(data_0: np.ndarray, total_steps:int(), add_noise_flag):
    dt = 5e-3
    tv = (np.array(range(total_steps)))*dt # time vector   
    state_vector_shape = np.ndarray((total_steps,3))
    state_vector_x = generate_acc_sensor_read_x(tv, state_vector_shape, add_noise=add_noise_flag)
    # state_vector_y = generate_acc_sensor_read_y(tv,state_vector_shape)
    state_vector_x[0] = data_0[0]
    # state_vector_y[0] = data_0[1]
    
    for instant_i in range(total_steps-1):
        states_x = [state_vector_x[instant_i+0], state_vector_x[instant_i+1]]
        # states_y = [state_vector_y[instant_i+0], state_vector_y[instant_i+1]]
        state_vector_x[instant_i+1] = calculate_next_state(states_x, dt)
        # state_vector_y[instant_i+1]= calculate_next_state(states_y, dt)
    
    simulation_data = [tv, state_vector_x]
    return simulation_data

def probe(data, probe_spacing:int(), probe_samples:int(), plot:bool=False):
    data_size = len(data[0])
    probe_relative_indexes = list(range(0,probe_spacing*probe_samples,probe_spacing))
    max_start_index = (data_size-probe_relative_indexes[-1]) -1
    start_index = np.random.randint(0,max_start_index)
    probe_absolute_indexes = np.add(start_index,probe_relative_indexes)
    probed_data_0 = []
    probed_data_1 = []
    for i in probe_absolute_indexes:
        probed_data_0.append(data[0][i])
        probed_data_1.append(data[1][i])
    if plot:
        plt.plot(probe_absolute_indexes, probed_data_0)
        plt.plot(probe_absolute_indexes, probed_data_1)
    return [probed_data_0, probed_data_1]
    
def create_data():
    state_x_0 = np.array([0,0,0])
    state_y_0 = np.array([0,0,0])
    data_0 = np.array([state_x_0,state_y_0])
    
    data_clean = simulate(data_0,int(4e3), add_noise_flag=False)
    data_noised = simulate(data_0,int(4e3), add_noise_flag=True)
    
    model_dataVA = data_noised[1][:,2]
    model_dataVA_clean = data_clean[1][:,2]
    plt.show()
    return [model_dataVA, model_dataVA_clean]

def create_bunch_data(bunch_size):
    model_bunch_data = []
    model_data = create_data()
    for i in range(bunch_size):
        probe_data = probe(model_data, probe_spacing = 10, probe_samples = 50, plot=False)
        model_bunch_data.append(probe_data)
    return model_bunch_data

print('bunch_data[i][0] is de noised data, bunch_data[i][1] is de clean one')
bunch_data = create_bunch_data(1000)