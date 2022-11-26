# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 16:29:56 2018
@author: henrique.ferreira
"""

import random
import numpy as np
import matplotlib.pyplot as plt

def generate_acc_sensor_read_x(time_vector, state_vector, limit = 1e-2):
    tv = time_vector
    size = len(tv)
    for i in range(size):
        ins = tv[i]*1e-1 # time instant
        acc   = 0.002010664 + 0.682648*ins - 1.041474*ins**2 + 0.3509789*ins**3
        # acc = min(acc, 0.05)
        acc_noise = 0 + random.uniform(-2,1)*limit
        state_vector[i][2]  = acc + acc_noise
    return state_vector

def generate_acc_sensor_read_y(time_vector, state_vector, limit = 10e-2):
    return None

def calculate_next_state(states, dt):
    p_0, v_0, a_0 = states[0]
    p_1, v_1, a_1 = states[1]
    a_1 = a_1
    v_1 = v_0+a_0*dt
    p_1 = p_0+v_0*dt
    next_state = [p_1, v_1, a_1]
    return next_state

def simulate(data_0: np.ndarray, total_steps:int()):
    dt = 5e-3
    tv = (np.array(range(total_steps)))*dt # time vector   
    state_vector_shape = np.ndarray((total_steps,3))
    state_vector_x = generate_acc_sensor_read_x(tv,state_vector_shape)
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

state_x_0 = np.array([0,0,0])
state_y_0 = np.array([0,0,0])
data_0 = np.array([state_x_0,state_y_0])

data = simulate(data_0,int(4e3))
# acc_data = (1*data[0]**4 - 10*data[0]**3 + 25*data[0]**2 - data[0]*2)*0.05 + acc_noise

time = data[0]
position_x = data[1][:,0]
accelera_x = data[1][:,2]


plt.subplot(1, 2, 1)
plt.plot(time,accelera_x)

plt.subplot(1, 2, 2)
plt.plot(time,position_x)




# figure = plt.figure()
# subplot_0 = figure.add_subplot()
# # subplot_0.plot(data[0], data[2][:,2])
# subplot_0.plot(data[0], acc_data, lw=0.1)

# import random
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# dt = 2e-3
# tsampler = 500*1
# a,v,p = [0]*tsampler,[0]*tsampler,[0]*tsampler
# bias = 0e-0
# xmax = []


# for i in range(tsampler): #simulating x
#     a[i] = random.uniform(-1,1)*10e-2
#     v[i] = v[i-1]+a[i]*dt
#     p[i] = p[i-1]+v[i]*dt
# x = np.array(p)

#for j in range(1000):
#    for i in range(tsampler): #simulating x
#        a[i] = random.uniform(-1,1)*10e-2
#        v[i] = v[i-1]+a[i]*dt
#        p[i] = p[i-1]+v[i]*dt
#    buff = np.array(p)
#    if abs(buff.max()) >= abs(buff.min()):
#        buff = buff.max()
#    else:
#        buff = buff.min()
#    xmax.append(buff)
#    a,v,p = [0]*tsampler,[0]*tsampler,[0]*tsampler
#xmax = np.array(xmax)
#print(np.mean(xmax))
#print(np.var(xmax)**0.5)
#xmax = np.sort(xmax)
#axis = np.array(range(1000))


# a,v,p = [0]*tsampler,[0]*tsampler,[0]*tsampler
# for i in range(tsampler): # simulating y
#     a[i] = random.uniform(-1,1)*10e-2
#     v[i] = v[i-1]+a[i]*dt
#     p[i] = p[i-1]+v[i]*dt

# y = np.array(p)

# plt.plot(x,y)
#fig, ax = plt.subplots()
#
#LinesList = ax.plot(x,y)
#line = LinesList[0]
#axissize = 2e-3
#plt.xlim(-axissize,axissize)
#plt.ylim(-axissize,axissize)
#window = 100
#datasize = len(x)
#
#def animate(i):
#    start = i
#    if start >= datasize-window:
#        start = datasize-window
#    line.set_xdata(x[start:start+window])
#    line.set_ydata(y[start:start+window])
#    if i % 100 == 0:
#        print(i)
#    return line
#
#AnimationObject = animation.FuncAnimation(
#    fig, animate, interval=1,frames = datasize-window)