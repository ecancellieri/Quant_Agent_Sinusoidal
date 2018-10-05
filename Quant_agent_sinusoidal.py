'''
Author:      Emiliano Canvellieri
Created:     05/10/2018
Requirements:
Numpy
Pandas
MatplotLib
scikit-learn
Keras, https://keras.io/
TensorFlow
'''

import timeit
import numpy as np
import pandas as pd
import subroutines as sub
from matplotlib import pyplot as plt

# =========================
# Settings
# =========================

epochs     = 50
epsilon    = 1    # to explore non-greedly other actions
n_points   = 100  # number of data points in market price

# =========================
# Start trading simulation
# =========================

start_time = timeit.default_timer()
market     = sub.load_market_data(n_points)
data       = sub.build_features(market)
model      = sub.build_NN(data.shape[1]) # data.shape[1] is the number of features
historical = []

for i in range(epochs):
    signal = np.zeros(n_points)
    #Start in state S
    state = data[0:1,:]
    terminal_state = 0
    time_step = 0
    while(terminal_state == 0):
        # Run the Q function on state S to get
        # predicted reward on all possible actions
        qval = model.predict(state)

        # Chose which action to take in the future
        # (greedy or not with probability epsilon)
        action = sub.chose_action(qval, epsilon)

        # Time passes...make necessary adjustments to time step
        time_step += 1

        # Check if the final sate is reached
        if time_step < n_points:
            # Take action and return signal for time=time_step
            signal = sub.take_action(signal, action, time_step)

            # Observe reward at the new time
            reward, df = sub.get_reward(time_step, data, signal, terminal_state)
        else:
            # If the last iteration is reached
            terminal_state = 1

            # The last signal is 0 (i.e. close position)
            signal[-1:] = 0

            # Observe reward at the new time
            reward, df = sub.get_reward(time_step, data, signal, terminal_state)

        # Update the target output of the last action taken
        y = qval[:] # target output
        y[0][action] = reward

        # Fit the agent to take into account the effect of the last action
        model.fit(state, y, batch_size=1, epochs=1, verbose=0)

        # Move the market data window one step forward
        state = data[time_step:time_step+1, :]

    # At the end of each epoch evaluate the agent
    eval_reward, df_out = sub.evaluate_Q(model, data)
    historical = np.append(historical ,eval_reward)
    print("Epoch #: %s Reward: %f Epsilon: %f" % (i,eval_reward, epsilon))
    if epsilon > 0.1:
        epsilon = epsilon - (1.0/epochs)

elapsed = np.round(timeit.default_timer() - start_time)
print("Completed in %.3f" %elapsed)

# =========================
# Plot results
# =========================

# Create data frame with last agent
aux, df_best_agent = sub.evaluate_Q(model, data)

# Calculate the optimal solution
df_best_agent['best_signal'] = np.where(df_best_agent['delta']>0, 1, 0)
optimal_reward, df_optimal = sub.get_reward(0, data, df_best_agent.best_signal.values, 1)
print('Best possible reward', optimal_reward)
df_best_agent['optimal_signal'] = df_optimal['signal']
df_best_agent['optimal_cum_ret'] = df_optimal['cum_ret']

sub.plot_final(df_best_agent,np.asarray(historical))


