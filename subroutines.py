import warnings
warnings.filterwarnings("ignore")

#import random
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Activation
from keras.optimizers import RMSprop

# For reproducibility
np.random.seed(42)
tf.set_random_seed(42)

# =========================
# Subroutines
# =========================

def load_market_data(n_points):
    # Creates the market price
    price = 1.1+np.sin(6.1*np.pi*np.arange(n_points)/n_points) #sine prices
    return price

def build_features(market_data):
    # Creates the features from the market data
    # 1 - value of the market
    data = market_data
    # 2 - change in the value of the market
    diff = np.diff(data)
    diff = np.insert(diff, 0, 0)    
    # Stack the two features
    data = np.column_stack((data, diff))
    return data

def chose_action(qval, epsilon):
    # Decides whether the action will be greedy or not
    if (np.random.random() < epsilon):
        # Random action
        action = np.random.randint(0,2)
    else:
        # Greedily choose best action from Q(s,a)
        action = (np.argmax(qval))

    return action

def take_action(signal, action, time_step):
    # Generates a trade signal assuming
    # 2 actions (buy or hold) and
    # take action/store signal
    if action == 0:
        # hold
        signal[time_step] = 0
    elif action == 1:
        # buy
        signal[time_step] = 1
    return signal

def get_reward(time_step, data, signal, terminal_state):
    # Calculates the reward. Inputs are np vectors with the signal
    # (0 or 1 as a function of time) and the price of the market.
    # The output is a pandas dataframe with market price, the market change,
    # the signal, and the comulative return.

    # The reward is the product between the signal 1 (open position)
    # and the variation of the market. If not fnal state it uses
    # only aux_signal, the signal for the last set of times where
    # the position was open. 
    if terminal_state == 0:
        aux_signal = np.zeros(len(signal))
        if signal[time_step] == 1:
            aux_signal[time_step] = signal[time_step]
    else:
        aux_signal = signal

    df = pd.DataFrame()
    df['price']   = data[:,0]
    df['delta']   = data[:,1]
    df['signal']  = aux_signal
    df['ret']     = df['signal']*df['delta']
    df['cum_ret'] = df['ret'].cumsum()
    reward = df.ret.values.sum()

    return reward, df

def evaluate_Q(eval_model,data):
    # This evaluates the perofrmance of the system at each epoch
    # without the influence of epsilon/random actions

    signal = np.zeros(len(data))
    # Start in state S
    state = data[0:1,:]
    terminal_state = 0
    time_step = 0
    while(terminal_state == 0):
        # Run the Q function on state S to get
        # predicted reward on all possible actions
        qval = eval_model.predict(state)
        # Chose alwayst to take greedy action
        action = chose_action(qval, 0)

        # Time passes...make necessary adjustments to time step
        time_step += 1

        # Check if the final sate is reached
        if time_step < data.shape[0]:
            # Take action and return signal for time=time_step
            signal = take_action(signal, action, time_step)
            state = data[time_step:time_step+1, :]
        else:
            # If the last iteration is reached
            terminal_state = 1

            # The last signal is 0 (i.e. close position)
            signal[-1:] = 0
    
    # Observe total reward by setting terminal_state = 1
    eval_reward, df_out = get_reward(time_step, data, signal, 1)
    return eval_reward, df_out

def build_NN(n_features):
    # This neural network is the the Q-function
    model = Sequential()
    model.add(Dense(4, init='lecun_uniform', input_shape=(n_features,), activation='relu'))
    model.add(Dense(2, init='lecun_uniform', activation='linear'))
    # Two outputs corresponding to two possible actions
    # linear output so we can have range of real-valued outputs
    model.compile(loss='mse', optimizer=RMSprop())
    return model

def plot_final(df,historical):
    plt.figure(figsize=(10,10))
    # Market
    ax = plt.subplot2grid((4,1),(0,0))
    df.price.plot(label='Market price', color='Black')
    df.delta.plot(label='Market variation', color='Red')
    plt.axhline(y=0,dashes=[6, 2])
    plt.xlabel('Time')
    plt.ylabel('Market data')
    plt.legend(loc='upper left')
    # Signals
    ax = plt.subplot2grid((4,1),(1,0))
    df.signal.plot(label='Last agent')
    df.optimal_signal.plot(label='Optimal')
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.legend(loc='upper left')
    # Cumulative returns
    ax = plt.subplot2grid((4,1),(2,0))
    df.cum_ret.plot(label='Last agent')
    df.optimal_cum_ret.plot(label='Optimal')
    plt.axhline(y=0,dashes=[6, 2])
    plt.xlabel('Time')
    plt.ylabel('Cumulative return')
    plt.legend(loc='upper left')
    # Historical data
    ax = plt.subplot2grid((4,1),(3,0))
    plt.plot(historical,label='Total return')
    plt.axhline(y=0,dashes=[6, 2])
    plt.xlabel('Epochs')
    plt.ylabel('Return')
    plt.legend(loc='upper left')

    plt.savefig('results.pdf', figsize=(10,10))
    plt.tight_layout()
    plt.show()

    