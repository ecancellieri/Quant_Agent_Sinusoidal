# Quant_Agent_Sinusoidal
A reinforcement learning code for an agent that learns to trade on a market that is a simple sinusoidal curve. The Q function is a Neural Network with 2 layers, 2 features as imput, and 2 linear outputs. The possible actions are either to buy (1) or to hold (0). A buy signal after a buy signal means the position is kept open, a hold signal after a buy signal is equivalent to sell.

Needs:
Numpy
Pandas
MatplotLib
scikit-learn
Keras
TensorFlow
