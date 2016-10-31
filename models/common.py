from collections import namedtuple
from keras.layers import PReLU
Specs = namedtuple('Specs', ['input', 'output'], verbose=False)

def act(x, activation='relu'):
    if activation == 'prelu':
        x = PReLU()(x)
    else:
        x = Activation(activation)(x)
    return x
