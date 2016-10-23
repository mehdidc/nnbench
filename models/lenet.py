from keras.layers import *
from .common import Specs, act

def lenet(hp, input_shape=(3, 224, 224), nb_outputs=10):
    nb_filters = hp['nb_filters']
    pr = hp['dropout']
    batch_norm = hp['batch_norm']
    fc = hp['fc']
    size_filters = hp['size_filters']
    activation = hp['activation']

    inp = Input(input_shape)
    x = inp
    for k in nb_filters:
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = Convolution2D(k, size_filters, size_filters, init=init)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        x = act(x, activation=activation)
        x = MaxPooling2D((2, 2))(x)
        x = Dropout(pr)(x)
    x = Flatten()(x)
    for units in fc:
        x = Dense(nb_outputs)(x)
        x = act(x, activation=activation)
        x = Dropout(pr)(x)
    x = Dense(nb_outputs)(x)
    out = x
    return Specs(input=inp, output=out)
