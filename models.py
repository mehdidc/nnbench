from keras.layers import * # NOQA
from keras.models import Model


def fully(input_shape=(1, 28, 28), nb_outputs=10, hp=None):
    assert hp, 'the hyper-parameters of the model should be set'
    nb_hidden = hp['nb_hidden']
    nb_layers = hp['nb_layers']
    activation = hp['activation']
    inp = Input(shape=input_shape)
    x = Flatten()(inp)
    for i in range(nb_layers):
        x = Dense(nb_hidden, activation=activation)(x)
    x = Dense(nb_outputs, activation='softmax')(x)
    out = x
    return Model(inp, out)
