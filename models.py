from keras.layers import * # NOQA
from keras.models import Model


def fc(hp, input_shape=(1, 28, 28), nb_outputs=10):
    nb_hidden = hp['nb_hidden']
    activation = hp['activation']
    inp = Input(shape=input_shape)
    x = Flatten()(inp)
    for nb_units in nb_hidden:
        x = Dense(nb_units, activation=activation)(x)
    x = Dense(nb_outputs, activation='softmax')(x)
    out = x
    return Model(inp, out)


def vgg_partial_(nb_filters=[64, 128, 256, 512, 512],
                 size_filters=[3, 3, 3, 3, 3],
                 stride=[2, 2, 2, 2, 2],
                 size_blocks=[2, 2, 3, 3, 3],
                 activation='relu',
                 input_shape=(3, 227, 227)):
    assert (len(nb_filters) == len(size_filters) ==
            len(stride) == len(size_blocks))
    nb_blocks = len(nb_filters)

    inp = Input(shape=input_shape)
    x = inp
    for i in range(nb_blocks):
        pad = (size_filters[i] - 1) / 2
        x = ZeroPadding2D((pad, pad))(x)
        for j in range(size_blocks[i]):
            x = Convolution2D(nb_filters[i],
                              size_filters[i], size_filters[i],
                              activation=activation)(x)
        x = MaxPooling2D((stride[i], stride[i]))(x)
    out = x
    return inp, out


def vgg(hp, input_shape=(3, 227, 227), nb_outputs=10):
    """
    nb_filters : list
    size_filters : int
    stride : int
    size_blocks : list
    activation : str
    fc : list
    fc_dropout : float
    """
    inp, out = vgg_partial_(
        nb_filters=hp['nb_filters'],
        size_filters=[hp['size_filters']] * len(hp['nb_filters']),
        stride=[hp['stride']] * len(hp['nb_filters']),
        size_blocks=hp['size_blocks'],
        activation=hp['activation'],
        input_shape=input_shape
    )
    x = out
    x = Flatten()(x)
    pr = hp['fc_dropout']
    for nb_units in hp['fc']:
        x = Dense(nb_units, activation=hp['activation'])(x)
        if pr > 0:
            x = Dropout(pr)(x)
    x = Dense(nb_outputs, activation='softmax')(x)
    out = x
    return Model(inp, out)


def nin(hp, input_shape=(3, 227, 227), nb_outputs=10):
    activation = hp['activation']

    inp = Input(shape=input_shape)
    x = inp
    x = Convolution2D(64, 5, 5, activation=activation)(x)
    x = Convolution2D(64, 1, 1, activation=activation)(x)
    x = Convolution2D(64, 1, 1, activation=activation)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)
    x = Convolution2D(128, 5, 5, activation=activation)(x)
    x = Convolution2D(128, 1, 1, activation=activation)(x)
    x = Convolution2D(128, 1, 1, activation=activation)(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.5)(x)
    x = Convolution2D(256, 5, 5, activation=activation)(x)
    x = Convolution2D(256, 1, 1, activation=activation)(x)
    x = Convolution2D(nb_outputs, 1, 1, activation=activation)(x)
    x = Flatten()(x)
    out = x
    return Model(inp, out)
