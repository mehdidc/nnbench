from keras.layers import * # NOQA
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU

def get_activation(act):
    if act == 'leaky_relu':
        return LeakyReLU(alpha=0.3)
    else:
        return Activation(act)

def fc(hp, input_shape=(1, 28, 28), nb_outputs=10):
    nb_hidden = hp['nb_hidden']
    act = hp['activation']
    inp = Input(shape=input_shape)
    x = Flatten()(inp)
    for nb_units in nb_hidden:
        x = Dense(nb_units)(x)
        x = activation(x)
        x = get_activation(act)(x)
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
        for j in range(size_blocks[i]):
            x = ZeroPadding2D((pad, pad))(x)
            x = Convolution2D(nb_filters[i],
                              size_filters[i], size_filters[i])(x)
            x = get_activation(activation)(x)
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

    act = hp['activation']
    inp, out = vgg_partial_(
        nb_filters=hp['nb_filters'],
        size_filters=[hp['size_filters']] * len(hp['nb_filters']),
        stride=[hp['stride']] * len(hp['nb_filters']),
        size_blocks=hp['size_blocks'],
        activation=act,
        input_shape=input_shape
    )
    x = out
    x = Flatten()(x)
    pr = hp['fc_dropout']
    for nb_units in hp['fc']:
        x = Dense(nb_units)(x)
        x = get_activation(act)(x)
        if pr > 0:
            x = Dropout(pr)(x)
    x = Dense(nb_outputs, activation='linear')(x)
    out = x
    return Model(inp, out)

if __name__ == '__main__':
    from keras.utils.visualize_util import plot
    hp = {
        'nb_filters': [64, 128, 256, 512, 512],
        'size_blocks': [2, 2, 3, 3, 3],
        'size_filters': 3,
        'stride': 2,
        'fc': [4096, 4096],
        'fc_dropout': 0.5,
        'activation': 'relu'
    }
    model = vgg(hp, input_shape=(3, 224, 224), nb_outputs=1000)
    print(model.summary())
    nb = sum(1 for layer in model.layers if hasattr(layer, 'W'))
    print('Number of learnable layers : {}'.format(nb))
    plot(model, to_file='vgg.svg', show_shapes=True)
