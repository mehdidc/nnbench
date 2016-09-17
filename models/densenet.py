from keras.layers import * # NOQA
from keras.models import Model


def dense_block_(x, nb_layers, k=12, act='relu', p=0.2, size_filters=3):
    prev = x
    pad = (size_filters - 1) / 2
    for i in range(nb_layers):
        x = BatchNormalization()(x)
        x = Activation(act)(x)
        x = ZeroPadding2D((1, 1))(x)
        x = Convolution2D(k, size_filters, size_filters)(x)
        if p: 
            x = Dropout(p)(x)
        x = merge([prev, x], concat_axis=1, mode='concat')
        prev = x
    return x

def dense_transition_(x, k, act='relu', p=0.2, size_filter=1):
    x = BatchNormalization()(x)
    x = Activation(act)(x)
    x = Convolution2D(k, size_filter, size_filter)(x)
    x = AveragePooling2D((2, 2))(x)
    return x


def densenet(hp, input_shape=(3, 227, 227), nb_outputs=10):
    """
    growth : int, nb of feature maps to add between each dense block
    size_filter_block : int
    size_filter_transition: int
    dropout : float
    per_block : int
    nb_blocks : int
    act: str
    init_feature_maps : int
    """
    growth = hp['growth']
    size_filter_block = hp['size_filter_block']
    size_filter_transition = hp['size_filter_transition']
    dropout_p = hp['dropout']
    per_block = hp['per_block']
    nb_blocks = hp['nb_blocks']
    act = hp['activation']
    nb_init_fmaps = hp['init_feature_maps']

    inp = Input(shape=input_shape)
    x = inp

    pad = (size_filter_block - 1) / 2
    x = ZeroPadding2D((pad, pad))(x)
    x = Convolution2D(nb_init_fmaps, size_filter_block, size_filter_block)(x)
 
    for i in range(nb_blocks):
        x = dense_block_(
                x, 
                per_block, 
                k=growth, 
                act=act, 
                p=dropout_p, 
                size_filters=size_filter_block)
        if i < nb_blocks - 1:
            x = dense_transition_(
                x, 
                k=growth, 
                act=act, 
                p=dropout_p, 
                size_filter=size_filter_transition)
    x = GlobalAveragePooling2D()(x)
    x = Dense(nb_outputs, activation='softmax')(x)
    out = x
    return Model(inp, out)

