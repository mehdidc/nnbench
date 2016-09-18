import os

from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten,
    GlobalAveragePooling2D
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.normalization import BatchNormalization
from keras.utils.visualize_util import plot


# Helper to build a conv -> BN -> relu block
def _conv_bn_relu(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init="he_normal", border_mode="same")(input)
        norm = BatchNormalization(axis=1)(conv)
        return Activation("relu")(norm)

    return f

def _bn_relu(x):
    x = BatchNormalization(axis=1)(x)
    return Activation("relu")(x)

# Helper to build a conv -> BN 
def _conv_bn(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init="he_normal", border_mode="same")(input)
        norm = BatchNormalization(axis=1)(conv)
        return norm

    return f



# Helper to build a BN -> relu -> conv block
# This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        norm = BatchNormalization(axis=1)(input)
        activation = Activation("relu")(norm)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init="he_normal", border_mode="same")(activation)

    return f


# Bottleneck architecture for > 34 layer resnet.
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
# Returns a final conv layer of nb_filters * 4
def _bottleneck(nb_filters, init_subsample=(1, 1)):
    def f(input):
        conv_1_1 = _bn_relu_conv(nb_filters, 1, 1, subsample=init_subsample)(input)
        conv_3_3 = _bn_relu_conv(nb_filters, 3, 3)(conv_1_1)
        residual = _bn_relu_conv(nb_filters * 4, 1, 1)(conv_3_3)
        return _shortcut(input, residual)

    return f


# Basic 3 X 3 convolution blocks.
# Use for resnet with layers <= 34
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
def _basic_block(nb_filters, init_subsample=(1, 1)):
    def f(input):
        conv1 = _conv_bn_relu(nb_filters, 3, 3, subsample=init_subsample)(input)
        residual = _conv_bn_relu(nb_filters, 3, 3)(conv1)
        return _shortcut(input, residual)

    return f


# Adds a shortcut between input and residual block and merges them with "sum"
def _shortcut(input, residual):
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    stride_width = input._keras_shape[2] / residual._keras_shape[2]
    stride_height = input._keras_shape[3] / residual._keras_shape[3]
    equal_channels = residual._keras_shape[1] == input._keras_shape[1]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(nb_filter=residual._keras_shape[1], nb_row=1, nb_col=1,
                                 subsample=(stride_width, stride_height),
                                 bias=False,
                                 init="he_normal", border_mode="valid")(input)

    shortcut =  merge([shortcut, residual], mode="sum")
    shortcut = _bn_relu(shortcut)
    return shortcut

# Builds a residual block with repeating bottleneck blocks.
def _residual_block(block_function, nb_filters, repetations, is_first_layer=False):
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (2, 2)
            input = block_function(nb_filters=nb_filters, init_subsample=init_subsample)(input)
        return input

    return f

def resnet(hp, input_shape=(3, 224, 224), nb_outputs=10):
    size_blocks = hp['size_blocks']
    nb_filters = hp['nb_filters']
    block_fn = {'bottleneck': _bottleneck, 'basic': _basic_block}[hp['block']]
    assert len(size_blocks) == len(nb_filters)
    nb_blocks = len(size_blocks)

    inp = Input(input_shape)
    x = inp
    x = _conv_bn_relu(nb_filter=nb_filters[0], nb_row=3, nb_col=3)(x)
    
    # Build residual blocks..
    for i in range(nb_blocks):
        x = _residual_block(block_fn, nb_filters=nb_filters[i], repetations=size_blocks[i], is_first_layer=(i==0))(x)
    # Classifier block
    x = GlobalAveragePooling2D()(x)
    x = Dense(output_dim=nb_outputs, init="he_normal", activation="softmax")(x)
    out = x
    model = Model(input=inp, output=out)
    return model

def main():
    import time
    start = time.time()
    model = resnet({'nb_filters': [16, 32, 64], 'size_blocks': [5, 5, 5], 'block': 'basic'}, input_shape=(3, 32, 32))
    print(model.summary())
    plot(model, to_file='resnet.svg', show_shapes=True)

if __name__ == '__main__':
    main()
