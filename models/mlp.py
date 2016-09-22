from keras.layers import *
from keras.models import Model
def mlp(hp, input_shape=(3, 224, 224), nb_outputs=10):
    nb_hidden_units = hp['nb_hidden_units']
    act = hp['activation']
    inp = Input(input_shape)
    x = inp
    x = Flatten()(x)
    for nb in nb_hidden_units:
        x = Dense(nb)(x)
        x = Activation(act)(x)
    x = Dense(output_dim=nb_outputs, activation="softmax")(x)
    out = x
    model = Model(input=inp, output=out)
    return model