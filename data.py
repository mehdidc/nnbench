import numpy as np
from helpers import BatchIterator
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils

from datakit.mnist import MNIST


def load_data(name, batch_size=128,
              shuffle=True,
              valid_ratio=None,
              random_state=None):
    """
    if valid_ratio is None,
    take a good default, depending on dataset based on what people do in papers
    """

    info = {}
    if name == 'mnist':
        data = MNIST('train')
        data.load()
        data.X = data.X.reshape((data.X.shape[0], 1, 28, 28))
        train_X_full = data.X
        train_y_full = np_utils.to_categorical(data.y)

        if valid_ratio is not None:
            train_X, valid_X, train_y, valid_y = train_test_split(
                train_X_full, train_y_full,
                test_size=valid_ratio)
        else:
            rng = np.random.RandomState(random_state)
            indices = np.arange(train_X_full.shape[0])
            rng.shuffle(indices)
            train_X = train_X_full[10000:]
            train_y = train_y_full[10000:]
            valid_X = train_X_full[0:10000]
            valid_y = train_y_full[0:10000]

        data = MNIST('test')
        data.load()
        data.X = data.X.reshape((data.X.shape[0], 1, 28, 28))
        test_X = data.X
        test_y = np_utils.to_categorical(data.y)

        train_iterator = BatchIterator(train_X, train_y,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       random_state=random_state)
        valid_iterator = BatchIterator(valid_X, valid_y,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       random_state=random_state)
        test_iterator = BatchIterator(test_X, test_y,
                                      batch_size=batch_size,
                                      shuffle=False)
        info['nb_train_samples'] = train_X.shape[0]
        info['nb_valid_samples'] = valid_X.shape[0]
        info['nb_test_samples'] = test_X.shape[0]
        info['input_shape'] = (1, 28, 28)
        info['nb_outputs'] = 10

    else:
        raise Exception('unknown dataset : {}'.format(name))
    return train_iterator, valid_iterator, test_iterator, info
