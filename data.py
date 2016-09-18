import numpy as np
from helpers import BatchIterator
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils

from datakit.mnist import MNIST
from datakit.cifar10 import Cifar10


def load_data(name,
              shuffle=True,
              valid_ratio=None,
              random_state=None):
    """
    if valid_ratio is None,
    take a good default, depending on dataset, and
    based on what people do in papers
    """

    info = {}
    if name == 'mnist':
        data = MNIST('train')
        data.load()
        data.X = data.X.reshape((data.X.shape[0], 1, 28, 28))
        train_X_full = data.X
        train_y_full = np_utils.to_categorical(data.y)
        data = MNIST('test')
        data.load()
        data.X = data.X.reshape((data.X.shape[0], 1, 28, 28))
        test_X = data.X
        test_y = np_utils.to_categorical(data.y)
        default_validation_size = 10000
    elif name == 'cifar10':
        data = Cifar10(train_or_test='train')
        data.load()
        data.X = data.X.reshape((data.X.shape[0], 3, 32, 32))
        train_X_full = data.X / 255.
        train_y_full = np_utils.to_categorical(data.y)
        data = Cifar10(train_or_test='test')
        data.load()
        data.X = data.X.reshape((data.X.shape[0], 3, 32, 32))
        test_X = data.X / 255.
        test_y = np_utils.to_categorical(data.y)
        default_validation_size = 10000
    else:
        raise Exception('Unknown dataset : {}'.format(name))
    train_iterator, valid_iterator, test_iterator, info = build_iterators(
        train_X_full,
        train_y_full,
        test_X,
        test_y,
        valid_ratio=valid_ratio,
        default_validation_size=default_validation_size,
        shuffle=shuffle,
        random_state=random_state
    )
    return train_iterator, valid_iterator, test_iterator, info


def build_iterators(train_X_full, train_y_full, test_X, test_y,
                    valid_ratio=None,
                    default_validation_size=10000,
                    shuffle=True,
                    random_state=None):
    if valid_ratio is not None:
        train_X, valid_X, train_y, valid_y = train_test_split(
            train_X_full, train_y_full,
            test_size=valid_ratio,
            random_state=random_state)
    else:
        rng = np.random.RandomState(random_state)
        indices = np.arange(train_X_full.shape[0])
        rng.shuffle(indices)
        train_X = train_X_full[default_validation_size:]
        train_y = train_y_full[default_validation_size:]
        valid_X = train_X_full[0:default_validation_size]
        valid_y = train_y_full[0:default_validation_size]

    print('Shape of training set   : {}'.format(train_X.shape))
    print('Shape of validation set : {}'.format(valid_X.shape))
    print('Shape of test set       : {}'.format(test_X.shape))

    train_iterator = BatchIterator(train_X, train_y,
                                   shuffle=shuffle,
                                   random_state=random_state)
    valid_iterator = BatchIterator(valid_X, valid_y,
                                   shuffle=False,
                                   random_state=random_state)
    test_iterator = BatchIterator(test_X, test_y,
                                  shuffle=False,
                                  random_state=random_state)
    info = {}
    info['nb_train_samples'] = train_X.shape[0]
    info['nb_valid_samples'] = valid_X.shape[0]
    info['nb_test_samples'] = test_X.shape[0]
    info['input_shape'] = train_X.shape[1:]
    info['nb_outputs'] = train_y.shape[1]
    return train_iterator, valid_iterator, test_iterator, info
