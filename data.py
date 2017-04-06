from itertools import cycle, imap
from functools import partial

from helpers import floatX
import numpy as np
from helpers import ArrayBatchIterator, BatchIterator
from sklearn.cross_validation import train_test_split
from skimage.util import pad as skimage_pad
from keras.utils import np_utils
from keras.preprocessing.image import random_rotation, random_shift, random_shear, random_zoom, random_channel_shift
from datakit.image import operators, pipeline_load, apply_to
from datakit import mnist, cifar
from datakit.helpers import dict_apply
from itertools import islice

import h5py
import os

def load_mnist(random_state=None, params=None):
    data = mnist.load()
    train_X_full = data['train']['X'] / 255.
    train_y_full = np_utils.to_categorical(data['train']['y'])
    test_X = data['test']['X'] / 255.
    test_y = np_utils.to_categorical(data['test']['y'])
    default_validation_size = 10000
    train_iterator, valid_iterator, test_iterator, info = build_iterators(
        train_X_full,
        train_y_full,
        test_X,
        test_y,
        valid_ratio=params['valid_ratio'],
        default_validation_size=10000,
        shuffle=params['shuffle'],
        random_state=random_state
    )
    return {'train': train_iterator, 'valid': valid_iterator, 'test': test_iterator, 'info': info}

def load_cifar(random_state=None, params=None):
    data = cifar.load(coarse_label=True)
    train_X_full = data['train']['X'] / 255.
    train_y_full = np_utils.to_categorical(data['train']['y'])
    test_X = data['test']['X'] / 255.
    test_y = np_utils.to_categorical(data['test']['y'])
    train_iterator, valid_iterator, test_iterator, info = build_iterators(
        train_X_full,
        train_y_full,
        test_X,
        test_y,
        valid_ratio=params['valid_ratio'],
        default_validation_size=5000,
        shuffle=params['shuffle'],
        random_state=random_state
    )
    return {'train': train_iterator, 'valid': valid_iterator, 'test': test_iterator, 'info': info}

def load_ilc(random_state=None, params=None):
    version = params.get('version', 'Ecal-0-12GeV-uniform-10374-Events')
    def get_X_and_y(dataset):
        X = dataset['X']
        X = X.reshape((X.shape[0], -1))
        X = X.reshape((X.shape[0], 18, 18, 30))
        X = X.transpose((0, 3, 1, 2))
        y = dataset['y']
        return X, y
    dataset = np.load('../ILC/data/{}/train.npz'.format(version))
    train_X_full, train_y_full = get_X_and_y(dataset)
    dataset = np.load('../ILC/data/{}/test.npz'.format(version))
    test_X, test_y = get_X_and_y(dataset)
    train_iterator, valid_iterator, test_iterator, info = build_iterators(
        train_X_full,
        train_y_full,
        test_X,
        test_y,
        valid_ratio=params['valid_ratio'],
        default_validation_size=10000,
        shuffle=params['shuffle'],
        random_state=random_state
    )
    return {'train': train_iterator, 'valid': valid_iterator, 'test': test_iterator, 'info': info}

def pipeline_load_hdf5(iterator, filename, cols=['X', 'y'], start=0, nb=None, batch_size=128):
    filename = os.path.join(os.getenv('DATA_PATH'), filename)
    hf = h5py.File(filename)
        
    def iter_func():
        for i in xrange(start, start + nb, batch_size):
            d = {}
            for c in cols:
                d[c] = hf[c][i:i+batch_size]
            for n in range(len(d[cols[0]])):
                p = {}
                for c in cols:
                    p[c] = d[c][n]
                yield p
    return iter_func()

def pipeline_load_numpy(iterator, filename, cols=['X', 'y'], start=0, nb=None, shuffle=False, random_state=None):
    rng = np.random.RandomState(random_state)
    filename = os.path.join(os.getenv('DATA_PATH'), filename)
    data = np.load(filename)
    if shuffle:
        indices = np.arange(len(data[cols[0]]))
        rng.shuffle(indices)
        data_shuffled = {}
        for c in cols:
            data_shuffled[c] = data[c][indices]
        data = data_shuffled
    return iterate(data, start=start, nb=nb, cols=cols)

def iterate(data, start=0, nb=None, cols=['X', 'y']):
    it = {}
    for c in cols:
        d = data[c]
        if nb:
            d = d[start:start+nb]
        else:
            d = d[start:]
        it[c] = iter(d)
    def iter_func():
        while True:
            d = {}
            for c in cols:
                d[c] = next(it[c])
            yield d
    return iter_func()

def pipeline_lambda(iterator, code, cols=['X']):
    fn = eval(code)
    iterator = imap(partial(dict_apply, fn=fn, cols=cols), iterator)
    return iterator

def random_padcrop(X, pad=4, rng=np.random):
    h, w = X.shape[1:]
    random_cropped = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
    for c in range(X.shape[0]):
        padded = skimage_pad(X[c], pad, mode='constant', constant_values=(0, 0))
        crops = rng.random_integers(0,high=2*pad,size=(2,))
        random_cropped[c, :, :] = padded[crops[0]:(crops[0]+h),crops[1]:(crops[1]+w)]
    return random_cropped

operators['load_hdf5'] = pipeline_load_hdf5
operators['load_numpy'] = pipeline_load_numpy
operators['lambda'] = pipeline_lambda
operators['random_padcrop'] = apply_to(random_padcrop, cols=['X'])
operators['random_rotation'] = apply_to(random_rotation, cols=['X'])
operators['random_shift'] = apply_to(random_shift, cols=['X'])
operators['random_zoom'] = apply_to(random_zoom, cols=['X'])
loader = partial(pipeline_load, operators=operators)
def pipeline_loader(random_state=None, params=None):
    train_pipeline = params['train']['pipeline']
    valid_pipeline = params['valid']['pipeline']
    test_pipeline = params['test']['pipeline']
    train = BatchIterator(lambda:loader(train_pipeline))
    valid = BatchIterator(lambda:loader(valid_pipeline))
    test = BatchIterator(lambda:loader(test_pipeline))
    info = {}
    # if big number of samples, you must use nb
    info['nb_train_samples'] = params['train']['nb'] if 'nb' in params['train'] else len(list(loader(train_pipeline[0:1])))
    info['nb_valid_samples'] = params['valid']['nb'] if 'nb' in params['valid'] else len(list(loader(valid_pipeline[0:1])))
    info['nb_test_samples'] = params['test']['nb'] if 'nb' in params['test'] else len(list(loader(test_pipeline[0:1])))
    sample = next(loader(params['train']['pipeline']))
    print(sample['X'].shape)
    info['input_shape'] = sample['X'].shape
    info['nb_outputs'] = len(sample['y'])
    return {'train': train, 'valid': valid, 'test': test, 'info': info}
 
data_loader = {
    'mnist': load_mnist,
    'cifar10': load_cifar,
    'ilc': load_ilc,
    'loader': pipeline_loader
}

def load_data(name,
              shuffle=True,
              valid_ratio=None,
              random_state=None,
              params=None):
    """
    if valid_ratio is None,
    take a good default, depending on dataset, and
    based on what people do in papers
    """
    info = {}
    if not params:
        params = {}
    params['valid_ratio'] = valid_ratio
    params['shuffle'] = shuffle
    data = data_loader[name](random_state=random_state, params=params)
    return data['train'], data['valid'], data['test'], data['info']

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

    train_iterator = ArrayBatchIterator(train_X, train_y,
                                   shuffle=shuffle,
                                   random_state=random_state)
    valid_iterator = ArrayBatchIterator(valid_X, valid_y,
                                   shuffle=False,
                                   random_state=random_state)
    test_iterator = ArrayBatchIterator(test_X, test_y,
                                  shuffle=False,
                                  random_state=random_state)
    info = {}
    info['nb_train_samples'] = train_X.shape[0]
    info['nb_valid_samples'] = valid_X.shape[0]
    info['nb_test_samples'] = test_X.shape[0]
    info['input_shape'] = train_X.shape[1:]
    info['nb_outputs'] = train_y.shape[1]
    return train_iterator, valid_iterator, test_iterator, info
