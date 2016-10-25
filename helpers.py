from __future__ import print_function
import os
import numpy as np
import keras
from keras import backend as K
import time
from keras.layers.advanced_activations import LeakyReLU

import json

from datakit.helpers import minibatch, expand_dict, dict_apply
from functools import partial
from itertools import imap, cycle

def floatX(X):
    return np.array(X).astype('float32')

def touch(filename):
    open(filename, 'w').close()

class BatchIterator(object):

    def __init__(self, iterator_func):
        self.iterator_func = iterator_func

    def flow(self, batch_size=128, repeat=True):
        iterator = self.iterator_func()
        iterator = minibatch(iterator, batch_size=batch_size)
        iterator = expand_dict(iterator)
        iterator = imap(partial(dict_apply, fn=floatX, cols=['X', 'y']), iterator)
        iterator = imap(lambda data: (data['X'], data['y']), iterator)
        if repeat:
            iterator = cycle(iterator)
        return iterator

class ArrayBatchIterator(object):

    def __init__(self, inputs, targets=None,
                 shuffle=False, random_state=None):
        self.inputs = inputs
        self.targets = targets
        self.shuffle = shuffle
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        if targets is not None:
            assert len(inputs) == len(targets)

    def flow(self, batch_size=128, repeat=True):
        while True:
            if self.shuffle:
                indices = np.arange(len(self.inputs))
                self.rng.shuffle(indices)
            for start_idx in range(0, len(self.inputs), batch_size):
                if self.shuffle:
                    excerpt = indices[start_idx:start_idx + batch_size]
                else:
                    excerpt = slice(start_idx, start_idx + batch_size)
                if self.targets is not None:
                    yield self.inputs[excerpt], self.targets[excerpt]
                else:
                    yield self.inputs[excerpt]
            if repeat is False:
                break


def apply_transformers(iterator, transformers, rng=np.random):
    for X, y in iterator:
        for t in transformers:
            X, y = t(X, y, rng=rng)
        yield X, y


def horiz_flip(X, y, rng=np.random):
    X_ = np.concatenate((X, X[:, :, :, ::-1]), axis=0)
    y_ = np.concatenate((y, y), axis=0)
    indices = np.arange(0, X_.shape[0])
    rng.shuffle(indices)
    indices = indices[0:len(X)]
    return X_[indices], y_[indices]


def vert_flip(X, y, rng=np.random):
    X_ = np.concatenate((X, X[:, :, ::-1, :]), axis=0)
    y_ = np.concatenate((y, y), axis=0)
    indices = np.arange(0, X_.shape[0])
    rng.shuffle(indices)
    indices = indices[0:len(X)]
    return X_[indices], y_[indices]


def input_only(gen):

    def gen_():
        for x, y in gen():
            yield x
    return gen_

def compute_metric(model, generator, metric='accuracy'):
    vals = []
    sizes = []
    for X, y in generator:
        vals.append(compute_metric_on(y, model.predict(X), metric))
        sizes.append(len(X))
    # TODO there is a bug in sizes here
    val = np.dot(vals, sizes) / np.sum(sizes)
    return val

def compute_metric_on(y, y_pred, metric='accuracy', backend=np):
    B = backend
    if metric == 'accuracy':
        return B.equal(y_pred.argmax(axis=1), y.argmax(axis=1)).mean()
    elif metric == 'mean_squared_error':
        return B.mean(((y_pred - y)**2), axis=1)
    else:
        raise Exception('Unknown metric : {}'.format(metric))

class RecordEachEpoch(keras.callbacks.Callback):

    def __init__(self, name, compute_fn, on_logs=True):
        self.name = name
        self.compute_fn = compute_fn
        self.values = []
        self.on_logs = on_logs

    def on_epoch_end(self, batch, logs={}):
        val = self.compute_fn()
        if self.on_logs:
            logs[self.name] = val
        self.values.append(val)

class RecordEachMiniBatch(keras.callbacks.Callback):

    def __init__(self, name, source):
        self.name = name
        self.source = source
        self.epoch_values = []
        self.epoch_sizes = []
        self.values = []
    
    def on_epoch_begin(self, batch, logs={}):
        self.epoch_values = []
        self.epoch_sizes = []

    def on_epoch_end(self, batch, logs={}):
        vals = self.epoch_values
        vals = np.array(vals)
        sizes = self.epoch_sizes
        sizes = np.array(sizes)
        sizes = sizes.reshape((sizes.shape[0],) + (1,) * (len(vals.shape) - 1) )
        val = (vals * sizes).sum() / np.sum(sizes)
        logs[self.name] = val
        self.values.append(val)
        del logs[self.source]

    def on_batch_end(self, batch, logs={}):
        self.epoch_values.append(logs[self.source])
        self.epoch_sizes.append(logs['size'])

eps = 1e-8
class LearningRateScheduler(keras.callbacks.Callback):
    def __init__(self,
                 name='decrease_when_stop_improving',
                 params=None):
        super(LearningRateScheduler, self).__init__()
        if params is None:
            params = {}
        self.name = name
        self.schedule_params = params
    
    def on_epoch_end(self, epoch, logs={}):
        assert hasattr(self.model.optimizer, 'lr'), \
            'Optimizer must have a "lr" attribute.'
        
        params = self.schedule_params

        model = self.model
        old_lr = float(model.optimizer.lr.get_value())
        if epoch == 0:
            new_lr = old_lr
        elif self.name == 'constant':
            new_lr = old_lr
        elif self.name == 'decrease_when_stop_improving':
            patience = params['patience']
            mode = params.get('mode', 'auto')
            loss = params['loss']
            shrink_factor = params['shrink_factor']
            if epoch < patience:
                new_lr = old_lr
            else:
                hist = model.history.history
                value_epoch = logs[loss]
                if mode == 'auto':
                    best = max if 'acc' in loss else min
                else:
                    best = {'max': max, 'min': min}[mode]
                arg_best = np.argmax if best == max else np.argmin
                best_index = arg_best(hist[loss])
                best_value = hist[loss][best_index]
                if ( best(value_epoch, best_value) == best_value and
                     epoch - best_index + 1 >= patience):
                    print('shrinking learning rate, loss : {},'
                          'prev best epoch : {}, prev best value : {},'
                          'current value: {}'.format(loss,
                                                     best_index + 1, best_value,
                                                     value_epoch))
                    new_lr = old_lr / shrink_factor
                else:
                    new_lr = old_lr
        elif self.name == 'decrease_every':
            every = params['every']
            shrink_factor = params['shrink_factor']
            if epoch % (every) == 0:
                new_lr = old_lr / shrink_factor
            else:
                new_lr = old_lr
        elif self.name == 'cifar':
            # source : https://github.com/gcr/torch-residual-networks/blob/master/train-cifar.lua#L181-L187
            if epoch == 80:
                new_lr = old_lr / 10.
            elif epoch == 120:
                new_lr = old_lr / 10.
            else:
                new_lr = old_lr
        elif self.name == 'manual':
            schedule = params['schedule']
            new_lr = old_lr
            for s in schedule:
                first, last = s['range']
                lr = s['lr']
                if epoch >= first and epoch <= last:
                    new_lr = lr
                    break
        else:
            raise Exception('Unknown lr schedule : {}'.format(self.name))
        min_lr = params.get('min_lr', 0)
        new_lr = max(new_lr, min_lr)
        if abs(new_lr - old_lr) > eps:
            print('prev learning rate : {}, '
                  'new learning rate : {}'.format(old_lr, new_lr))
        K.set_value(self.model.optimizer.lr, new_lr)
        logs['lr'] = new_lr

class Time(keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs={}):
        self.time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        logs['duration_sec'] = time.time() - self.time


class Show(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
        print('')
        for k, v in logs.items():
            print('{}:{:.5f}'.format(k, v))
        print('')


class Report(keras.callbacks.Callback):
    
    def __init__(self, func, name='',print_func=print):
        self.func = func
        self.name = name
        self.print_func = print_func

    def on_epoch_end(self, epoch, logs={}):
        self.print_func(self.name)
        self.print_func(self.func())

class LiveHistoryEpoch(keras.callbacks.Callback):

    def __init__(self, filename):
        self.filename = filename

    def on_epoch_end(self, epoch, logs={}):
        logs = {k: float(v) for k, v in logs.items()}
        with open(self.filename, 'a') as fd:
            fd.write(json.dumps(logs))
            fd.write('\n')


class LiveHistoryBatch(keras.callbacks.Callback):

    def __init__(self, filename):
        self.filename = filename

    def on_epoch_end(self, epoch, logs={}):
        with open(self.filename, 'a') as fd:
            fd.write('\n')

    def on_batch_end(self, epoch, logs={}):
        logs = {k: np.array(v).tolist() for k, v in logs.items()}
        with open(self.filename, 'a') as fd:
            fd.write(json.dumps(logs))
            fd.write('\n')


class BudgetFinishedException(Exception):
    pass


class TimeBudget(keras.callbacks.Callback):

    def __init__(self, budget_secs=float('inf')):
        self.start = time.time()
        self.budget_secs = budget_secs

    def on_epoch_end(self, epoch, logs={}):
        t = time.time()
        if t - self.start >= self.budget_secs:
            raise BudgetFinishedException()

leaky_relu = LeakyReLU(0.3)


def dispims_color(M, border=0, bordercolor=[0.0, 0.0, 0.0], shape = None, normalize=False):
    """ Display an array of rgb images. 
    The input array is assumed to have the shape numimages x numpixelsY x numpixelsX x 3
    """
    bordercolor = np.array(bordercolor)[None, None, :]
    numimages = len(M)
    M = M.copy()

    if normalize:
        for i in range(M.shape[0]):
            M[i] -= M[i].flatten().min()
            M[i] /= M[i].flatten().max()
    height, width, three = M[0].shape
    assert three == 3
    if shape is None:
        n0 = np.int(np.ceil(np.sqrt(numimages)))
        n1 = np.int(np.ceil(np.sqrt(numimages)))
    else:
        n0 = shape[0]
        n1 = shape[1]
        
    im = np.array(bordercolor)*np.ones(
                             ((height+border)*n1+border,(width+border)*n0+border, 1),dtype='<f8')
    for i in range(n0):
        for j in range(n1):
            if i*n1+j < numimages:
                im[j*(height+border)+border:(j+1)*(height+border)+border,
                   i*(width+border)+border:(i+1)*(width+border)+border,:] = np.concatenate((
                  np.concatenate((M[i*n1+j,:,:,:],
                         bordercolor*np.ones((height,border,3),dtype=float)), 1),
                  bordercolor*np.ones((border,width+border,3),dtype=float)
                  ), 0)
    return im

def named(func, name):
    func.__name__ = name
    return func

def mkdir_path(path):
    if not os.access(path, os.F_OK):
        os.makedirs(path)



