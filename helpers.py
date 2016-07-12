import numpy as np
import keras
from keras import backend as K
import time
from keras.layers.advanced_activations import LeakyReLU

import json


def floatX(X):
    return X.astype('float32')


class BatchIterator(object):

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
    preds = []
    for X, y in generator:
        preds.extend(compute_metric_on(model.predict(X), y, metric))
    return np.mean(preds)


def compute_metric_on(y_pred, y, metric='accuracy'):
    if metric == 'accuracy':
        return (y_pred.argmax(axis=1) == y.argmax(axis=1))
    else:
        raise Exception('Unknown metric : {}'.format(metric))


class RecordEachEpoch(keras.callbacks.Callback):

    def __init__(self, name, compute_fn):
        self.name = name
        self.compute_fn = compute_fn
        self.values = []

    def on_epoch_end(self, batch, logs={}):
        logs[self.name] = self.compute_fn()
        self.values.append(self.name)

eps = 1e-10


class LearningRateScheduler(keras.callbacks.Callback):
    def __init__(self,
                 type_='decrease_when_stop_improving',
                 shrink_factor=10, loss='train_acc', mode='max',
                 patience=1,
                 min_lr=0.00001):
        super(LearningRateScheduler, self).__init__()
        self.type = type_
        self.shrink_factor = shrink_factor
        self.loss = loss
        self.mode = mode
        self.patience = patience
        self.min_lr = min_lr

    def on_epoch_end(self, epoch, logs={}):
        assert hasattr(self.model.optimizer, 'lr'), \
            'Optimizer must have a "lr" attribute.'
        model = self.model
        old_lr = float(model.optimizer.lr.get_value())
        if epoch == 0:
            new_lr = old_lr
        elif self.type == 'constant':
            new_lr = old_lr
        elif self.type == 'decrease_when_stop_improving':
            if epoch < self.patience:
                new_lr = old_lr
            else:
                hist = model.history.history
                value_epoch = logs[self.loss]
                if self.mode == 'auto':
                    best = max if 'acc' in self.loss else min
                else:
                    best = {'max': max, 'min': min}[self.mode]
                arg_best = np.argmax if best == max else np.argmin
                best_index = arg_best(hist[self.loss])
                best_value = hist[self.loss][best_index]
                if ( best(value_epoch, best_value) == best_value and
                     epoch - best_index + 1 >= self.patience):
                    print('shrinking learning rate, loss : {},'
                          'prev best epoch : {}, prev best value : {},'
                          'current value: {}'.format(self.loss,
                                                     best_index + 1, best_value,
                                                     value_epoch))
                    new_lr = old_lr / self.shrink_factor
                else:
                    new_lr = old_lr
        else:
            raise Exception('Unknown lr schedule : {}'.format(self.type))
        new_lr = max(new_lr, self.min_lr)
        if abs(new_lr - old_lr) > eps:
            print('prev learning rate : {}, '
                  'new learning rate : {}'.format(old_lr, new_lr))
        K.set_value(self.model.optimizer.lr, new_lr)
        logs['lr'] = new_lr


class Show(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
        print('')
        for k, v in logs.items():
            print('{}:{:.5f}'.format(k, v))
        print('')


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
        logs = {k: float(v) for k, v in logs.items()}
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
