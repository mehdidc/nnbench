import numpy as np
import keras
from keras import backend as K


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
        elif self.type == 'decrease_when_stop_improving':
            if epoch < self.patience:
                new_lr = old_lr
            else:
                value_epoch = logs[self.loss]
                hist = model.history.history
                value_epoch_past = hist[self.loss][epoch - self.patience]
                if self.mode == 'min':
                    cond = value_epoch <= value_epoch_past
                elif self.mode == 'max':
                    cond = value_epoch >= value_epoch_past
                elif self.mode == 'auto':
                    if 'acc' in self.loss:
                        cond = cond = value_epoch >= value_epoch_past
                    else:
                        cond = cond = value_epoch >= value_epoch_past
                else:
                    cond = value_epoch >= value_epoch_past
                if not cond:
                    new_lr = old_lr / self.shrink_factor
                else:
                    new_lr = old_lr
        else:
            raise Exception('Unknown lr schedule : {}'.format(self.type))
        new_lr = max(new_lr, self.min_lr)
        if new_lr != old_lr:
            print('prev learning rate : {}, '
                  'new learning rate : {}'.format(old_lr, new_lr))
        K.set_value(self.model.optimizer.lr, new_lr)


class Show(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
        print('')
        for k, v in logs.items():
            print('{}:{:.5f}'.format(k, v))
        print('')
