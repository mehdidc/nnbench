import numpy as np
import keras


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
