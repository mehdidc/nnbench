import click
from data import load_data

from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint

import models

from helpers import compute_metric, RecordEachEpoch

import numpy as np
from tempfile import NamedTemporaryFile


@click.group()
def main():
    pass


@click.command()
def smalltest():
    np.random.seed(1)
    model_params = {
        'name': 'vgg',
        'params': {
            'nb_filters': [64, 64],
            'size_filters': 3,
            'stride': 2,
            'size_blocks': [3, 3],
            'fc': [200],
            'activation': 'relu'
        }
    }
    optim_params = {
        'algo': 'SGD',
        'algo_params': {},
        'patience': 5,
        'nb_epoch': 10,
        'batch_size': 128,
        'patience_loss': 'val_acc'
    }
    data_params = {
        'shuffle': True,
        'name': 'mnist',
        'prep_random_state': 1,
        'valid_ratio': None
    }
    params = {
        'optim': optim_params,
        'data': data_params,
        'model': model_params
    }
    model, hist = train_model(params)


def train_model(params):
    optim_params = params['optim']
    data_params = params['data']
    model_params = params['model']

    shuffle = data_params['shuffle']
    valid_ratio = data_params['valid_ratio']
    dataset_name = data_params['name']
    data_preparation_random_state = data_params['prep_random_state']

    batch_size = optim_params['batch_size']
    patience = optim_params['patience']
    patience_loss = optim_params['patience_loss']
    algo = optim_params['algo']
    algo_params = optim_params['algo_params']
    nb_epoch = optim_params['nb_epoch']

    model_name = model_params['name']
    model_params_ = model_params['params']

    train_iterator, valid_iterator, test_iterator, info = load_data(
        dataset_name,
        shuffle=shuffle,
        valid_ratio=valid_ratio,
        random_state=data_preparation_random_state,
        batch_size=batch_size)

    input_shape = info['input_shape']
    nb_outputs = info['nb_outputs']
    model_builder = get_model_builder(model_name)
    model = model_builder(
        model_params_,
        input_shape=input_shape,
        nb_outputs=nb_outputs)

    optimizer = get_optimizer(algo)
    optimizer = optimizer(**algo_params)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer)

    def compute_valid_accuracy():
        val_acc = compute_metric(
            model,
            valid_iterator.flow(repeat=False),
            metric='accuracy'
        )
        return val_acc

    record_callbacks = [
        RecordEachEpoch(name='val_acc', compute_fn=compute_valid_accuracy)
    ]
    model_filename = NamedTemporaryFile(prefix='nnbench_', delete=False).name
    callbacks = record_callbacks + [
        EarlyStopping(monitor=patience_loss,
                      patience=patience,
                      verbose=1,
                      mode='auto'),
        ModelCheckpoint(model_filename, monitor=patience_loss,
                        verbose=1,
                        save_best_only=True, mode='auto')
    ]
    hist = model.fit_generator(
        train_iterator.flow(repeat=True),
        nb_epoch=nb_epoch,
        samples_per_epoch=info['nb_train_samples'],
        callbacks=callbacks)
    model.load_weights(model_filename)
    test_acc = compute_metric(
        model,
        test_iterator.flow(repeat=False),
        metric='accuracy')
    print(test_acc)
    return model, hist


def get_optimizer(name):
    if hasattr(optimizers, name):
        return getattr(optimizers, name)
    else:
        raise Exception('unknown optimizer : {}'.format(name))


def get_model_builder(model_name):
    if hasattr(models, model_name):
        model = getattr(models, model_name)
        return model
    else:
        raise Exception('unknown model : {}'.format(model_name))


if __name__ == '__main__':
    main.add_command(smalltest)
    main()
