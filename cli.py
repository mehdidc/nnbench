import click
from data import load_data

from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint

import models

from helpers import compute_metric, RecordEachEpoch, LearningRateScheduler, Show

import numpy as np
from tempfile import NamedTemporaryFile
import time

import examples


@click.group()
def main():
    pass


@click.command()
def smalltest():
    np.random.seed(1)
    model, hist = train_model(examples.small_test)


def train_model(params):
    optim_params = params['optim']
    data_params = params['data']
    model_params = params['model']

    shuffle = data_params['shuffle']
    valid_ratio = data_params['valid_ratio']
    dataset_name = data_params['name']
    data_preparation_random_state = data_params['prep_random_state']

    batch_size = optim_params['batch_size']
    pred_batch_size = optim_params['pred_batch_size']
    patience = optim_params['patience']
    patience_loss = optim_params['patience_loss']
    algo = optim_params['algo']
    algo_params = optim_params['algo_params']
    nb_epoch = optim_params['nb_epoch']
    lr_schedule = optim_params['lr_schedule']
    lr_schedule_type = lr_schedule['type']
    lr_schedule_shrink_factor = lr_schedule['shrink_factor']
    lr_schedule_loss = lr_schedule['loss']
    lr_schedule_patience = lr_schedule['patience']
    min_lr = lr_schedule['min_lr']

    model_name = model_params['name']
    model_params_ = model_params['params']

    train_iterator, valid_iterator, test_iterator, info = load_data(
        dataset_name,
        shuffle=shuffle,
        valid_ratio=valid_ratio,
        random_state=data_preparation_random_state)

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

    def compute_train_accuracy():
        train_acc = compute_metric(
            model,
            train_iterator.flow(repeat=False, batch_size=pred_batch_size),
            metric='accuracy'
        )
        return train_acc

    def compute_valid_accuracy():
        val_acc = compute_metric(
            model,
            valid_iterator.flow(repeat=False, batch_size=pred_batch_size),
            metric='accuracy'
        )
        return val_acc

    record_callbacks = [
        RecordEachEpoch(name='train_acc', compute_fn=compute_train_accuracy),
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
                        save_best_only=True, mode='auto'),
        LearningRateScheduler(type_=lr_schedule_type,
                              shrink_factor=lr_schedule_shrink_factor,
                              loss=lr_schedule_loss,
                              patience=lr_schedule_patience,
                              mode='auto',
                              min_lr=min_lr),
        Show()
    ]
    hist = model.fit_generator(
        train_iterator.flow(repeat=True, batch_size=batch_size),
        nb_epoch=nb_epoch,
        samples_per_epoch=info['nb_train_samples'],
        callbacks=callbacks,
        verbose=2)
    model.load_weights(model_filename)
    test_acc = compute_metric(
        model,
        test_iterator.flow(repeat=False, batch_size=pred_batch_size),
        metric='accuracy')
    hist.test_acc = test_acc
    print('test acc : {}'.format(test_acc))
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
