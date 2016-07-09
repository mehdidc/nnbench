import click
from data import load_data

from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

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
            'size_blocks': [2, 2],
            'fc': [200],
            'activation': 'relu'
        }
    }
    optim_params = {
        'algo': 'SGD',
        'algo_params': {'lr': 0.01, 'momentum': 0.9},
        'patience': 5,
        'nb_epoch': 10,
        'batch_size': 128,
        'pred_batch_size': 1000,
        'patience_loss': 'val_acc',
        'lr_schedule': {
            'type': 'decrease_when_stop_improving',
            'loss': 'train_acc',
            'shrink_factor': 10,
            'patience': 1,
            'min_lr': 0.00001
        }
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
        print('\ntrain acc : {}'.format(train_acc))
        return train_acc

    def compute_valid_accuracy():
        val_acc = compute_metric(
            model,
            valid_iterator.flow(repeat=False, batch_size=pred_batch_size),
            metric='accuracy'
        )
        print('val acc : {}'.format(val_acc))
        return val_acc

    def lr_scheduler(epoch):
        print(epoch)
        if epoch == 0:
            return float(model.optimizer.lr.get_value())
        if lr_schedule_type == 'decrease_when_stop_improving':
            old_lr = float(model.optimizer.lr.get_value())
            if epoch < lr_schedule_patience:
                new_lr = old_lr
            else:
                print(len(model.history[lr_schedule_loss]))
                value_epoch = model.history.history[lr_schedule_loss][epoch]
                value_epoch_past = model.history.history[lr_schedule_loss][epoch - lr_schedule_patience]
                if value_epoch >= value_epoch_past:
                    new_lr = old_lr / lr_schedule_shrink_factor
                    print('prev learning rate : {}, new learning rate : {}'.format(old_lr, new_lr))
                else:
                    new_lr = model.lr.get_value()
        else:
            raise Exception('Unknown lr schedule : {}'.format(lr_schedule_type))
        new_lr = max(new_lr, min_lr)
        return new_lr

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
        LearningRateScheduler(lr_scheduler),
    ]
    hist = model.fit_generator(
        train_iterator.flow(repeat=True, batch_size=batch_size),
        nb_epoch=nb_epoch,
        samples_per_epoch=info['nb_train_samples'],
        callbacks=callbacks)
    model.load_weights(model_filename)
    test_acc = compute_metric(
        model,
        test_iterator.flow(repeat=False, batch_size=pred_batch_size),
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
