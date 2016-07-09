import click
from data import load_data

from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint

import models

from helpers import compute_metric, RecordEachEpoch

from tempfile import NamedTemporaryFile


@click.group()
def main():
    pass


@click.command()
@click.option('--dataset', default='mnust',
              help='dataset : mnist', required=False)
def train(dataset):
    train_model('mnist', 'fully',
                model_params={'nb_layers': 2,
                              'nb_hidden': 100,
                              'activation': 'relu'},
                optim_params={'algo': 'SGD',
                              'algo_params': {},
                              'patience': 5,
                              'nb_epoch': 100,
                              'batch_size': 128})


def train_model(dataset, model,
                model_params,
                optim_params,
                shuffle=True,
                valid_ratio=None,
                random_state=None):
    """
    model_params : nb hidden units, nb filters etc
    optim_params : learning rate, momentum etc
    """
    train_iterator, valid_iterator, test_iterator, info = load_data(
        dataset,
        shuffle=shuffle,
        valid_ratio=valid_ratio,
        random_state=random_state,
        batch_size=optim_params['batch_size'])

    input_shape = info['input_shape']
    nb_outputs = info['nb_outputs']
    model_builder = get_model_builder(model)
    model = model_builder(
        input_shape=input_shape,
        nb_outputs=nb_outputs,
        hp=model_params)

    optimizer = get_optimizer(optim_params['algo'])
    optimizer = optimizer(**optim_params['algo_params'])
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
    model_filename = NamedTemporaryFile(delete=False).name
    callbacks = record_callbacks + [
        EarlyStopping(monitor='val_acc',
                      patience=optim_params['patience'],
                      verbose=1,
                      mode='auto'),
        ModelCheckpoint(model_filename, monitor='val_acc',
                        verbose=1,
                        save_best_only=True, mode='auto')
    ]
    model.fit_generator(
        train_iterator.flow(repeat=True),
        nb_epoch=optim_params['nb_epoch'],
        samples_per_epoch=info['nb_train_samples'],
        callbacks=callbacks)
    model.load_weights(model_filename)
    test_acc = compute_metric(
        model,
        test_iterator.flow(repeat=False),
        metric='accuracy')
    print(test_acc)


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
    main.add_command(train)
    main()
