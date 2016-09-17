from data import load_data

from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

import os

import models

from helpers import (
    compute_metric,
    RecordEachEpoch, LearningRateScheduler,
    Show,
    apply_transformers, TimeBudget, BudgetFinishedException,
    LiveHistoryBatch, LiveHistoryEpoch,
    Time)

import numpy as np
from tempfile import NamedTemporaryFile


def train_model(params, outdir=None):
    optim_params = params['optim']
    data_params = params['data']
    model_params = params['model']

    shuffle = data_params['shuffle']
    valid_ratio = data_params['valid_ratio']
    dataset_name = data_params['name']
    data_preparation_random_state = data_params['prep_random_state']
    zca = data_params.get('use_zca', False)

    augmentation = data_params['augmentation']
    use_horiz_flip = augmentation['horiz_flip']
    use_vert_flip = augmentation['vert_flip']
    shear_range = augmentation['shear_range']
    zoom_range = augmentation['zoom_range']
    rotation_range = augmentation['rotation_range']

    seed = optim_params['seed']
    np.random.seed(seed)
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
    budget_secs = float(optim_params['budget_secs'])

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
        train_flow = train_iterator.flow(repeat=False, batch_size=pred_batch_size),
        #train_flow = apply_transformers(train_flow, [prep])
        train_acc = compute_metric(
            model,
            train_flow,
            metric='accuracy'
        )
        return train_acc

    def compute_valid_accuracy():
        valid_flow = valid_iterator.flow(repeat=False, batch_size=pred_batch_size)
        #valid_flow = apply_transformers(valid_flow, [prep])
        val_acc = compute_metric(
            model,
            valid_flow,
            metric='accuracy'
        )
        return val_acc

    record_callbacks = [
        RecordEachEpoch(name='train_acc', compute_fn=compute_train_accuracy),
        RecordEachEpoch(name='val_acc', compute_fn=compute_valid_accuracy)
    ]

    if outdir is None:
        model_filename = NamedTemporaryFile(
            prefix='nnbench_model_', delete=False).name
    else:
        try:
            os.mkdir(outdir)
        except OSError:
            pass
        model_filename = os.path.join(outdir, 'model.pkl')
    live_epoch_filename = os.path.join(outdir, 'epoch')
    open(live_epoch_filename, 'w').close()
    live_batch_filename = os.path.join(outdir, 'batch')
    open(live_batch_filename, 'w').close()

    callbacks = record_callbacks + [Time()]

    if patience_loss:
        callbacks.extend([
            EarlyStopping(monitor=patience_loss,
                          patience=patience,
                          verbose=1,
                          mode='auto'),
         ])
    
    callbacks.append(
        ModelCheckpoint(model_filename, monitor=patience_loss,
                        verbose=1,
                        save_best_only=True if patience_loss else False, 
                        mode='auto' if patience_loss else 'min')
    )
   
    if lr_schedule_loss:
        callbacks.extend([
        LearningRateScheduler(type_=lr_schedule_type,
                              shrink_factor=lr_schedule_shrink_factor,
                              loss=lr_schedule_loss,
                              patience=lr_schedule_patience,
                              mode='auto',
                              min_lr=min_lr)
        ])
    callbacks.extend([
        Show(),
        LiveHistoryBatch(live_batch_filename),
        LiveHistoryEpoch(live_epoch_filename),
        TimeBudget(budget_secs)
    ])

    train_flow = train_iterator.flow(repeat=True, batch_size=batch_size)

    data_prep = ImageDataGenerator(
        zca_whitening=zca
    )
    data_prep.fit(train_iterator.inputs)

    data_augment = ImageDataGenerator(
        rotation_range=rotation_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=use_horiz_flip,
        vertical_flip=use_vert_flip)
    
    def prep(X, y, rng):
        for X_, y_ in data_prep.flow(X, y, batch_size=X.shape[0]):
            return X_, y_

    def augment(X, y, rng):
        for X_, y_ in data_augment.flow(X, y, batch_size=X.shape[0]):
            return X_, y_

    transformers = []
    transformers.append(augment)
    transformers.append(prep)
    train_flow = apply_transformers(train_flow, transformers, rng=np.random)
    
    print('Number of parameters : {}'.format(model.count_params()))
    try:
        model.fit_generator(
            train_flow,
            nb_epoch=nb_epoch,
            samples_per_epoch=info['nb_train_samples'],
            callbacks=callbacks,
            verbose=2)
    except BudgetFinishedException:
        pass

    model.load_weights(model_filename)
    
    test_flow = test_iterator.flow(repeat=False, batch_size=pred_batch_size),
    test_flow = apply_transformers(test_flow, [prep])
    test_acc = compute_metric(
        model,
        test_flow,
        metric='accuracy')
    model.history.final = {'test_acc': test_acc}
    print('test acc : {}'.format(test_acc))

    os.remove(live_batch_filename)
    os.remove(live_epoch_filename)
    return model


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
