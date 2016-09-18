from data import load_data

from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

import os

import models

from helpers import (
    compute_metric,
    RecordEachEpoch, LearningRateScheduler,
    Show,
    apply_transformers, TimeBudget, BudgetFinishedException,
    LiveHistoryBatch, LiveHistoryEpoch,
    Time,
    touch)

import numpy as np
from tempfile import NamedTemporaryFile


def train_model(params, outdir='out'):
    optim_params = params['optim']
    data_params = params['data']
    model_params = params['model']
    
    # RETRIEVE PARAMS

    # data params

    preprocessing = data_params['preprocessing']
    data_preparation_random_state = data_params['seed']
    shuffle = data_params['shuffle']
    valid_ratio = data_params['valid_ratio']
    dataset_name = data_params['name']


    # optim params

    nb_epoch = optim_params['nb_epoch']

    algo = optim_params['algo']
    algo_name = algo['name']
    algo_params = algo['params']

    batch_size = optim_params['batch_size']
    pred_batch_size = optim_params['pred_batch_size']

    regularization = optim_params['regularization']
    l1_coef = regularization['l1']
    l2_coef = regularization['l2']

        
    lr_schedule = optim_params['lr_schedule']
    lr_schedule_name = lr_schedule['name']
    lr_schedule_params = lr_schedule['params']

    early_stopping = optim_params['early_stopping']
    early_stopping_name = early_stopping['name']
    early_stopping_params = early_stopping['params']

    checkpoint = optim_params['checkpoint']

    seed = optim_params['seed']
    budget_secs = float(optim_params['budget_secs'])
    
    # model params
    model_name = model_params['name']
    model_params_ = model_params['params']
    
    # PREPARE DATA

    train_iterator, valid_iterator, test_iterator, info = load_data(
        dataset_name,
        shuffle=shuffle,
        valid_ratio=valid_ratio,
        random_state=data_preparation_random_state)
    
    # COMPILE MODEL

    np.random.seed(seed)
    input_shape = info['input_shape']
    nb_outputs = info['nb_outputs']
    model_builder = get_model_builder(model_name)
    model = model_builder(
        model_params_,
        input_shape=input_shape,
        nb_outputs=nb_outputs)

    optimizer = get_optimizer(algo_name)
    optimizer = optimizer(**algo_params)

    def loss_fn(y_true, y_pred):
        reg = 0
        if l1_coef: reg += l1_coef * sum(K.abs(layer.W).sum() for layer in model.layers if hasattr(layer, 'W'))
        if l2_coef: reg += l2_coef * sum((layer.W**2).sum() for layer in model.layers if hasattr(layer, 'W'))
        return K.categorical_crossentropy(y_pred, y_true) + reg

    model.compile(loss=loss_fn,
                  optimizer=optimizer)

    # PREPROCESSING

    train_transformers = []
    test_transformers = []
    for prep in preprocessing:
        transformer = build_transformer(prep['name'], prep['params'])
        only_train = prep.get('only_train', False)
        if only_train:
            train_transformers.append(transformer)
        else:
            train_transformers.append(transformer)
            test_transformers.append(transformer)
    
    def compute_metric_fn(iterator, transformers, metric):
        def fn():
            flow = iterator.flow(repeat=False, batch_size=pred_batch_size)
            flow = apply_transformers(flow, transformers)
            value = compute_metric(
                model,
                flow,
                metric=metric
            )
            return value
        return fn
    
    ## CALLBACKS
    callbacks = []

    compute_train_accuracy = compute_metric_fn(train_iterator, train_transformers, 'accuracy')
    compute_valid_accuracy = compute_metric_fn(valid_iterator, test_transformers , 'accuracy')
    compute_test_accuracy = compute_metric_fn(test_iterator  , test_transformers , 'accuracy')
    
    # compute train and valid accuracy callbacks

    callbacks.extend([
        RecordEachEpoch(name='train_acc', compute_fn=compute_train_accuracy),
        RecordEachEpoch(name='val_acc', compute_fn=compute_valid_accuracy)
    ])
        
    # Epoch duration time measure callback
    callbacks.append(Time())
    
    # Early stopping callback
    callbacks.extend(
        build_early_stopping_callbacks(name=early_stopping_name, params=early_stopping_params, outdir=outdir)
     )

    # Checkpoint callback
    model_filename = os.path.join(outdir, 'model.pkl')
    callbacks.append(
        build_model_checkpoint_callback(model_filename=model_filename, params=checkpoint)
    )
    
    # lr schedule callback
    callbacks.append(
        build_lr_schedule_callback(name=lr_schedule_name, params=lr_schedule_params)
    )
    # the rest of callbacks
    live_epoch_filename = os.path.join(outdir, 'epoch')
    live_batch_filename = os.path.join(outdir, 'batch')
    touch(live_epoch_filename)
    touch(live_batch_filename)

    callbacks.extend([
        Show(),
        LiveHistoryBatch(live_batch_filename),
        LiveHistoryEpoch(live_epoch_filename),
        TimeBudget(budget_secs)
    ])

    print('Number of parameters : {}'.format(model.count_params()))
    nb = sum(1 for layer in model.layers if hasattr(layer, 'W'))
    print('Number of learnable layers : {}'.format(nb))

    train_flow = train_iterator.flow(repeat=True, batch_size=batch_size)
    train_flow = apply_transformers(train_flow, train_transformers, rng=np.random)

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
    test_acc = compute_test_accuracy()
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

def build_transformer(name, params):

    if name == 'augmentation':
        return build_data_augmentation_transformer(**params)
    else:
        raise Exception('Unknown transformer : {}'.format(name))

def build_data_augmentation_transformer(rotation_range=0, 
                                        shear_range=0, 
                                        zoom_range=0, 
                                        horizontal_flip=False, 
                                        vertical_flip=False,
                                        width_shift_range=0,
                                        height_shift_range=0):
    data_augment = ImageDataGenerator(
        rotation_range=rotation_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range)
    
    def augment(X, y, rng):
        for X_, y_ in data_augment.flow(X, y, batch_size=X.shape[0]):
            return X_, y_
    return augment

def build_early_stopping_callbacks(name, params, outdir='out'):
    if name == 'basic':
        patience_loss = params['patience_loss']
        patience = params['patience']
        callback = EarlyStopping(monitor=patience_loss,
                                 patience=patience,
                                 verbose=1,
                                 mode='auto')
        return [callback]
    elif name == 'none':
        return []

def build_model_checkpoint_callback(params, model_filename='model.pkl'):
    loss = params['loss']
    save_best_only = params['save_best_only']
    return ModelCheckpoint(model_filename, 
                           monitor=loss,
                           verbose=1,
                           save_best_only=save_best_only,
                           mode='auto' if loss else 'min')

def build_lr_schedule_callback(name, params):
    return LearningRateScheduler(name=name, params=params)
