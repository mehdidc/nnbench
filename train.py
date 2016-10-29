import sys
import os
import json
import logging
import time
from functools import partial
import pprint

import numpy as np
from data import load_data
from skimage.io import imsave
from sklearn.metrics import confusion_matrix, roc_auc_score

from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import objectives
from keras import backend as K
from keras.backend.common import set_epsilon
from keras.layers import Activation, Input
from keras.models import Model

import models

from helpers import (
    compute_metric,
    compute_metric_on,
    RecordEachEpoch, RecordEachMiniBatch,
    LearningRateScheduler,
    Show, TimeBudget, BudgetFinishedException,
    LiveHistoryBatch, LiveHistoryEpoch,
    Time, Report, mkdir_path,
    touch, dispims_color, named)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def train_model(params, outdir='out'):
    set_epsilon(0)
    optim_params = params['optim']
    data_params = params['data']
    model_params = params['model']
    outdir = params.get('outdir', outdir)
    mkdir_path(outdir) 
    # RETRIEVE PARAMS
    # data params
    data_preparation_random_state = data_params['seed']
    shuffle = data_params['shuffle']
    valid_ratio = data_params['valid_ratio']
    dataset_name = data_params['name']
    data_loader_params = data_params['params']
    # optim params

    nb_epoch = optim_params['nb_epoch']

    algo = optim_params['algo']
    algo_name = algo['name']
    algo_params = algo['params']

    loss_function = optim_params['loss']
    metrics = optim_params['metrics']

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
    np.random.seed(seed)
    budget_secs = float(optim_params['budget_secs'])
    
    # model params
    model_name = model_params['name']
    model_params_ = model_params['params']
    
    # PREPARE DATA

    train_iterator, valid_iterator, test_iterator, info = load_data(
        dataset_name,
        shuffle=shuffle,
        valid_ratio=valid_ratio,
        random_state=data_preparation_random_state,
        params=data_loader_params)
    
    # BUILD AND COMPILE MODEL
    input_shape = info['input_shape']
    nb_outputs = info['nb_outputs']
    model_builder = get_model_builder(model_name)
    specs = model_builder(
        model_params_,
        input_shape=input_shape,
        nb_outputs=nb_outputs)

    # for classification add a softmax nonlinearity
    if loss_function in ('categorical_crossentropy',):
        inp, out = specs.input, specs.output
        out = Activation('softmax')(out)
        model = Model(input=inp, output=out)
    elif loss_function in ('multilabel', 'binary_crossentropy'):
        inp, out = specs.input, specs.output
        out = Activation('sigmoid')(out)
        model = Model(input=inp, output=out)
    else:
        model = Model(input=specs.input, output=specs.output)

    logger.info('Number of parameters : {}'.format(model.count_params()))
    nb = sum(1 for layer in model.layers if hasattr(layer, 'W'))
    nb_W_params = sum(np.prod(layer.W.get_value().shape) for layer in model.layers if hasattr(layer, 'W'))
    logger.info('Number of weight parameters : {}'.format(nb_W_params))
    logger.info('Number of learnable layers : {}'.format(nb))

    model.summary()

    optimizer = get_optimizer(algo_name)
    optimizer = optimizer(**algo_params)

    def loss_fn(y_true, y_pred):
        reg = 0
        if l1_coef: reg += l1_coef * sum(K.abs(layer.W).sum() for layer in model.layers if hasattr(layer, 'W'))
        if l2_coef: reg += l2_coef * sum((layer.W**2).sum() for layer in model.layers if hasattr(layer, 'W'))
        return get_loss(loss_function)(y_true, y_pred) + reg

    minibatch_metrics = [named(partial(compute_metric_on, metric=metric, backend=K), name=metric) for metric in metrics] 
    model.compile(loss=loss_fn,
                  optimizer=optimizer,
                  metrics=minibatch_metrics)
    json_string = model.to_json()
    s = json.dumps(json.loads(json_string), indent=4)
    with open(os.path.join(outdir, 'model.json'), 'w') as fd:
        fd.write(s)

    # PREPROCESSING
    def compute_metric_fn(iterator, metric, name=''):
        def fn():
            logger.debug('Computing {} {}...'.format(metric, name))
            t = time.time()
            flow = iterator.flow(repeat=False, batch_size=pred_batch_size)
            value = compute_metric(
                model,
                flow,
                metric=metric
            )
            delta_t = time.time() - t
            logger.debug('Computing {} {} took {:.3f} secs.'.format(metric, name, delta_t))
            return value
        return fn
    ## CALLBACKS
    callbacks = []
    
    compute_train_metric = {}
    compute_valid_metric = {}
    compute_test_metric = {}
    for metric in metrics:
        compute_train_metric[metric] = compute_metric_fn(train_iterator, metric, name='train')
        compute_valid_metric[metric] = compute_metric_fn(valid_iterator, metric, name='valid')
        compute_test_metric[metric]  = compute_metric_fn(test_iterator,  metric, name='test')
    
    # compute train and valid metrics/scores callbacks

    callbacks.append(Time())
    
    for metric in metrics:
        each_minibatch = RecordEachMiniBatch(name='train_{}'.format(metric), source=metric)
        each_epoch = RecordEachEpoch(name='val_{}'.format(metric), compute_fn=compute_valid_metric[metric])
        callbacks.extend([each_minibatch, each_epoch])
    
    fn = partial(compute_confusion_matrix, iterator=train_iterator, model=model, batch_size=pred_batch_size)
    callbacks.append(Report(fn, name='train_confusion_matrix'))

    fn = partial(compute_confusion_matrix, iterator=valid_iterator, model=model, batch_size=pred_batch_size)
    callbacks.append(Report(fn, name='valid_confusion_matrix'))
 
    # Epoch duration time measure callback
    
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
    
    train_flow = train_iterator.flow(repeat=True, batch_size=batch_size)
    x0, y0 = next(train_iterator.flow())
    logger.debug('Shape of x : {}'.format(x0.shape))
    logger.debug('Shape of y : {}'.format(y0.shape))
    logger.debug('Min of x : {}, Max of x : {}'.format(x0.min(), x0.max()))

    img = dispims_color(x0.transpose((0, 2, 3, 1)) * np.ones((1, 1, 1, 3)), border=1, bordercolor=(0.3, 0, 0))
    imsave(outdir+'/data.png', img)
    try:
        model.fit_generator(
            train_flow,
            nb_epoch=nb_epoch,
            samples_per_epoch=info['nb_train_samples'],
            callbacks=callbacks,
            max_q_size=10,
            verbose=0)
    except BudgetFinishedException:
        pass
    model.load_weights(model_filename)
    model.history.final = {}
    for metric in metrics:
        value = compute_test_metric[metric]()
        model.history.final['test_' + metric] = value
        logger.info('test {} : {}'.format(metric, value))
    return model

def multilabel(y_true, y_pred):
    mask = 1 - 2 * y_true # transform [1 0 0 0] into [-1 1 1 1 1]
    loss = (mask * K.log(y_pred)) # make the true class prob. bigger, make the others prob. smaller
    return loss.mean()

def get_loss(name):
    if name == 'multilabel':
        return multilabel
    return getattr(objectives, name)


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

def compute_auc(iterator, model, batch_size=128):
    flow = iterator.flow(repeat=False, batch_size=batch_size)
    preds = []
    reals = []
    for X, y in flow:
        y_pred = model.predict(X)
        reals.append(y)
        preds.append(y_pred)
    preds = np.concatenate(preds, axis=0)
    reals = np.concatenate(reals, axis=0)
    nb_classes = preds.shape[1]
    nb = 0
    score = 0
    for c in range(nb_classes):
        real_positive = reals[reals.argmax(axis=1) == c, c]
        if real_positive.sum() == 0: # only existing classes are considered
            continue
        real_negative = reals[reals.argmax(axis=1) != c, c]
        preds_positive = preds[real_positive, c]
        preds_negative = 1 - preds[real_negative, c]
        r_all = np.concatenate((real_positive, real_positive), axis=0)
        p_all = np.concatenate((preds_positive, preds_negative), axis=0)
        print(r_all.shape, p_all.shape)
        s = roc_auc_score(r_all, p_all)
        print(s)
        score += s 
        nb += 1
    score /= nb
    return score

def compute_confusion_matrix(iterator, model, batch_size=128):
    flow = iterator.flow(repeat=False, batch_size=batch_size)
    m = None
    for X, y in flow:
        y_pred = model.predict(X).argmax(axis=1)
        y = y.argmax(axis=1)
        m = (0 if m is None else m) + confusion_matrix(y, y_pred)
    if m is not None:m = m.astype(np.int32)
    return m
