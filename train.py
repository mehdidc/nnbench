import numpy as np
from data import load_data

from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras import objectives
from keras import backend as K
from keras.backend.common import set_epsilon
from keras.layers import Activation, Input
from keras.models import Model

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


from tempfile import NamedTemporaryFile
import json

def train_model(params, outdir='out'):
    set_epsilon(0)
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
        random_state=data_preparation_random_state)
    
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
    else:
        model = Model(input=specs.input, output=specs.output)

    print('Number of parameters : {}'.format(model.count_params()))
    nb = sum(1 for layer in model.layers if hasattr(layer, 'W'))
    nb_W_params = sum(np.prod(layer.W.get_value().shape) for layer in model.layers if hasattr(layer, 'W'))
    print('Number of weight parameters : {}'.format(nb_W_params))
    print('Number of learnable layers : {}'.format(nb))

    optimizer = get_optimizer(algo_name)
    optimizer = optimizer(**algo_params)

    def loss_fn(y_true, y_pred):
        reg = 0
        if l1_coef: reg += l1_coef * sum(K.abs(layer.W).sum() for layer in model.layers if hasattr(layer, 'W'))
        if l2_coef: reg += l2_coef * sum((layer.W**2).sum() for layer in model.layers if hasattr(layer, 'W'))
        return get_loss(loss_function)(y_true, y_pred) + reg
    
    def get_loss(name):
        return getattr(objectives, name)

    model.compile(loss=loss_fn,
                  optimizer=optimizer)

    json_string = model.to_json()
    s = json.dumps(json.loads(json_string), indent=4)
    with open(os.path.join(outdir, 'model.json'), 'w') as fd:
        fd.write(s)

    # PREPROCESSING

    train_transformers = []
    test_transformers = []
    for prep in preprocessing:
        transformer = build_transformer(prep['name'], prep['params'], inputs=train_iterator.inputs)
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
    
    compute_train_metric = {}
    compute_valid_metric = {}
    compute_test_metric = {}
    for metric in metrics:
        compute_train_metric[metric] = compute_metric_fn(train_iterator, test_transformers , metric)
        compute_valid_metric[metric] = compute_metric_fn(valid_iterator, test_transformers , metric)
        compute_test_metric[metric]  = compute_metric_fn(test_iterator, test_transformers , metric)
    
    # compute train and valid metrics callbacks

    callbacks.append(Time())
    
    for metric in metrics:
        callbacks.extend([
            RecordEachEpoch(name='train_{}'.format(metric), compute_fn=compute_train_metric[metric]),
            RecordEachEpoch(name='val_{}'.format(metric), compute_fn=compute_valid_metric[metric])
        ])
            
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
    train_flow = apply_transformers(train_flow, train_transformers, rng=np.random)

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
        print('test {} : {}'.format(metric, value))
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

def build_transformer(name, params, inputs=None):

    if name == 'augmentation':
        return build_data_augmentation_transformer(**params)
    elif name == 'padcrop':
        return build_padcrop_transformer(**params)
    elif name == 'standardization':
        return build_standardization_transformer(inputs=inputs, **params)
    else:
        raise Exception('Unknown transformer : {}'.format(name))

def build_standardization_transformer(inputs):
    st = ImageDataGenerator(
        featurewise_center=True
    )
    st.fit(inputs)

    def fn(X, y, rng):
        for X_, y_ in st.flow(X, y, batch_size=X.shape[0]):
            #from skimage.io import imsave
            #imsave('out.png', X[0].transpose((1, 2, 0)))
            return X_, y_
    return fn


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

def build_padcrop_transformer(pad=4):

    def padcrop(X, y, rng):
        batchsize = X.shape[0]
        h, w = X.shape[2:]
        padded = np.pad(X,((0,0),(0,0),(pad,pad),(pad,pad)),mode='constant')
        random_cropped = np.zeros(X.shape, dtype=np.float32)
        crops = rng.random_integers(0,high=2*pad,size=(batchsize,2))
        for r in range(batchsize):
            random_cropped[r,:,:,:] = padded[r,:,crops[r,0]:(crops[r,0]+h),crops[r,1]:(crops[r,1]+w)]
        X = random_cropped
        return X, y
    return padcrop

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
