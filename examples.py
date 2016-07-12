from copy import deepcopy

default_optim = {
    'algo': 'SGD',
    'algo_params': {'lr': 0.01, 'momentum': 0.95},
    'patience': 1000,
    'patience_loss': 'val_acc',
    'nb_epoch': 1000,
    'batch_size': 128,
    'pred_batch_size': 1000,
    'l2': 0,
    'l1': 0,
    'lr_schedule': {
        'type': 'decrease_when_stop_improving',
        'loss': 'val_acc',
        'shrink_factor': 2,
        'patience': 10,
        'min_lr': 0.00001
    },
    'seed': 42,
    'budget_secs': 'inf'
}
small_test_cnn = {
    'model': {
        'name': 'vgg',
        'params': {
            'nb_filters': [64, 64],
            'size_blocks': [2, 2],
            'size_filters': 3,
            'stride': 2,
            'fc': [500],
            'fc_dropout': 0.5,
            'activation': 'relu',
        }
    },
    'optim': default_optim,
    'data': {
        'shuffle': True,
        'name': 'mnist',
        'prep_random_state': 1,
        'valid_ratio': None,
        'augmentation': {
            'horiz_flip': True,
            'vert_flip': False,
            'shear_range': 0,
            'rotation_range': 0,
            'zoom_range': 0
        }
    }
}

small_test_fc = deepcopy(small_test_cnn)
small_test_fc['model'] = {
    'name': 'fc',
    'params': {
        'nb_hidden': [500, 500],
        'activation': 'relu',
    }
}

small_test = deepcopy(small_test_fc)
small_test['optim']['budget_secs'] = 60 * 100


def random_optim(rng):
    optim = deepcopy(default_optim)
    algo = rng.choice(('Adam', 'RMSprop', 'SGD', 'Adadelta'))
    optim['algo'] = algo

    lr = rng.choice((0.1, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005))
    algo_params = {}
    algo_params['lr'] = lr
    if algo == 'SGD':
        momentum = rng.choice((0.5, 0.9, 0.95, 0.99, 0))
        nesterov = bool(rng.choice((True, False)))
        algo_params.update({'momentum': momentum, 'nesterov': nesterov})
    optim['algo_params'] = algo_params
    return optim


def random_data(rng, datasets=('mnist', 'cifar10')):
    return {'shuffle': True,
            'name': rng.choice(datasets),
            'prep_random_state': 1,
            'valid_ratio': None,
            'augmentation': {
                'horiz_flip': True,
                'vert_flip': False,
                'shear_range': 0,
                'rotation_range': 0,
                'zoom_range': 0}}


def model_vgg_A(fc=[4096, 4096]):
    return {'name': 'vgg',
            'params': {
                'nb_filters': [64, 128, 256, 512, 512],
                'size_blocks': [1, 1, 2, 2, 2],
                'size_filters': 3,
                'stride': 2,
                'fc': fc,
                'fc_dropout': 0.5,
                'activation': 'relu'}}


def model_vgg_B():
    return {'name': 'vgg',
            'params': {
                'nb_filters': [64, 128, 256, 512, 512],
                'size_blocks': [2, 2, 2, 2, 2],
                'size_filters': 3,
                'stride': 2,
                'fc': [4096, 4096],
                'fc_dropout': 0.5,
                'activation': 'relu'}}


def model_vgg_D(fc=[4096, 4096]):
    return {'name': 'vgg',
            'params': {
                'nb_filters': [64, 128, 256, 512, 512],
                'size_blocks': [2, 2, 3, 3, 3],
                'size_filters': 3,
                'stride': 2,
                'fc': fc,
                'fc_dropout': 0.5,
                'activation': 'relu'}}


def random_model_vgg(rng):
    stride = 2
    size_filters = rng.choice((2, 3))
    nb_blocks = rng.randint(1, 4)
    size_blocks = []
    nb_filters = []
    for i in range(nb_blocks):
        size_blocks.append(rng.randint(1, 5))
        nb_filters.append(2 ** rng.randint(4, 10))

    nb_fc = rng.randint(1, 4)
    fc = []
    for i in range(nb_fc):
        fc.append(2 ** rng.randint(9, 12))
    return {
        'name': 'vgg',
        'params': {
            'nb_filters': nb_filters,
            'size_blocks': size_blocks,
            'size_filters': size_filters,
            'stride': stride,
            'fc': fc,
            'fc_dropout': 0.5,
            'activation': random_activation(rng)}}


def random_activation(rng):
    return rng.choice(('relu', 'leaky_relu'))


def random_model(rng):
    return random_model_vgg(rng)

# real ones to use with where=

def vgg_D_optim_cifar(rng):
    optim = random_optim(rng)
    fc = 512
    model = model_vgg_D(fc=[fc, fc])
    data = random_data(rng, datasets=('cifar10',))
    return {'optim': optim, 'model': model, 'data': data}

def vgg_D_optim_cifar_24h(rng):
    optim = random_optim(rng)
    optim['budget_secs'] = 24 * 3600
    fc = 512
    model = model_vgg_D(fc=[fc, fc])
    data = random_data(rng, datasets=('cifar10',))
    return {'optim': optim, 'model': model, 'data': data}

def vgg_A_optim_cifar_24h(rng):
    optim = random_optim(rng)
    optim['budget_secs'] = 24 * 3600
    fc = 512
    model = model_vgg_A(fc=[fc, fc])
    data = random_data(rng, datasets=('cifar10',))
    return {'optim': optim, 'model': model, 'data': data}

def mini_random(rng):
    optim = random_optim(rng)
    model = random_model(rng)
    data = random_data(rng)
    optim['budget_secs'] = 3600
    return {'optim': optim, 'model': model, 'data': data}


def micro_random(rng):
    optim = random_optim(rng)
    model = random_model(rng)
    data = random_data(rng, datasets=('cifar10',))
    optim['budget_secs'] = 60 * 15
    return {'optim': optim, 'model': model, 'data': data}
