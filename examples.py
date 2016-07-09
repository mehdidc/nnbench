default_optim = {
    'algo': 'SGD',
    'algo_params': {'lr': 0.01, 'momentum': 0.95},
    'patience': 50,
    'patience_loss': 'val_acc',
    'nb_epoch': 1000,
    'batch_size': 128,
    'pred_batch_size': 1000,
    'lr_schedule': {
        'type': 'decrease_when_stop_improving',
        'loss': 'val_acc',
        'shrink_factor': 2,
        'patience': 10,
        'min_lr': 0.00001
    },
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
        'name': 'cifar10',
        'prep_random_state': 1,
        'valid_ratio': None,
        'horiz_flip': True
    }
}

small_test_fc = small_test_cnn.copy()
small_test_fc['model'] = {
    'name': 'fc',
    'params': {
        'nb_hidden': [500, 500],
        'activation': 'relu',
    }
}

small_test = small_test_cnn.copy()
small_test['optim']['budget_secs'] = 60 * 100


def random_data(rng):
    datasets = ('mnist', 'cifar10')
    return {'shuffle': True,
            'name': rng.choice(datasets),
            'prep_random_state': 1,
            'valid_ratio': None,
            'horiz_flip': True}


def model_vgg_A():
    return {'name': 'vgg',
            'params': {
                'nb_filters': [64, 128, 256, 512, 512],
                'size_blocks': [1, 1, 2, 2, 2],
                'size_filters': 3,
                'stride': 2,
                'fc': [4096, 4096],
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


def model_vgg_D():
    return {'name': 'vgg',
            'params': {
                'nb_filters': [64, 128, 256, 512, 512],
                'size_blocks': [2, 2, 3, 3, 3],
                'size_filters': 3,
                'stride': 2,
                'fc': [4096, 4096],
                'fc_dropout': 0.5,
                'activation': 'relu'}}


def random_model_vgg(rng):
    stride = 2
    size_filters = rng.choice((2, 3))
    nb_blocks = rng.randint(2, 6)
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
