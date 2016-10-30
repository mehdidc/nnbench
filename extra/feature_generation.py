import numpy as np
import os

optim = {
    "loss": "categorical_crossentropy",
    "metrics": ["accuracy"],
    "nb_epoch": 800, 
    "algo": {
        "name": "adam",
        "params": {
            "lr": 1e-3
        }
    }, 
    "batch_size": 100, 
    "pred_batch_size": 1024, 
    "regularization":{
        "l2": 0.000001,
        "l1": 0
    },
    "lr_schedule": {
        "name": "constant", 
        "params":{
        }
    }, 
    "early_stopping":{
        "name": "none",
        "params":{
        }
    },
    "checkpoint":{
        "loss": "val_accuracy",
        "save_best_only": True
    },
    "seed": 42, 
    "budget_secs": 86400
}
model = {
    "name": "lenet",
    "params": {
        "nb_filters": [32, 64, 128],
        "dropout": 0.5,
        "fc_dropout": 0.5,
        "batch_norm": False,
        "fc": [512, 256, 128],
        "size_filters": 3,
        "activation": "prelu"
    }
}

def get_mnist_data(filename, start_train, nb_train, start_valid, nb_valid, start_test, nb_test, nb_classes, shuffle=False):
    data = {
        "name": "loader",
        "params": {
            "train":{
                "pipeline":[
                    {"name": "load_numpy", "params": {"filename": filename, "start":start_train, "nb": nb_train, "shuffle": shuffle, "random_state": random_state}},
                    {"name": "divide_by", "params": {"value": 255}},
                    {"name": "onehot", "params": {"nb_classes": nb_classes}}
                ]
            },
            "valid":{
                "pipeline":[
                    {"name": "load_numpy", "params": {"filename": filename, "start": start_valid, "nb": nb_valid, "shuffle": shuffle, "random_state": random_state}},
                    {"name": "divide_by", "params": {"value": 255}},
                    {"name": "onehot", "params": {"nb_classes": nb_classes}}
                ]
            },
            "test":{
                "pipeline": [
                    {"name": "load_numpy", "params": {"filename": filename, "start": start_test, "nb": nb_test, "shuffle": shuffle, "random_state": random_state}},
                    {"name": "divide_by", "params": {"value": 255}},
                    {"name": "onehot", "params": {"nb_classes": nb_classes}}
                ]
            }
        },
        "seed": 1, 
        "shuffle": None, 
        "valid_ratio": None
    }
    return data

def get_base(filename, ratio_valid, ratio_test, outdir, nb_classes=None, get_data=get_mnist_data):
    data = np.load(os.getenv('DATA_PATH') + '/' + filename)
    X = data['X']
    y = data['y']
    if nb_classes is None:
        nb_classes = len(set(y))
    nb_examples = X.shape[0]
    print('Nb of examples : {}, Nb of classes : {}'.format(nb_examples, nb_classes))
    ratio_train = 1 - ratio_valid - ratio_test
    nb_train = int(nb_examples * ratio_train)
    nb_valid = int(nb_examples * ratio_valid)
    nb_test = nb_examples - nb_train - nb_valid
    
    start_train, nb_train = 0, nb_train
    start_valid, nb_valid = nb_train, nb_valid
    start_test, nb_test =  nb_train + nb_valid, nb_test
    
    print('start train : {} nb train : {}'.format(start_train, nb_train))
    print('start valid : {} nb valid : {}'.format(start_valid, nb_valid))
    print('start test : {} nb test : {}'.format(start_test, nb_test))
    
    base = {
        'optim': optim,
        'model': model,
        'data': get_data(filename, start_train, nb_train, start_valid, nb_valid, start_test, nb_test, nb_classes),
        'outdir': outdir
    }
    return base

ratio_valid = 0.1
ratio_test = 0.2
def all_vs_fake_jobset75():
    return get_base('feature_generation/datasets/all_vs_fake_jobset75.npz', ratio_valid, ratio_test, 'out/feature_generation/all_vs_fake_jobset75')

def _5_vs_fake_jobset75():
    return get_base('feature_generation/datasets/5_vs_fake_jobset75.npz', ratio_valid, ratio_test, 'out/feature_generation/5_vs_fake_jobset75')

def _12457_vs_fake_jobset76():
    return get_base('feature_generation/datasets/12457_vs_fake_jobset76.npz', ratio_valid, ratio_test, 'out/feature_generation/12457_vs_fake_jobset76')

def mnist_classifier():
    outdir = 'out/feature_generation/mnist_classifier'
    data = {
        "name": "loader",
        "params": {
            "train":{
                "pipeline":[
                    {"name": "dataset", "params": {"name": "mnist", "which": "train"}},
                    {"name": "shuffle", "params": {"random_state": 2}},
                    {"name": "offset", "params": {"start": 0, "nb": 50000}},
                    {"name": "divide_by", "params": {"value": 255}},
                    {"name": "onehot", "params": {"nb_classes": 10}}
                ]
            },
            "valid":{
                "pipeline":[
                    {"name": "dataset", "params": {"name": "mnist", "which": "train"}},
                    {"name": "shuffle", "params": {"random_state": 2}},
                    {"name": "offset", "params": {"start": 50000, "nb": 10000}},
                    {"name": "divide_by", "params": {"value": 255}},
                    {"name": "onehot", "params": {"nb_classes": 10}}
                ]
            },
            "test":{
                "pipeline": [
                    {"name": "dataset", "params": {"name": "mnist", "which": "test"}},
                    {"name": "shuffle", "params": {"random_state": 2}},
                    {"name": "divide_by", "params": {"value": 255}},
                    {"name": "onehot", "params": {"nb_classes": 10}}
                ]
            }
        },
        "seed": 1, 
        "shuffle": True, 
        "valid_ratio": None
    }
    optim_ = optim.copy()
    optim_['nb_epoch'] = 400
    return {'optim': optim_, 'model': model, 'data': data, 'outdir': outdir}

def mnist_classifier_v2():
    params = mnist_classifier()
    params['model']['fc'] = [625]
    outdir = 'out/feature_generation/mnist_classifier_v2'
    params['outdir'] = outdir
    return params
def mnist_classifier_multilabel():
    params = mnist_classifier()
    outdir = 'out/feature_generation/mnist_classifier_multilabel'
    params['outdir'] = outdir
    params['optim']['loss'] = 'binary_crossentropy'
    return params

def hwrt():
    params = get_base('hwrt/data.npz', ratio_valid, ratio_test, 'out/feature_generation/hwrt', shuffle=True, random_state=42)
    model = {
        "name": "lenet",
        "params": {
            "nb_filters": [64, 128, 256],
            "dropout": 0,
            "fc_dropout": 0,
            "batch_norm": True,
            "fc": [256, 256],
            "size_filters": 3,
            "activation": "prelu"
        }
    }
    params['model'] = model
    return params


def get_fonts_data(filename, start_train, nb_train, start_valid, nb_valid, start_test, nb_test, nb_classes, shuffle=False, random_state=None):
    data = {
        "name": "loader",
        "params": {
            "train":{
                "pipeline":[
                    {"name": "load_numpy", "params": {"filename": filename, "start":start_train, "nb": nb_train, "shuffle": shuffle, "random_state": random_state}},
                    {"name": "order", "params": {"order": "tf"}},
                    {"name": "resize", "params": {"shape": [28, 28]}},
                    {"name": "order", "params": {"order": "th"}},
                    {"name": "divide_by", "params": {"value": 255}},
                    {"name": "invert", "params": {}},
                    {"name": "onehot", "params": {"nb_classes": nb_classes}}
                ]
            },
            "valid":{
                "pipeline":[
                    {"name": "load_numpy", "params": {"filename": filename, "start": start_valid, "nb": nb_valid, "shuffle": shuffle, "random_state": random_state}},
                    {"name": "order", "params": {"order": "tf"}},
                    {"name": "resize", "params": {"shape": [28, 28]}},
                    {"name": "order", "params": {"order": "th"}},
                    {"name": "divide_by", "params": {"value": 255}},
                    {"name": "invert", "params": {}},
                    {"name": "onehot", "params": {"nb_classes": nb_classes}}
                ]
            },
            "test":{
                "pipeline": [
                    {"name": "load_numpy", "params": {"filename": filename, "start": start_test, "nb": nb_test, "shuffle": shuffle, "random_state": random_state}},
                    {"name": "order", "params": {"order": "tf"}},
                    {"name": "resize", "params": {"shape": [28, 28]}},
                    {"name": "order", "params": {"order": "th"}},
                    {"name": "divide_by", "params": {"value": 255}},
                    {"name": "invert", "params": {}},
                    {"name": "onehot", "params": {"nb_classes": nb_classes}}
                ]
            }
        },
        "seed": 1, 
        "shuffle": None, 
        "valid_ratio": None
    }
    return data


def fonts():
    params = get_base('fonts/fonts.npz', ratio_valid, ratio_test, 'out/feature_generation/fonts', get_data=get_fonts_data)
    model = {
        "name": "lenet",
        "params": {
            "nb_filters": [64, 128, 256],
            "dropout": 0.2,
            "fc_dropout": 0.5,
            "batch_norm": False,
            "fc": [256, 256],
            "size_filters": 3,
            "activation": "prelu"
        }
    }

    model = {
        "params": {
            "size_blocks":[5,  5,  5],
            "nb_filters": [16, 32, 64],
            "block": "basic",
            "option": "B"
        }, 
        "name": "resnet"
    }
    params['model'] = model
    return params
