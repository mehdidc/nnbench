import numpy as np
import os

optim = {
    "loss": "categorical_crossentropy",
    "metrics": ["accuracy"],
    "nb_epoch": 100, 
    "algo": {
        "name": "adam",
        "params": {
            "lr": 1e-3
        }
    }, 
    "batch_size": 100, 
    "pred_batch_size": 1024, 
    "regularization":{
        "l2": 0.01,
        "l1": 0.01
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
    "name": "mlp",
    "params": {
        "nb_hidden_units": [100],
        "activation": "relu"
    }
}

def get_data(filename, start_train, nb_train, start_valid, nb_valid, start_test, nb_test, nb_classes, shuffle=True, random_state=42):
    data = {
        "name": "loader",
        "params": {
            "train":{
                "pipeline":[
                    {"name": "load_numpy", "params": {"filename": filename, "start":start_train, "nb": nb_train, "shuffle": shuffle, "random_state": random_state}},
                    {"name": "onehot", "params": {"nb_classes": nb_classes}}
                ]
            },
            "valid":{
                "pipeline":[
                    {"name": "load_numpy", "params": {"filename": filename, "start": start_valid, "nb": nb_valid, "shuffle": shuffle, "random_state": random_state}},
                    {"name": "onehot", "params": {"nb_classes": nb_classes}}
                ]
            },
            "test":{
                "pipeline": [
                    {"name": "load_numpy", "params": {"filename": filename, "start": start_test, "nb": nb_test, "shuffle": shuffle, "random_state": random_state}},
                    {"name": "onehot", "params": {"nb_classes": nb_classes}}
                ]
            }
        },
        "seed": 1, 
        "shuffle": None, 
        "valid_ratio": None
    }
    return data

def get_base(filename, ratio_valid, ratio_test, outdir, **kw):
    data = np.load(filename)
    X = data['X']
    y = data['y']
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
        'data': get_data(filename, start_train, nb_train, start_valid, nb_valid, start_test, nb_test, nb_classes, **kw),
        'outdir': outdir
    }
    return base

ratio_valid = 0.1
ratio_test = 0.1

def mlp_basic():
    filename = '/home/mcherti/work/code/molecules_dev/data/raw/activity/5ht2a_ic50.npz'
    params = get_base(filename, ratio_valid, ratio_test, 'out/molecule/5ht2a/mlp', shuffle=False, random_state=42)
    return params
