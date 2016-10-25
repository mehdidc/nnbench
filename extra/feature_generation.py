import numpy as np
import os

def get_base(filename, ratio_valid, ratio_test, outdir):
    data = np.load(os.getenv('DATA_PATH') + '/' + filename)['X']
    nb_examples = data.shape[0]
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
        "optim": {
            "loss": "categorical_crossentropy",
            "metrics": ["accuracy"],
            "nb_epoch": 200, 
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
        }, 
        "model": {
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
        }, 
        "data": {
            "name": "loader",
            "params": {
                "train":{
                    "pipeline":[
                        {"name": "load_numpy", "params": {"filename": filename, "start":start_train, "nb": nb_train}},
                        {"name": "divide_by", "params": {"value": 255}},
                        {"name": "onehot", "params": {"nb_classes": 11}}
                    ]
                },
                "valid":{
                    "pipeline":[
                        {"name": "load_numpy", "params": {"filename": filename, "start": start_valid, "nb": nb_valid}},
                        {"name": "divide_by", "params": {"value": 255}},
                        {"name": "onehot", "params": {"nb_classes": 11}}
                    ]
                },
                "test":{
                    "pipeline": [
                        {"name": "load_numpy", "params": {"filename": filename, "start": start_test, "nb": nb_test}},
                        {"name": "divide_by", "params": {"value": 255}},
                        {"name": "onehot", "params": {"nb_classes": 11}}
                    ]
                }
            },
            "seed": 1, 
            "shuffle": None, 
            "valid_ratio": None
        },
        "outdir": outdir
    }
    return base

ratio_valid = 0.1
ratio_test = 0.2
def all_vs_fake_jobset75():
    return get_base('feature_generation/figs/all_vs_fake_jobset75.npz', ratio_valid, ratio_test, 'out/feature_generation/all_vs_fake_jobset75')

def _5_vs_fake_jobset75():
    return get_base('feature_generation/figs/5_vs_fake_jobset75.npz', ratio_valid, ratio_test, 'out/feature_generation/5_vs_fake_jobset75')

def _12457_vs_fake_jobset76():
    return get_base('feature_generation/figs/12457_vs_fake_jobset76.npz', ratio_valid, ratio_test, '12457_vs_fake_jobset76')
