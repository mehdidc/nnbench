from feature_generation import optim
from feature_generation import model
from feature_generation import resnet
from feature_generation import get_base

ratio_valid = 0.1
ratio_test = 0.2

optim = {
    "loss": "binary_crossentropy",
    "metrics": ["binary_crossentropy", "multilabel_accuracy"],
    "nb_epoch": 400, 
    "algo": {
        "name": "adam",
        "params": {
            "lr": 1e-3
        }
    }, 
    "batch_size": 32, 
    "pred_batch_size": 32, 
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
        "loss": "binary_crossentropy",
        "save_best_only": True
    },
    "seed": 42, 
    "budget_secs": 86400
}


def get_data(filename, start_train, nb_train, start_valid, nb_valid, start_test, nb_test, nb_classes, shuffle=False, random_state=42):
    #nb_train = 100
    #nb_valid = 100
    data = {
        "name": "loader",
        "params": {
            "train":{
                "pipeline":[
                    {"name": "load_numpy", "params": {"filename": filename, "start":start_train, "nb": nb_train, "shuffle": shuffle, "random_state": random_state}},
                    {"name": "divide_by", "params": {"value": 255}},
                ]
            },
            "valid":{
                "pipeline":[
                    {"name": "load_numpy", "params": {"filename": filename, "start": start_valid, "nb": nb_valid, "shuffle": shuffle, "random_state": random_state}},
                    {"name": "divide_by", "params": {"value": 255}},
                ]
            },
            "test":{
                "pipeline": [
                    {"name": "load_numpy", "params": {"filename": filename, "start": start_test, "nb": nb_test, "shuffle": shuffle, "random_state": random_state}},
                    {"name": "divide_by", "params": {"value": 255}},
                ]
            }
        },
        "seed": 1, 
        "shuffle": None, 
        "valid_ratio": None
    }
    return data

def celeba_resnet():
    params = get_base('celeba/train.npz', ratio_valid, ratio_test, 'out/celeba/resnet', get_data=get_data)
    params['optim']= optim
    params['model'] = resnet
    return params
