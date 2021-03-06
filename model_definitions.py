from copy import deepcopy
import numpy as np

##############################################
# Defaults
##############################################

mnist_data = {
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

cifar10_data = {
    "name": "loader",
    "params": {
        "train":{
            "pipeline":[
                {"name": "dataset", "params": {"name": "cifar", "which": "train"}},
                {"name": "shuffle", "params": {"random_state": 2}},
                {"name": "offset", "params": {"start": 0, "nb": 50000}},
                {"name": "divide_by", "params": {"value": 255}},
                {"name": "onehot", "params": {"nb_classes": 10}}
            ]
        },
        "valid":{
            "pipeline":[
                {"name": "dataset", "params": {"name": "cifar", "which": "train"}},
                {"name": "shuffle", "params": {"random_state": 2}},
                {"name": "offset", "params": {"start": 50000, "nb": 10000}},
                {"name": "divide_by", "params": {"value": 255}},
                {"name": "onehot", "params": {"nb_classes": 10}}
            ]
        },
        "test":{
            "pipeline": [
                {"name": "dataset", "params": {"name": "cifar", "which": "test"}},
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

cifar10_data_2 = {
    "name": "cifar10",
    "params": {
    },
    "seed": 1, 
    "shuffle": True, 
    "valid_ratio": None
}


default_optim = {
    "loss": "categorical_crossentropy",
    "metrics": ["accuracy"],
    "nb_epoch": 200, 
    "algo": {
        "name": "adam",
        "params": {
            "lr": 1e-3
        }
    }, 
    "batch_size": 128, 
    "pred_batch_size": 1024, 
    "regularization":{
        "l2": 0.,
        "l1": 0
    },
    "lr_schedule":{
        "name": "decrease_when_stop_improving",
        "params": {
            "loss": "val_accuracy",
            "shrink_factor": 2,
            "patience": 30,
            "min_lr": 0.00001
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

ok_optim = deepcopy(default_optim)
ok_optim["algo"]["name"] = "SGD"
ok_optim["algo"]["params"] = {"nesterov": True, "lr": 0.0005, "momentum": 0.99}

torch_blog_optim = deepcopy(default_optim)
torch_blog_optim["algo"]["name"] = "SGD"
torch_blog_optim["algo"]["params"] = {"nesterov": True, "lr": 1., "momentum": 0.9}
torch_blog_optim["lr_schedule"] = {
    "name": "decrease_every",
    "params": {
        "shrink_factor": 2,
        "every": 25,
    }
}

small_test_cnn = {
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
    "optim": default_optim,
    "data": {
        "name": "cifar10",
        "params": {},
        "seed": 1,
        "shuffle": True,
        "valid_ratio": None, # meaning use default
    }
}

small_test_mlp = deepcopy(small_test_cnn)
small_test_mlp["model"] = {
    "name": "mlp",
    "params": {
        "activation": "relu",
        "nb_hidden_units": [100]
    }
}

##############################################
# VGG
##############################################

def model_vgg_A(fc=[4096, 4096]):
    return {"name": "vgg",
            "params": {
                "nb_filters": [64, 128, 256, 512, 512],
                "size_blocks": [1, 1, 2, 2, 2],
                "size_filters": 3,
                "stride": 2,
                "fc": fc,
                "fc_dropout": 0.5,
                "activation": "relu"}}

def model_vgg_B():
    return {"name": "vgg",
            "params": {
                "nb_filters": [64, 128, 256, 512, 512],
                "size_blocks": [2, 2, 2, 2, 2],
                "size_filters": 3,
                "stride": 2,
                "fc": [4096, 4096],
                "fc_dropout": 0.5,
                "activation": "relu"}}


def model_vgg_D(fc=[4096, 4096]):
    return {"name": "vgg",
            "params": {
                "nb_filters": [64, 128, 256, 512, 512],
                "size_blocks": [2, 2, 3, 3, 3],
                "size_filters": 3,
                "stride": 2,
                "fc": fc,
                "fc_dropout": 0.5,
                "activation": "relu"}}


def model_vgg_E(fc=[4096, 4096]):
    return {"name": "vgg",
            "params": {
                "nb_filters": [64, 128, 256, 512, 512],
                "size_blocks": [3, 3, 4, 4, 4],
                "size_filters": 3,
                "stride": 2,
                "fc": fc,
                "fc_dropout": 0.5,
                "activation": "relu"}}


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
        "name": "vgg",
        "params": {
            "nb_filters": nb_filters,
            "size_blocks": size_blocks,
            "size_filters": size_filters,
            "stride": stride,
            "fc": fc,
            "fc_dropout": 0.5,
            "activation": random_activation(rng)}}


def vgg_D_optim_cifar(random_state=None):
    rng = np.random.RandomState(random_state)
    optim = random_optim(rng)
    fc = 512
    model = model_vgg_D(fc=[fc, fc])
    data = random_data(rng, datasets=("cifar10",))
    return {"optim": optim, "model": model, "data": data}


def vgg_D_optim_cifar_24h(random_state=None):
    rng = np.random.RandomState(random_state)
    optim = random_optim(rng)
    optim["budget_secs"] = 24 * 3600
    fc = 512
    model = model_vgg_D(fc=[fc, fc])
    data = random_data(rng, datasets=("cifar10",))
    return {"optim": optim, "model": model, "data": data}


def vgg_D_optim_cifar_24h_no_valid(random_state=None):
    rng = np.random.RandomState(random_state)
    optim = deepcopy(ok_optim)
    optim["early_stopping"]["params"]["patience_loss"] = "train_accuracy"
    optim["lr_schedule"]["params"]["loss"] = "train_accuracy"
    optim["budget_secs"] = 24 * 3600
    fc = 512
    model = model_vgg_D(fc=[fc, fc])
    data = random_data(rng, datasets=("cifar10",))
    data["val_ratio"] = 0
    return {"optim": optim, "model": model, "data": data}


def vgg_D_optim_cifar_schedule_24h(random_state=None):
    rng = np.random.RandomState(random_state)
    optim = deepcopy(ok_optim)
    optim["lr_schedule"] = {
        "name": "decrease_when_stop_improving",
        "params":{
            "loss": rng.choice(("train_accuracy", "val_accuracy")),
            "shrink_factor": rng.choice((2, 5, 10)),
            "patience": rng.choice((10, 15, 20, 30, 40, 50)),
            "min_lr": rng.choice((0.000001, 0.00001, 0.0001, 0.001))
        }
    }
    optim["budget_secs"] = 24 * 3600
    fc = 512
    model = model_vgg_D(fc=[fc, fc])
    data = random_data(rng, datasets=("cifar10",))
    return {"optim": optim, "model": model, "data": data}


def vgg_D_optim_cifar_torch_blog_24h(random_state=None):
    rng = np.random.RandomState(random_state)
    optim = deepcopy(torch_blog_optim)
    optim["algo"]["params"] = { 
          "nesterov": bool(rng.choice((True, False))),
          "lr":  rng.choice((0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1)),
          "momentum": 0.9
    }
    optim["budget_secs"] = 24 * 3600
    fc = 512
    model = model_vgg_D(fc=[fc, fc])
    data = random_data(rng, datasets=("cifar10",))
    return {"optim": optim, "model": model, "data": data}


def vgg_A_optim_cifar_24h(random_state=None):
    rng = np.random.RandomState(random_state)
    optim = deepcopy(ok_optim)
    optim["lr_schedule"] = {
        "name": "decrease_when_stop_improving",
        "params":{
            "loss": rng.choice(("train_accuracy", "val_accuracy")),
            "shrink_factor": rng.choice((2, 5, 10)),
            "patience": rng.choice((10, 15, 20, 30, 40, 50)),
            "min_lr": rng.choice((0.000001, 0.00001, 0.0001, 0.001))
        }
    }
    optim["budget_secs"] = 24 * 3600
    fc = rng.choice((64, 128, 256, 512, 800))
    model = model_vgg_A(fc=[fc, fc])
    data = random_data(rng, datasets=("cifar10",))
    return {"optim": optim, "model": model, "data": data}

def vgg_E_optim_cifar_24h(random_state=None):
    rng = np.random.RandomState(random_state)
    optim = deepcopy(ok_optim)
    optim["lr_schedule"] = {
        "name": "decrease_when_stop_improving",
        "params":{
            "loss": rng.choice(("train_accuracy", "val_accuracy")),
            "shrink_factor": rng.choice((2, 5, 10)),
            "patience": rng.choice((10, 15, 20, 30, 40, 50)),
            "min_lr": rng.choice((0.000001, 0.00001, 0.0001, 0.001))
        }
    }
    lr = rng.choice((0.1, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005))
    optim["algo_params"] = {"nesterov": True, "lr":lr, "momentum": 0.99}
    optim["budget_secs"] = 24 * 3600
    fc = rng.choice((64, 128, 256, 512, 800))
    model = model_vgg_E(fc=[fc, fc])
    data = random_data(rng, datasets=("cifar10",))
    return {"optim": optim, "model": model, "data": data}

##############################################
# DenseNet 
##############################################

def default_densenet_model(rng):
    return {
        "name": "densenet",
        "params": {
            "growth": 12,
            "size_filter_block": 3,
            "size_filter_transition": 1,
            "dropout": 0.2,
            "per_block": 12,
            "nb_blocks": 3,
            "activation": "relu",
            "init_feature_maps": 16
        }
    }

def default_densenet(random_state=None):
    rng = np.random.RandomState(random_state)
    optim = deepcopy(ok_optim)
    optim["budget_secs"] = 24 * 3600
    model = default_densenet_model(rng)
    data = random_data(rng, datasets=("cifar10",))
    return {"optim": optim, "model": model, "data": data}


##############################################
# Squeezenet 
##############################################

def default_squeezenet_model(rng):
    return {"name": "squeezenet",
            
            "params":{
                "squeeze_filters_size": [16, 16, 32,  32,  48,  48,  64, 64    ],
                "expand_filters_size":  [64, 64, 128, 128, 192, 192, 256, 256  ],
                "pool":                 [0,  0,  1,   0,    0,  1,   0,   0,  1],
                "dropout": 0.5
            }}

def default_squeezenet(random_state=None):
    rng = np.random.RandomState(random_state)
    optim = deepcopy(ok_optim)
    optim["budget_secs"] = 24 * 3600
    model = default_squeezenet_model(rng)
    data = random_data(rng, datasets=("cifar10",))
    return {"optim": optim, "model": model, "data": data}


##############################################
# Random ones
##############################################

def random(random_state=None):
    rng = np.random.RandomState(random_state)
    optim = random_optim(rng)
    model = random_model(rng)
    data = random_data(rng)
    data["val_ratio"] = 0.1
    optim["budget_secs"] = 3600 * 4
    return {"optim": optim, "model": model, "data": data}


def mini_random(random_state=None):
    rng = np.random.RandomState(random_state)
    optim = random_optim(rng)
    model = random_model(rng)
    data = random_data(rng)
    optim["budget_secs"] = 3600
    return {"optim": optim, "model": model, "data": data}


def micro_random(random_state=None):
    rng = np.random.RandomState(random_state)
    optim = random_optim(rng)
    model = random_model(rng)
    data = random_data(rng, datasets=("cifar10",))
    optim["budget_secs"] = 60 * 15
    return {"optim": optim, "model": model, "data": data}


##############################################
# General ones
##############################################


def take_bests_on_validation_set_and_use_full_training_data(random_state=None):
    rng = np.random.RandomState(random_state)
    from lightjob.cli import load_db
    from lightjob.db import SUCCESS
    db = load_db()
    jobs = db.jobs_with(state=SUCCESS)
    jobs = list(jobs)
    db.close()

    jobs = filter(lambda j:"val_accuracy" in j["results"], jobs)
    jobs = sorted(jobs, key=lambda j:(j["results"]["val_accuracy"][-1]), reverse=True)

    j = rng.choice(jobs[0:10])
    nb_epochs = 1 + np.argmax(j["results"]["val_accuracy"])
    params = j["content"].copy()
    optim = params["optim"]
    optim["nb_epoch"] = nb_epochs
    optim["early_stopping"]["name"] = "none"
    optim["early_stopping"]["params"] = {}
    optim["checkpoint"] = {"save_best_only": False, "loss": None}
    if "loss" in optim["lr_schedule"]:
        optim["lr_schedule"]["loss"] = "train_accuracy"
    data = params["data"]
    data["val_ratio"] = 0
    return params

##############################################
# General subparts
##############################################


def random_activation(rng):
    return rng.choice(("relu", "leaky_relu"))

def random_model(rng):
    return random_model_vgg(rng)


def random_optim(rng, extended=False):
    optim = deepcopy(default_optim)
    algo = rng.choice(("Adam", "RMSprop", "SGD", "Adadelta"))

    optim["algo"] = {}
    optim["algo"]["name"] = algo
    
    lr = rng.choice((0.1, 0.01, 0.05, 0.001, 0.005, 0.0001, 0.0005))
    algo_params = {}
    algo_params["lr"] = lr
    if algo == "SGD":
        momentum = rng.choice((0.5, 0.9, 0.95, 0.99, 0))
        nesterov = bool(rng.choice((True, False)))
        algo_params.update({"momentum": momentum, "nesterov": nesterov})
    optim["algo"]["params"] = algo_params
    return optim


def random_data(rng, datasets=("mnist", "cifar10")):
    ds = rng.choice(datasets)
    return cifar10_data_2
#### TUNING SOME SPECIFIC DATASETS

def model_vgg_1(fc=[100, 100]):
    return {"name": "vgg",
            "params": {
                "nb_filters": [16, 16, 32, 32, 64, 64],
                "size_blocks": [3, 3,  3,  3,  3,  3],
                "stride":      [1, 2,  1,  1,  1,  1],
                "size_filters": 3,
                "fc": fc,
                "fc_dropout": 0.5,
                "activation": "relu"}}
def vgg1(random_state=None):
    rng = np.random.RandomState(random_state)
    optim = deepcopy(torch_blog_optim)
    optim["budget_secs"] = 24 * 3600
    fc = [100]
    model = model_vgg_1(fc=fc)
    data = random_data(rng, datasets=("ilc",))
    return {"optim": optim, "model": model, "data": data}
