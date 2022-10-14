import argparse
from TrainRobustNN.utils.conv_models_define import *
import json

class Config(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def get_parameters():
    
    parser = argparse.ArgumentParser(description="Training Example", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, help="the config file to storage model, dataset and hyperparameter")
    parser.add_argument('--train_version', type=str, help="training method, including origin, robust+adversarial and different abstraction granularity", default='v1')
    args = parser.parse_args()
    
    config = Config()
    default_config_file = json.load(open("./config/default.json", 'r'))
    for k, v in default_config_file.items():
        config[k] = v
    config_file = json.load(open("./config/" + args.config, 'r'))
    for k, v in config_file.items():
        config[k] = v
    config['train_version'] = args.train_version
    return config

