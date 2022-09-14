import argparse
from utils.conv_models_define import *
import json

class Config(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def get_parameters():
    
    parser = argparse.ArgumentParser(description="Training Example", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, help="the config file to storage model, dataset and hyperparameter")
    args = parser.parse_args()
    
    config = Config()
    default_config_file = json.load(open("./config/default.json", 'r'))
    for k, v in default_config_file.items():
        config[k] = v
    config_file = json.load(open("./config/" + args.config, 'r'))
    for k, v in config_file.items():
        config[k] = v
    
    return config

if __name__ == '__main__':
    
    args = get_parameters()

    # parameters of dataset
    dataset = args.dataset
    in_ch = args.in_ch * 2
    in_dim = args.in_dim
    
    # parameters of models
    model_dir = args.model_dir
    model_name = args.model_name
    model_path = model_dir + model_name
    
    structure = args.structure
    if structure == 'DM_Small':
        model_struc = DM_Small(in_ch, in_dim)
    elif structure == 'DM_Medium':
        model_struc = DM_Medium(in_ch, in_dim)
    elif structure == 'DM_Large':
        model_struc = DM_Large(in_ch, in_dim)
    fnn = args.fnn == "True"
    
    # parameters of perturbation
    epsilon = args.epsilon
    k = args.k
    interval_num = 2 // (k * epsilon)
    
    # parameters of training
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    
    # parameters of logs
    log_dir = args.log_dir
    log_path = log_dir + model_name
    