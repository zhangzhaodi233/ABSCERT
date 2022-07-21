import argparse
from my_models_define import *
    

def get_parameters():
    
    parser = argparse.ArgumentParser(description="Training Example", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default="mnist", help='the kind of dataset')
    parser.add_argument('--in_ch', type=int, default=1, help='the number of channels of the dataset image')
    parser.add_argument('--in_dim', type=int, default=28, help='the width/hight of the dataset image')
    
    parser.add_argument('--model_dir', type=str, default="exp_results/", help='the folder where the model is saved')
    parser.add_argument('--model_name', type=str, default="model_name", help='the name of the trained model')
    parser.add_argument('--structure', type=str, default="DM_Small", help='the strcture of the trained model')
    parser.add_argument('--fnn', type=bool, default=False, help='whether the trained model is fnn')
    
    parser.add_argument('--epsilon', type=float, default=0.1, help='the epsilon for L_infinity perturbation')
    parser.add_argument('--k', type=float, default=2.0, help='how many times the size of the abstract interval is the size of epsilon.')
    
    parser.add_argument('--batch_size', type=int, default=256, help='the number of samples per gradient update')
    parser.add_argument('--epochs', type=int, default=10, help='the number of epochs to train the model')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='the step size at each iteration while moving toward a minimum of a loss function.')
    
    parser.add_argument('--log_dir', type=str, default="runs/", help='the folder where the log is saved')
    
    args = parser.parse_args()
    
    return args

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
    fnn = args.fnn
    
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
    