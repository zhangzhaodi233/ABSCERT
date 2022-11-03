import os
from TrainRobustNN.train.train import *
from TrainRobustNN.refinement.refinement import refinement
from TrainRobustNN.utils.params import get_parameters
from TrainRobustNN.utils.conv_models_define import *
from TrainRobustNN.utils.fc_models_define import *



def make_dir(filepath):
    '''
    create the dir if not exists
    '''
    if not os.path.exists(filepath):
        os.mkdir(filepath)

def generate_d(epsilon_list, d_step=0.01):
    """compute Abstract Granularity d until d < 2 * epsilon"""
    d_list = []
    d_threshold = []
    last_d = 2.0
    e = 0
    for i in range(2, 256, 1):  # divide the input interval to i parts
        d = 2.0 / i
        if d < 2 * epsilon_list[e]:
            real_d_for_epsilon = 2 / int(1 / epsilon_list[e])
            if real_d_for_epsilon not in d_list:
                d_list.append(real_d_for_epsilon)
            d_threshold.append(real_d_for_epsilon)
            e += 1
        if e >= len(epsilon_list):
            break
        if last_d - d >= d_step:  # avoid too much d
            d_list.append(d)
            last_d = d
    
    return d_list, d_threshold




def main():
    args = get_parameters()

    # parameters of dataset
    dataset = args.dataset
    in_ch = args.in_ch * 2
    in_dim = args.in_dim

    # parameters of models
    model_dir = args.model_dir
    model_name = args.model_name
    model_path = model_dir + model_name
    os.makedirs(model_dir, exist_ok=True)

    structure = args.structure
    if structure == 'DM_Small':  # MNIST and CIFAR
        model_struc = DM_Small_Relu(in_ch, in_dim)
    elif structure == 'DM_Medium':  # MNIST and CIFAR
        model_struc = DM_Medium_Relu(in_ch, in_dim)
    elif structure == 'DM_Large':  # MNIST and CIFAR
        model_struc = DM_Large_Relu(in_ch, in_dim)
    elif structure == 'DM_Small_Sigmoid':  # MNIST and CIFAR
        model_struc = DM_Small_Sigmoid(in_ch, in_dim)
    elif structure == 'DM_Medium_Sigmoid':  # MNIST and CIFAR
        model_struc = DM_Medium_Sigmoid(in_ch, in_dim)
    elif structure == 'DM_Large_Sigmoid':  # MNIST and CIFAR
        model_struc = DM_Large_Sigmoid(in_ch, in_dim)
    elif structure == 'DM_Small_Tanh':  # MNIST and CIFAR
        model_struc = DM_Small_Tanh(in_ch, in_dim)
    elif structure == 'DM_Medium_Tanh':  # MNIST and CIFAR
        model_struc = DM_Medium_Tanh(in_ch, in_dim)
    elif structure == 'DM_Large_Tanh':  # MNIST and CIFAR
        model_struc = DM_Large_Tanh(in_ch, in_dim)
    elif structure == 'LeNet':  # MNIST and CIFAR
        model_struc = LeNet5(in_ch, in_dim)
    elif structure == 'AlexNet':  # IMAGENET
        model_struc = AlexNet(in_ch)
    elif structure == 'VGG11':  # IMAGENET
        model_struc = VGG11(in_ch)
    elif structure == 'ResNet18':  # IMAGENET
        model_struc = ResNet18(in_ch)
    elif structure == 'ResNet34':
        model_struc = ResNet34(in_ch)
    elif structure == 'Inceptionv1':
        model_struc = Inception_v1(in_ch)
    elif structure == 'FC3':
        model_struc = FC3_Relu(in_ch, in_dim, 512)
    elif structure == 'FC5':
        model_struc = FC5_Relu(in_ch, in_dim, 512)
    elif structure == 'FC3_Sigmoid':
        model_struc = FC3_Sigmoid(in_ch, in_dim, 512)
    elif structure == 'FC5_Sigmoid':
        model_struc = FC5_Sigmoid(in_ch, in_dim, 512)
    elif structure == 'FC3_Tanh':
        model_struc = FC3_Tanh(in_ch, in_dim, 512)
    elif structure == 'FC5_Tanh':
        model_struc = FC5_Tanh(in_ch, in_dim, 512)
    fnn = args.fnn == 'True'

    # parameters of logs
    log_dir = args.log_dir
    log_path = log_dir + model_name
    os.makedirs(log_dir, exist_ok=True)

    # parameters of training
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    optimizer = args.optimizer
    weight_decay = args.weight_decay
    momentum = args.momentum
    init = args.init
    lr_scheduler = args.lr_scheduler
    d_step = args.d_step
    epsilon_list = args.epsilon
    epsilon_list = sorted(epsilon_list, reverse=True)
    train_version = args.train_version

    # get the d list for train and valid
    d_range, d_threshold = generate_d(epsilon_list, d_step)
    print(f"List of abstract granularity to train: \n{d_range}")
    printlog(f"List of abstract granularity to train: \n{d_range}", log_path)

    record = []
    for i, d in enumerate(d_range):
        interval_num = 2 // d
        model_path_ = model_path + f'_{d}'

        print("--dataset {}, --model_name {}, --d {:.3f}, --batch_size {}, --epochs {}, --learning_rate {:.5f}".format(dataset, model_name, d, batch_size, epochs, learning_rate))
        printlog("--dataset {}, --model_name {}, --d {:.3f}, --batch_size {}, --epochs {}, --learning_rate {:.5f}".format(dataset, model_name, d, batch_size, epochs, learning_rate), log_path)
        d2l.plt.clf()


        test_acc = train(model_struc, dataset, model_path_, log_path, fnn=fnn, interval_num=interval_num, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate,
                        optimizer=optimizer, weight_decay=weight_decay, momentum=momentum, init=init, lr_scheduler=lr_scheduler)

        record.append([d, test_acc])

        # refinement process
        # if not refinement(model_struc, model_path_, batch_size, dataset, interval_num, fnn=fnn):
        #     break

    # print log
    printlog(f"Train result: \n{record}\n\n", log_path)
    max_acc, dd = 0, 2.0
    i = 0
    for r in record:
        if r[1] > max_acc:
            max_acc, dd = r[1], r[0]
        if r[0] in d_threshold:
            print(f"For epsilon {epsilon_list[i]}, the max verify acc = {max_acc} occers at d = {dd}.")
            printlog(f"For epsilon {epsilon_list[i]}, the max verify acc = {max_acc} occers at d = {dd}.", log_path)
            print_concise_log(f"For epsilon {epsilon_list[i]}, the max verify acc = {max_acc} occers at d = {dd}.", log_path)
            i += 1
    while i < len(epsilon_list):
        print(f"For epsilon {epsilon_list[i]}, the max verify acc = {max_acc} occers at d = {dd}.")
        printlog(f"For epsilon {epsilon_list[i]}, the max verify acc = {max_acc} occers at d = {dd}.", log_path)
        print_concise_log(f"For epsilon {epsilon_list[i]}, the max verify acc = {max_acc} occers at d = {dd}.", log_path)
        i += 1

if __name__ == '__main__':
    main()
