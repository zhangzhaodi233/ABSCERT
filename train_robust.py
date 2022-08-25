import math
import torch
import torch.nn as nn

from transformers import get_cosine_schedule_with_warmup
import os
from my_models_define import *
from utils import * 
from my_datasets import load_dataset, abstract_data, abstract_disturbed_data
import time
from d2l import torch as d2l


mnist_text_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
cifar_text_labels = ['Airplane', "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

def printlog(s, model_name):
    print(s, file=open("result/"+model_name+".log", "a"))

def print_concise_log(s, model_name):
    print(s, file=open("result/"+model_name+"_concise.log", "a"))

class RobustModel:
    def __init__(self, model_path, log_path, model, dataset, fnn=False, interval_num=1, batch_size=256, epochs=10, learning_rate=0.01, optimizer="Adam", weight_decay=0.0001, momentum=0.9, init="xavier", lr_scheduler='cosine'):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model_save_path = model_path
        self.log_path = log_path
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.fnn = fnn
        self.dataset = dataset
        self.interval_num = interval_num
        self.optimizer = optimizer
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.init = init
        self.lr_scheduler = lr_scheduler

    def train(self):
		
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                if self.init == "gaussian":  # gaussian init，used in alexnet and vgg
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                elif self.init == 'kaiming':  # proposed by Kaiming He(MSRA), used in resnet
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                else:  # default xavier init
                    nn.init.xavier_uniform_(m.weight)
		
        self.model.conv.apply(init_weights)
        self.model.fc.apply(init_weights)

        train_iter, test_iter = load_dataset(self.batch_size, self.dataset)
        last_epoch = -1
        # 接着之前的模型继续训练
        # if os.path.exists('exp_results/lenet5_test.pt'):
        #     checkpoint = torch.load('exp_results/lenet5_test.pt')
        #     last_epoch = checkpoint['last_epoch']
        #     self.model.load_state_dict(checkpoint['model_state_dict'])

        len_train_iter = len(train_iter)
        num_training_steps = len_train_iter * self.epochs
        if self.optimizer == "SGD":
            optimizer = torch.optim.SGD([{
                "params": self.model.parameters(), "lr": self.learning_rate, "weight_decay": self.weight_decay, "momentum": self.momentum
            }])
        else:
            optimizer = torch.optim.Adam([{
                "params": self.model.parameters(), "lr": self.learning_rate, "weight_decay": self.weight_decay
            }])
        if self.lr_scheduler == "steplr":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, math.ceil(self.epochs/5), 0.1, -1)  # 下调4次
        else:  # cosine
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(self.epochs * len_train_iter / 10),
                                                    num_training_steps=num_training_steps,
                                                    num_cycles=2, last_epoch=last_epoch)
        model_name = self.log_path.split("/")[1]
        
        animator = d2l.Animator(xlabel='epoch', xlim=[1, self.epochs], legend=['train loss', 'train acc', 'test acc'])
        self.model = self.model.to(self.device)
        max_test_acc = 0
        time_sum = 0
        
        for epoch in range(self.epochs):
            # print current learning rate
            # print(optimizer.state_dict()['param_groups'][0]['lr'])
            start = time.time()
            for i, data in enumerate(train_iter):
                if self.dataset == 'imagenet':
                    x = data[0]["data"]
                    y = data[0]["label"].squeeze(-1).long().to(x.device)
                else:
                    x, y = data
                    x, y = x.to(self.device), y.to(self.device)
                if self.fnn:
                    if self.dataset == 'mnist':
                        x = x.view(-1, 784)
                    elif self.dataset == 'cifar':
                        x = x.view(-1, 3*32*32)

                x = abstract_data(x, self.interval_num)
                
                loss, logits = self.model(x, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if self.lr_scheduler != "steplr":
                    scheduler.step()
                if i%50 == 0:  # print log every 50 iter
                    acc = (logits.argmax(1) == y).float().mean()
                    print("### Epochs [{}/{}] --- batch[{}/{}] --- acc {:.4} --- loss {:.4}".format(epoch+1, self.epochs, i, len(train_iter), acc, loss.item()))
                    printlog("### Epochs [{}/{}] --- batch[{}/{}] --- acc {:.4} --- loss {:.4}".format(epoch+1, self.epochs, i, len(train_iter), acc, loss.item()), model_name)
                    animator.add(epoch + (i + 1) / len_train_iter, (loss.detach().cpu(), acc.cpu(), None))

            time_sum += time.time() - start
            if self.lr_scheduler == "steplr":
                scheduler.step()

            test_acc = self.evaluate(test_iter)
            print("### Epochs [{}/{}] -- Acc on test {:.4}".format(epoch + 1, self.epochs, test_acc))
            printlog("### Epochs [{}/{}] -- Acc on test {:.4}".format(epoch + 1, self.epochs, test_acc), model_name)
            animator.add(epoch + 1, (None, None, test_acc))


            if test_acc > max_test_acc:
                max_test_acc = test_acc
                state_dict = self.model.state_dict()
            torch.save({'last_epoch': scheduler.last_epoch,
                        'model_state_dict': state_dict},
                        self.model_save_path + '.pt')
        print("### max test accuracy: {:.4}".format(max_test_acc))
        print("### error: {:.4}%".format((1 - max_test_acc)*100))
        print("### time of per epoch: {:.4}".format(time_sum / epoch))
        printlog("### max test accuracy: {:.4}".format(max_test_acc), model_name)
        printlog("### error: {:.4}%".format((1 - max_test_acc)*100), model_name)
        printlog("### time of per epoch: {:.4}".format(time_sum / epoch), model_name)
        d2l.plt.savefig(f'{self.model_save_path}.png')
        return max_test_acc
    
    def evaluate(self, data_iter):
        self.model.eval()
        with torch.no_grad():
            acc_sum, n = 0.0, 0
            for data in data_iter:
                if self.dataset == 'imagenet':
                    x = data[0]["data"]
                    y = data[0]["label"].squeeze(-1).long().to(x.device)
                else:
                    x, y = data
                    x, y = x.to(self.device), y.to(self.device)

                if self.fnn:
                    if self.dataset == 'mnist':
                        x_fnn = x.view(-1, 784)
                    elif self.dataset == 'cifar':
                        x_fnn = x.view(-1, 3*32*32)
                    x_abstract = abstract_data(x_fnn, self.interval_num)
                else:
                    x_abstract = abstract_data(x, self.interval_num)
                    
                logits = self.model(x_abstract)
            
                acc_sum += (logits.argmax(1) == y).float().sum().item()
                n += len(y)

            self.model.train()
            return acc_sum / n


    def verify(self, data_iter):
        self.model.eval()
        all_logits = []
        y_labels = []
        images = []
        with torch.no_grad():
            acc_sum, n = 0.0, 0
            for x, y in data_iter:
        
                if self.fnn:
                    if self.dataset == 'mnist':
                        x_fnn = x.view(-1, 784)
                    elif self.dataset == 'cifar':
                        x_fnn = x.view(-1, 3*32*32)
                    x_disturbed_abstract = abstract_disturbed_data(x_fnn, self.interval_num, self.epsilon)
                else:
                    x_disturbed_abstract = abstract_disturbed_data(x, self.interval_num, self.epsilon)
          
                x_disturbed_abstract, y = x_disturbed_abstract.to(self.device), y.to(self.device)
                logits = self.model(x_disturbed_abstract)
                
                acc_sum += (logits.argmax(1) == y).float().sum().item()
                n += len(y)

                all_logits.append(logits)
                y_pred = logits.argmax(1).view(-1)
                if self.dataset == 'mnist':
                    y_labels += (mnist_text_labels[i] for i in y_pred)
                elif self.dataset == 'cifar':
                    y_labels += (cifar_text_labels[i] for i in y_pred)
                images.append(x)
            self.model.train()
            return acc_sum / n, torch.cat(all_logits, dim=0), y_labels, torch.cat(images, dim=0)

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
    """read config file and begin to train and valid"""
    args = get_parameters()
    make_dir('exp_results')
    make_dir('result')
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
    elif structure == 'LeNet':  # MNIST和CIFAR
        model_struc = LeNet5(in_ch, in_dim)
    elif structure == 'AlexNet':  # IMAGENET
        model_struc = AlexNet(in_ch)
    elif structure == 'VGG11':  # IMAGENET
        model_struc = VGG11(in_ch)
    elif structure == 'ResNet18':  # IMAGENET
        model_struc = ResNet18(in_ch)
    elif structure == 'ResNet34':
        model_struc = ResNet34(in_ch)
    fnn = args.fnn == 'True'
    
    # parameters of logs
    log_dir = args.log_dir
    log_path = log_dir + model_name

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

    # get the d list for train and valid
    d_range, d_threshold = generate_d(epsilon_list, d_step)
    
    record = []
    for i, d in enumerate(d_range):
        interval_num = 2 // d
        model_path_ = model_path + f'_{d}'

        print("--dataset {}, --model_name {}, --d {:.3f}, --batch_size {}, --epochs {}, --learning_rate {:.5f}".format(dataset, model_name, d, batch_size, epochs, learning_rate))
        printlog("--dataset {}, --model_name {}, --d {:.3f}, --batch_size {}, --epochs {}, --learning_rate {:.5f}".format(dataset, model_name, d, batch_size, epochs, learning_rate), model_name)
        d2l.plt.clf()
        model = RobustModel(model_path_, log_path, model_struc, dataset=dataset, fnn=fnn, interval_num=interval_num, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate,
            optimizer=optimizer, weight_decay=weight_decay, momentum=momentum, init=init, lr_scheduler=lr_scheduler)
        test_acc = model.train()
        record.append([d, test_acc])

    # print log
    printlog(f"Train result: \n{record}\n\n", model_name)
    max_acc, dd = 0, 2.0
    i = 0
    for r in record:
        if r[1] > max_acc:
            max_acc, dd = r[1], r[0]
        if r[0] in d_threshold:
            print(f"For epsilon {epsilon_list[i]}, the max verify acc = {max_acc} occers at d = {dd}.")
            printlog(f"For epsilon {epsilon_list[i]}, the max verify acc = {max_acc} occers at d = {dd}.", model_name)
            print_concise_log(f"For epsilon {epsilon_list[i]}, the max verify acc = {max_acc} occers at d = {dd}.", model_name)
            i += 1


if __name__ == '__main__':
    main()
