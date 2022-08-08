import torch
import torch.nn as nn

from transformers import get_cosine_schedule_with_warmup
# from torch.utils.tensorboard import SummaryWriter

# import tensorflow as tf
# import tensorboard as tb

import os
from my_models_define import *
from utils import *
from my_datasets import load_dataset, abstract_data, abstract_disturbed_data
import time
from d2l import torch as d2l

# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

mnist_text_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
cifar_text_labels = ['Airplane', "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]


def printlog(s, model_name, dataset):
    print(s, file=open("result/" + model_name + ".txt", "a"))


class RobustModel:
    def __init__(self, model_path, log_path, model, dataset, fnn=False, interval_num=1, epsilon=0, batch_size=256, epochs=10, learning_rate=0.01, optimizer="Adam", weight_decay=0.0001, momentum=0.9, init="xavier"):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model_save_path = model_path + '.pt'
        self.log_path = log_path
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.fnn = fnn
        self.dataset = dataset
        self.interval_num = interval_num
        self.epsilon = epsilon
        self.optimizer = optimizer
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.init = init

    def train(self):

        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                if self.init == "gaussian":
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                else:  # 默认xavier分布
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
                "params": self.model.parameters(), "initial_lr": self.learning_rate, "weight_decay": self.weight_decay
            }])
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(self.epochs * len_train_iter / 10),
                                                    num_training_steps=num_training_steps,
                                                    num_cycles=2, last_epoch=last_epoch)
        model_name = self.log_path.split("/")[1]

        # writer = SummaryWriter(self.log_path)
        animator = d2l.Animator(xlabel='epoch', xlim=[1, self.epochs], legend=['train loss', 'train acc', 'test acc'])
        self.model = self.model.to(self.device)
        max_test_acc = 0
        time_sum = 0
        for epoch in range(self.epochs):
            # 查看学习率
            # print(optimizer.state_dict()['param_groups'][0]['lr'])
            start = time.time()
            for i, (x, y) in enumerate(train_iter):
                if self.fnn:
                    if self.dataset == 'mnist':
                        x = x.view(-1, 784)
                    elif self.dataset == 'cifar':
                        x = x.view(-1, 3 * 32 * 32)

                x = abstract_data(x, self.interval_num)
                x, y = x.to(self.device), y.to(self.device)
                loss, logits = self.model(x, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                if i % 50 == 0:
                    acc = (logits.argmax(1) == y).float().mean()
                    print("### Epochs [{}/{}] --- batch[{}/{}] --- acc {:.4} --- loss {:.4}".format(epoch + 1, self.epochs, i, len(train_iter), acc, loss.item()))
                    printlog("### Epochs [{}/{}] --- batch[{}/{}] --- acc {:.4} --- loss {:.4}".format(epoch + 1, self.epochs, i, len(train_iter), acc, loss.item()), model_name, self.dataset)
                    animator.add(epoch + (i + 1) / len_train_iter, (loss.detach().cpu(), acc.cpu(), None))
                    # writer.add_scalar('Training/Accuracy', acc, scheduler.last_epoch)
                # writer.add_scalar('Training/Loss', loss.item(), scheduler.last_epoch)
                # writer.add_scalar('Training/Learning Rate', scheduler.get_last_lr()[0], scheduler.last_epoch)
            time_sum += time.time() - start

            # test_acc, all_logits, y_labels, label_img = self.evaluate(test_iter)
            test_acc = self.evaluate(test_iter)
            print("### Epochs [{}/{}] -- Acc on test {:.4}".format(epoch + 1, self.epochs, test_acc))
            printlog("### Epochs [{}/{}] -- Acc on test {:.4}".format(epoch + 1, self.epochs, test_acc), model_name, self.dataset)
            animator.add(epoch + 1, (None, None, test_acc))
            # writer.add_scalar('Testing/Accuracy', test_acc, scheduler.last_epoch)
            # writer.add_embedding(mat=all_logits,       # 所有点
            #                      metadata=y_labels,    # 标签名称
            #                      label_img=label_img,  # 标签图片
            #                      global_step=scheduler.last_epoch)

            # robust_acc, all_logits, y_labels, label_img = self.verify(test_iter)
            # print("### Epochs [{}/{}] -- Robust Acc on test {:.4}".format(epoch + 1, self.epochs, robust_acc))
            # printlog("### Epochs [{}/{}] -- Robust Acc on test {:.4}".format(epoch + 1, self.epochs, robust_acc), model_name)
            # writer.add_scalar('Verifying/Accuracy', robust_acc, scheduler.last_epoch)

            if test_acc > max_test_acc:
                max_test_acc = test_acc
                state_dict = self.model.state_dict()
            torch.save({'last_epoch': scheduler.last_epoch,
                        'model_state_dict': state_dict},
                       self.model_save_path)
        print("### max test accuracy: {:.4}".format(max_test_acc))
        print("### error: {:.4}%".format((1 - max_test_acc) * 100))
        print("### time of per epoch: {:.4}".format(time_sum / epoch))
        printlog("### max test accuracy: {:.4}".format(max_test_acc), model_name, self.dataset)
        printlog("### error: {:.4}%".format((1 - max_test_acc) * 100), model_name, self.dataset)
        printlog("### time of per epoch: {:.4}".format(time_sum / epoch), model_name, self.dataset)
        d2l.plt.savefig(f'{self.model_save_path[:-3]}.png')

    def evaluate(self, data_iter):
        self.model.eval()
        # all_logits = []
        # y_labels = []
        # images = []
        with torch.no_grad():
            acc_sum, n = 0.0, 0
            for x, y in data_iter:

                if self.fnn:
                    if self.dataset == 'mnist':
                        x_fnn = x.view(-1, 784)
                    elif self.dataset == 'cifar':
                        x_fnn = x.view(-1, 3 * 32 * 32)
                    x_abstract = abstract_data(x_fnn, self.interval_num)
                else:
                    x_abstract = abstract_data(x, self.interval_num)

                x_abstract, y = x_abstract.to(self.device), y.to(self.device)
                logits = self.model(x_abstract)

                acc_sum += (logits.argmax(1) == y).float().sum().item()
                n += len(y)

                # all_logits.append(logits)
                # y_pred = logits.argmax(1).view(-1)
                # if self.dataset == 'mnist':
                #     y_labels += (mnist_text_labels[i] for i in y_pred)
                # elif self.dataset == 'cifar':
                #     y_labels += (cifar_text_labels[i] for i in y_pred)
                # images.append(x)
            self.model.train()
            # 在imagenet上这里后面的数据如果不注释掉，会报错内存溢出（需要分配28G显存）
            return acc_sum / n  # , torch.cat(all_logits, dim=0), y_labels, torch.cat(images, dim=0)

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
                        x_fnn = x.view(-1, 3 * 32 * 32)
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
    如果文件夹不存在就创建
    :param filepath:需要创建的文件夹路径
    '''
    if not os.path.exists(filepath):
        os.mkdir(filepath)


if __name__ == '__main__':

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
        model_struc = AlexNet()
    elif structure == 'VGG':  # IMAGENET
        model_struc = VGG11()
    elif structure == 'ResNet':  # IMAGENET
        model_struc = ResNet18()
    fnn = args.fnn == 'True'
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

    optimizer = args.optimizer
    weight_decay = args.weight_decay
    momentum = args.momentum
    init = args.init

    print("--dataset {}, --model_name {}, --epsilon {:.6f}, --k {}, --batch_size {}, --epochs {}, --learning_rate {:.5f}".format(dataset, model_name, epsilon, k, batch_size, epochs, learning_rate))
    printlog("--dataset {}, --model_name {}, --epsilon {:.6f}, --k {}, --batch_size {}, --epochs {}, --learning_rate {:.5f}".format(dataset, model_name, epsilon, k, batch_size, epochs, learning_rate), model_name, dataset)

    model = RobustModel(model_path, log_path, model_struc, dataset=dataset, fnn=fnn, interval_num=interval_num, epsilon=epsilon, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate,
                        optimizer=optimizer, weight_decay=weight_decay, momentum=momentum, init=init)
    model.train()


