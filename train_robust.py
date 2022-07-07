import torch
import torch.nn as nn

from transformers import get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

import tensorflow as tf
import tensorboard as tb

import os

from my_models_define import *
from my_datasets import load_dataset, abstract_data, abstract_disturbed_data

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

text_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


class RobustModel:
    def __init__(self, model_path, log_path, model, dataset, fnn=False, interval_num=1, epsilon=0, batch_size=256, epochs=10, learning_rate=0.01):
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

    def train(self):
        train_iter, test_iter = load_dataset(self.batch_size)
        last_epoch = -1
        # 接着之前的模型继续训练
        # if os.path.exists('exp_results/lenet5_test.pt'):
        #     checkpoint = torch.load('exp_results/lenet5_test.pt')
        #     last_epoch = checkpoint['last_epoch']
        #     self.model.load_state_dict(checkpoint['model_state_dict'])

        num_training_steps = len(train_iter) * self.epochs
        optimizer = torch.optim.Adam([{
            "params": self.model.parameters(), "initial_lr": self.learning_rate
        }])
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=300,
                                                    num_training_steps=num_training_steps,
                                                    num_cycles=2, last_epoch=last_epoch)
        
        writer = SummaryWriter(self.log_path)
        self.model = self.model.to(self.device)
        max_test_acc = 0
        for epoch in range(self.epochs):
            for i, (x, y) in enumerate(train_iter):
                if self.fnn:
                    if self.dataset == 'mnist':
                        x = x.view(-1, 784)
                
                x, y = abstract_disturbed_data(x, self.interval_num, self.epsilon, y)
                # x = abstract_data(x, self.interval_num)
                x, y = x.to(self.device), y.to(self.device)
                loss, logits = self.model(x, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                if i%50 == 0:
                    acc = (logits.argmax(1) == y).float().mean()
                    print("### Epochs [{}/{}] --- batch[{}/{}] --- acc {:.4} --- loss {:.4}".format(epoch+1, self.epochs, i, len(train_iter), acc, loss.item()))
                    writer.add_scalar('Training/Accuracy', acc, scheduler.last_epoch)
                writer.add_scalar('Training/Loss', loss.item(), scheduler.last_epoch)
                writer.add_scalar('Training/Learning Rate', scheduler.get_last_lr()[0], scheduler.last_epoch)

            test_acc, all_logits, y_labels, label_img = self.evaluate(test_iter)
            print("### Epochs [{}/{}] -- Acc on test {:.4}".format(epoch + 1, self.epochs, test_acc))
            writer.add_scalar('Testing/Accuracy', test_acc, scheduler.last_epoch)
            writer.add_embedding(mat=all_logits,       # 所有点
                                 metadata=y_labels,    # 标签名称
                                 label_img=label_img,  # 标签图片
                                 global_step=scheduler.last_epoch)

            robust_acc, all_logits, y_labels, label_img = self.verify(test_iter)
            print("### Epochs [{}/{}] -- Robust Acc on test {:.4}".format(epoch + 1, self.epochs, robust_acc))
            writer.add_scalar('Verifying/Accuracy', robust_acc, scheduler.last_epoch)


            if test_acc > max_test_acc:
                max_test_acc = test_acc
                state_dict = self.model.state_dict()
            torch.save({'last_epoch': scheduler.last_epoch,
                        'model_state_dict': state_dict},
                        self.model_save_path)
    
    def evaluate(self, data_iter):
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
                    x_abstract = abstract_data(x_fnn, self.interval_num)
                else:
                    x_abstract = abstract_data(x, self.interval_num)
                    
                x_abstract, y = x_abstract.to(self.device), y.to(self.device)
                logits = self.model(x_abstract)
            
                acc_sum += (logits.argmax(1) == y).float().sum().item()
                n += len(y)

                all_logits.append(logits)
                y_pred = logits.argmax(1).view(-1)
                y_labels += (text_labels[i] for i in y_pred)
                images.append(x)
            self.model.train()
            return acc_sum / n, torch.cat(all_logits, dim=0), y_labels, torch.cat(images, dim=0)

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
                    x_disturbed_abstract, y = abstract_disturbed_data(x_fnn, self.interval_num, self.epsilon, y)
                else:
                    x_disturbed_abstract, y = abstract_disturbed_data(x, self.interval_num, self.epsilon, y)
          
                x_disturbed_abstract, y = x_disturbed_abstract.to(self.device), y.to(self.device)
                logits = self.model(x_disturbed_abstract)
                
                acc_sum += (logits.argmax(1) == y).float().sum().item()
                n += len(y)

                all_logits.append(logits)
                y_pred = logits.argmax(1).view(-1)
                y_labels += (text_labels[i] for i in y_pred)
                images.append(x)
            self.model.train()
            return acc_sum / n, torch.cat(all_logits, dim=0), y_labels, torch.cat(images, dim=0)


if __name__ == '__main__':

    ######## 训练 ########
    # 1. 根据抽象粒度，将输入映射到区间
    # 2. 将区间映射到输入层
    # 3. 训练得到 clean accuracy
    model_path_pre = 'exp_results/'
    model_name = 'DM_Small_MNIST'
    log_path_pre = 'runs/'
    model_path = model_path_pre + model_name
    log_path = log_path_pre + model_name

    in_ch = 1*2
    in_dim= 28
    width = 4
    model_struc = DM_Small(in_ch, in_dim, width)
    interval_num = 10
    epsilon = 0.1

    
    model = RobustModel(model_path, log_path, model_struc, dataset='mnist', interval_num=interval_num, epsilon=epsilon)
    model.train()

    ######## 验证 ########
    # 4. 测试集+epsilon 作为网络的输入进行 evaluate，得到 robust accuracy
    # 5. 如果 robust accuray 到达阈值，则结束训练；否则进行精华



    ######## 精华 ########
    # 6. 精华 - 划分抽象区间 - 再回到第一步