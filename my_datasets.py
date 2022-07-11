from re import L
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

def load_dataset(batch_size=64, dataset='mnist'):

    # transforms.ToTensor() 只是将数据归一化到 [0, 1]
    # transforms.Normalize() 则是将数据归一化到 [-1, 1]
    if dataset == 'mnist':
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        
        data_train = torchvision.datasets.MNIST(root='data/MNIST', 
                                                train=True, download=True, 
                                                transform=trans)
        data_test = torchvision.datasets.MNIST(root='data/MNIST', 
                                                train=False, download=True, 
                                                transform=trans)
    elif dataset == 'cifar':
        std = [0.2023, 0.1994, 0.2010]
        mean = [0.4914, 0.4822, 0.4465]
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = mean, std = std)])
        
        data_train = torchvision.datasets.CIFAR10(root='data/CIFAR10', 
                                                train=True, download=True, 
                                                transform=trans)
        data_test = torchvision.datasets.CIFAR10(root='data/CIFAR10', 
                                                train=False, download=True, 
                                                transform=trans)
    
    train_iter = torch.utils.data.DataLoader(data_train,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers= 1) # num_workers 工作者数量，默认是0。使用多少个子进程来导入数据。设置为0，就是使用主进程来导入数据
    test_iter = torch.utils.data.DataLoader(data_test,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=1)
    return train_iter, test_iter

def abstract_data(x, interval_num):

    x_upper = torch.zeros_like(x)
    x_lower = torch.zeros_like(x)

    step = (1-(-1))/interval_num
    k = torch.div((x - (-1)), step, rounding_mode='floor')
    x_lower = -1 + k * step
    x_upper = x_lower + step

    x_result = torch.cat((x_upper, x_lower), dim=1)
    return x_result

# 如何将扰动区间映射到抽象区间
def abstract_disturbed_data(x, interval_num, epsilon):

    # 比如 x = (-0.55, 0.55), eps = 0.1, 则 x 的扰动区间为 ([-0.65, -0.45][0.45, 0.65])
    # 那么每个像素的扰动区间分别映射为 [-1, -0.5], [-0.5, 0] 和 [0, 0.5], [0.5, 1]
    # 这时，将每个像素的抽象子区间进行合并，合并为 [-1, 0] 和 [0, 1]
    # 这样既解决了空间爆炸问题，验证的时候又包含了所有情况。只不过可能对分类精确度和鲁棒精确度有所影响，后期根据实验结果再进行调整。

    # 计算被扰动之后下界的抽象区间
    x_lower = x - epsilon
    
    step = (1-(-1))/interval_num
    k = torch.div((x_lower - (-1)), step, rounding_mode='floor')
    x_lower_abstract_lower = -1 + k * step 

    # 计算扰动之后上界的抽象区间
    x_upper = x + epsilon
    
    k = torch.div((x_upper - (-1)), step, rounding_mode='floor')
    x_upper_abstract_lower = -1 + k * step
    x_upper_abstract_upper = x_upper_abstract_lower + step 

    # 取扰动区间下界的抽象区间的下界，取扰动区间上界的抽象区间的上界
    x_result = torch.cat((x_lower_abstract_lower, x_upper_abstract_upper), dim=1)

    return x_result