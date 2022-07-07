from re import L
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

def load_dataset(batch_size=64, abstra_param=0.05):

    # transforms.ToTensor() 只是将数据归一化到 [0, 1]
    # transforms.Normalize() 则是将数据归一化到 [-1, 1]
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    mnist_train = torchvision.datasets.MNIST(root='data/MNIST', 
                                             train=True, download=True, 
                                             transform=trans)
    mnist_test = torchvision.datasets.MNIST(root='data/MNIST', 
                                            train=False, download=True, 
                                            transform=trans)
    
    train_iter = torch.utils.data.DataLoader(mnist_train,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers= 1) # num_workers 工作者数量，默认是0。使用多少个子进程来导入数据。设置为0，就是使用主进程来导入数据
    test_iter = torch.utils.data.DataLoader(mnist_test,
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

def abstract_disturbed_data(x, interval_num, epsilon, y):

    # 比如 x = 0.425, eps = 0.4, 则 x' 属于 [0.025, 0.825]
    # 其实并不需要考虑划分区间，只需要分别计算 0.025 和 0.825 的抽象区间即可
    # 相当于一张图片变成了两张图片进行验证
    # 暂时只考虑扰动后的区间只会跨两个抽象区间（如果是跨三个抽象区间，其实就是将图片根据跨的分界点变成三张图片进行验证）

    # 计算被扰动之后下界的抽象区间
    x_lower = x - epsilon
    
    step = (1-(-1))/interval_num
    k = torch.div((x_lower - (-1)), step, rounding_mode='floor')
    x_lower_abstract_lower = -1 + k * step
    x_lower_abstract_upper = x_lower_abstract_lower + step

    x_lower_abstract = torch.cat((x_lower_abstract_upper, x_lower_abstract_lower), dim=1)

    # 计算扰动之后上界的抽象区间
    x_upper = x + epsilon
    
    k = torch.div((x_upper - (-1)), step, rounding_mode='floor')
    x_upper_abstract_lower = -1 + k * step
    x_upper_abstract_upper = x_upper_abstract_lower + step

    x_upper_abstract = torch.cat((x_upper_abstract_upper, x_upper_abstract_lower), dim=1)

    x_result = torch.cat((x_upper_abstract, x_lower_abstract), dim=0)

    y_result = torch.cat((y, y), dim=0)

    return x_result, y_result