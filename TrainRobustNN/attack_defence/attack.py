import torch
import TrainRobustNN.etc.datasets as datasets
import TrainRobustNN.etc.conv_models_define as conv_models_define
import numpy as np


def attack_abstract_data(x, interval_num):
    # 依据最近邻攻击法抽象出的抽象状态
    x_upper = torch.zeros_like(x)
    x_lower = torch.zeros_like(x)

    step = (1-(-1))/interval_num
    k = torch.div((x - (-1)), step, rounding_mode='floor')
    x_lower = -1 + k * step
    x_lower = torch.clamp(x_lower, -1, 1)
    x_upper = x_lower + step
    x_upper = torch.clamp(x_upper, -1, 1)

    # 改进，解决如果像素为1，映射到[1,1]的情况
    eq = torch.eq(x_lower, x_upper)
    x_lower = x_lower - step * eq.int()
    x_lower = torch.clamp(x_lower, min=-1)

    # 攻击，现在要映射到最近的别的抽象状态上
    larger = ((x_upper - x) < (x - x_lower)).int()  # 如果x更偏向x_upper，为True，否则为False
    uupper, ulower = torch.clamp(x_upper + step, max=1), torch.clamp(x_lower + step, max=1)
    eq = torch.eq(ulower, uupper)
    ulower = ulower - step * eq.int()
    ulower = torch.clamp(ulower, min=-1)
    lupper, llower = torch.clamp(x_upper - step, min=-1), torch.clamp(x_lower - step, min=-1)
    eq = torch.eq(llower, lupper)
    lupper = lupper + step * eq.int()
    lupper = torch.clamp(lupper, max=1)

    x_upper = uupper * larger + lupper * (1-larger)
    x_lower = ulower * larger + llower * (1-larger)
    x_result = torch.cat((x_upper, x_lower), dim=1)
    return x_result, larger*2-1  # (batch_size, 2*channels, height, width),  (batch_size, channels, height, width)


def abstract_PGDattack(model, epsilon, iter, d):
    # 如果扰动区间跨两个抽象状态，并且梯度增加的方向正是另一个抽象状态，则扰动
    # 不生成原图，而生成抽象状态组成的图，并检验是否分类正确
    
    correct_origin, correct_new = 0, 0
    n = 0
    for i, (x, y) in enumerate(iter):
        channel = x.shape[1]
        abstract_x = datasets.abstract_data(x, int(2.0 / d))
        abstract_x.requires_grad = True
        loss, logits = model(abstract_x, y)
        grad = torch.autograd.grad(loss, abstract_x)[0]  # batch_size, channel*2, height, width
        # 确定梯度增加的方向，1表示增大，-1表示减小
        grad_increase_direction = ((grad[:, :channel, :, :] > 0) & (grad[:, channel:, :, :] > 0) | \
            (grad[:, :channel, :, :] > 0) & (grad[:, channel:, :, :] < 0) & (grad[:, :channel, :, :] > grad[:, channel:, :, :]) | \
            (grad[:, :channel, :, :] < 0) & (grad[:, channel:, :, :] > 0) & (grad[:, :channel, :, :] < grad[:, channel:, :, :])).int() * 2 - 1
        # 扰动区间
        x_interval = torch.cat((torch.clamp(x + epsilon, max=1), torch.clamp(x - epsilon, min=-1)), dim=1)
        # 如果抽象状态上界 < 扰动区间上界 or 抽象状态下界 > 扰动区间下界，则为true，意为扰动，否则不扰动
        mask = (abstract_x[:, :channel, :, :] < x_interval[:, :channel, :, :]) | (abstract_x[:, channel:, :, :] > x_interval[:, channel:, :, :])

        attack_abstract_x, larger = attack_abstract_data(x, int(2.0/d))
        # 如果扰动逆梯度，则不扰动
        if_eps = torch.clamp(grad_increase_direction * larger, min=0)  # 梯度考虑0不扰动，1扰动
        mask = if_eps * mask.int()  # 0不扰动，1扰动
        mask = mask.repeat((1,2,1,1))  # 通道数加倍
        new_abstract_x = attack_abstract_x * mask + abstract_x * (1-mask)

        correct_origin += (logits.argmax(1) == y).float().sum().item()
        n += len(y)
        y_hat = model(new_abstract_x)
        correct_new += (y_hat.argmax(1) == y).float().sum().item()
    return round(correct_origin / n, 4), round(correct_new / n, 4)


if __name__ == '__main__':
    _, test_iter = datasets.load_dataset(64, dataset='mnist')
    model = conv_models_define.DM_Small(2, 28)
    pretrained_model = "../output/models/mnist_dm_small_0.1.pt"
    model.load_state_dict(torch.load(pretrained_model)['model_state_dict'])
    print(abstract_PGDattack(model, 0.05, test_iter, 0.1))

    _, test_iter = datasets.load_dataset(64, dataset='mnist')
    model = conv_models_define.DM_Medium(2, 28)
    pretrained_model = "../output/models/mnist_dm_medium_0.1.pt"
    model.load_state_dict(torch.load(pretrained_model)['model_state_dict'])
    print(abstract_PGDattack(model, 0.05, test_iter, 0.1))

    _, test_iter = datasets.load_dataset(64, dataset='mnist')
    model = conv_models_define.DM_Large(2, 28)
    pretrained_model = "../output/models/mnist_dm_large_0.1.pt"
    model.load_state_dict(torch.load(pretrained_model)['model_state_dict'])
    print(abstract_PGDattack(model, 0.05, test_iter, 0.1))

    _, test_iter = datasets.load_dataset(64, dataset='cifar')
    model = conv_models_define.DM_Small(6, 32)
    pretrained_model = "../output/models/cifar_dm_small_0.1.pt"
    model.load_state_dict(torch.load(pretrained_model)['model_state_dict'])
    print(abstract_PGDattack(model, 0.05, test_iter, 0.1))

    _, test_iter = datasets.load_dataset(64, dataset='cifar')
    model = conv_models_define.DM_Medium(6, 32)
    pretrained_model = "../output/models/cifar_dm_medium_0.1.pt"
    model.load_state_dict(torch.load(pretrained_model)['model_state_dict'])
    print(abstract_PGDattack(model, 0.05, test_iter, 0.1))

    _, test_iter = datasets.load_dataset(64, dataset='cifar')
    model = conv_models_define.DM_Large(6, 32)
    pretrained_model = "../output/models/cifar_dm_large_0.1.pt"
    model.load_state_dict(torch.load(pretrained_model)['model_state_dict'])
    print(abstract_PGDattack(model, 0.05, test_iter, 0.1))