import os
import torch
from TrainRobustNN.etc import datasets
from TrainRobustNN.etc import conv_models_define
from d2l import torch as d2l



# 找到所有的反例并绘制原图
def generate_counter_example(test_iter, model, d):
    for i, (x,y) in enumerate(test_iter):
        origin_x = x.clone()
        x = datasets.abstract_data(x, 2.0/d)
        x, y = x.to('cuda:0'), y.to('cuda:0')
        y_hat = model(x)
        if y_hat.argmax(1) != y:
            # 绘制原图
            ytrue = y.to('cpu').detach().numpy()[0]
            yhat = y_hat.argmax(1).to('cpu').detach().numpy()[0]
            d2l.plt.imshow(origin_x.squeeze(0).squeeze(0))
            d2l.plt.savefig(f'output/counter_examples/yhat_{yhat}_ytrue_{ytrue}_id_{i}.png')
            d2l.plt.clf()


def draw_same_examples(test_iter, model, d):
    count = 0
    for i, (x,y) in enumerate(test_iter):
        origin_x = x.clone()
        x = datasets.abstract_data(x, 2.0/d)
        x, y = x.to('cuda:0'), y.to('cuda:0')
        y_hat = model(x)
        if y == 9 and y_hat.argmax(1) == y:
            ytrue = y.to('cpu').detach().numpy()[0]
            yhat = y_hat.argmax(1).to('cpu').detach().numpy()[0]
            d2l.plt.imshow(origin_x.squeeze(0).squeeze(0))
            d2l.plt.savefig(f'output/same_examples/yhat_{yhat}_ytrue_{ytrue}_id_{i}.png')
            d2l.plt.clf()
            count += 1


if __name__ == "__main__":
    os.makedirs("output/counter_examples/", exist_ok=True)
    os.makedirs("output/same_examples/", exist_ok=True)
    dataset = 'mnist'
    model = 'DM_Small'
    in_ch, in_dim = 1, 28
    pretrained_model = "output/models/mnist_dm_small_0.1.pt"

    _, test_iter = datasets.load_dataset(1, dataset)
    if model == "DM_Small":
        model = conv_models_define.DM_Small(in_ch*2, in_dim)
    elif model == "DM_Medium":
        model = conv_models_define.DM_Medium(in_ch*2, in_dim)
    elif model == "DM_Large":
        model = conv_models_define.DM_Large(in_ch*2, in_dim)
    
    model.load_state_dict(torch.load(pretrained_model)['model_state_dict'])
    model = model.to('cuda:0')
    d = float(pretrained_model.split('_')[-1][:-3])
    generate_counter_example(test_iter, model, d)
    draw_same_examples(test_iter, model, d)


# 找规律 看这些同一错误分类的抽象状态是否有某种规律，然后在训练的过程中是否可以将这种“规律“规避掉以提高精度
# 在规避的过程中一定要考虑到“对之前的映射函数会有什么影响？映射函数是否会发生改变”
# 所做的一切都要保证：抽象状态之间不能有交集，抽象状态组合起来是整个状态空间。

