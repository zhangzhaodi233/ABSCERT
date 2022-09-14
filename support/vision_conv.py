import math
import os
import torch
from d2l import torch as d2l
from utils.conv_models_define import *
from utils import datasets

# Visualization of the output features for convolution layer
# We implement the process for DM_Small on ImageNet

def draw_vision_conv(dataset, model_name, model_path, interval, abstract=True):
    _, test_iter = datasets.load_dataset(1, dataset)
    if model_name == 'DM_Small':
        if dataset == 'mnist':
            if abstract:
                model = DM_Small(2, 28)
            else:
                model = DM_Small(1, 28)
    state_dict = torch.load(model_path)['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    for i, data in enumerate(test_iter):
        if dataset == 'mnist':
            x, y = data
            d2l.plt.imshow(x.squeeze(0).squeeze(0))
            d2l.plt.savefig(f'output/vision_conv/{y.item()}_ori.png')
            x_abstract = datasets.abstract_data(x, 2//interval)
            conv_out = model.conv(x_abstract).squeeze(0)

            fc = model.fc[1].weight.data.permute(1,0)  # [8192, 100]
            fc_sum = torch.sum(fc, dim=1)
            fs = torch.zeros(32)
            j = 0
            for i in range(32):
                for j in range(256):
                    fs[i] += fc_sum[i*j]
            sort_fs, indices = torch.sort(fs)
            img = torch.zeros((16,16))
            for i, indice in enumerate(indices[:32]):
                img += conv_out[indice] * sort_fs[i]


            d2l.plt.imshow(img.detach().numpy())
            d2l.plt.axis('off')
            d2l.plt.savefig(f'output/vision_conv/{y.item()}_{interval}_rob.png')

            break


if __name__ == '__main__':
    os.makedirs("output/vision_conv/", exist_ok=True)
    draw_vision_conv('mnist', 'DM_Small', 'exp_results/mnist_dm_small_1.0.pt', 1.0, True)
    draw_vision_conv('mnist', 'DM_Small', 'exp_results/mnist_dm_small_0.025.pt', 0.025, True)
