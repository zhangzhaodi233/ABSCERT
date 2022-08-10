import math
from os import mkdir
import torch
from d2l import torch as d2l
from my_models_define import *
import my_datasets

def draw_vision_conv(dataset, model_name, model_path, interval, abstract=True):
    _, test_iter = my_datasets.load_dataset(1, dataset)
    if model_name == 'DM_Small':
        if dataset == 'mnist':
            if abstract:
                model = DM_Small(2, 28)
            else:
                model = DM_Small(1, 28)
    state_dict = torch.load(model_path)['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    for data in test_iter:
        if dataset == 'imagenet':
            break
        else:
            x, y = data
            d2l.plt.imshow(x.squeeze(0).squeeze(0))
            d2l.plt.savefig(f'vision_conv_images/{y.item()}_ori.png')
            x_abstract = my_datasets.abstract_data(x, 2//interval)
            conv_out = model.conv(x_abstract).squeeze(0)
            d2l.show_images(conv_out.detach(), 4, math.ceil(conv_out.shape[0]/4), 
                titles=[i for i in range(conv_out.shape[0])])
            d2l.plt.savefig(f"vision_conv_images/{y.item()}_conv_{model_path.split('/')[-1].split('.')[0]}.png")
            break


if __name__ == '__main__':
    draw_vision_conv('mnist', 'DM_Small', 'exp_results/mnist_small_0_008.pt', 0.008, True)
