import math
import os
import torch
from d2l import torch as d2l
from TrainRobustNN.utils.conv_models_define import *
from TrainRobustNN.utils import datasets
from TrainRobustNN.utils.mapping_func import abstract_data

# Visualization of the output features for convolution layer
# We implement the process for DM_Small on ImageNet

def draw_vision_conv(dataset, model_name, model_path, d, abstract=True):
    _, test_iter = datasets.load_dataset(1, dataset)
    if model_name == 'DM_Small':
        if dataset == 'mnist':
            if abstract:
                model = DM_Small(2, 28)
            else:
                model = DM_Small(1, 28)
        elif dataset == 'cifar':
            if abstract:
                model = DM_Small(6, 32)
            else:
                model = DM_Small(3, 32)
    elif model_name == 'AlexNet':
        if abstract:
            model = AlexNet(6)
        else:
            model = AlexNet(3)
    state_dict = torch.load(model_path)['model_state_dict']
    model.load_state_dict(state_dict)
    model.eval()

    for i, data in enumerate(test_iter):
        if i < 17:
            continue
        if i > 17:
            break
        if dataset == 'mnist':
            x, y = data
            d2l.plt.imshow(x.squeeze(0).squeeze(0))
            d2l.plt.savefig(f'output/vision_conv/mnist_{y.item()}_ori.png')
            if abstract:
                d0 = 2 if d == 0 else d
                x_abstract = abstract_data(x, 2//d0, True)
                conv_out = model.conv(x_abstract).squeeze(0)
            else:
                conv_out = model.conv(x).squeeze(0)

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
                img += conv_out[indice] # * sort_fs[i]

            print(torch.var(img))

            d2l.plt.imshow(img.detach().numpy())
            d2l.plt.axis('off')
            d2l.plt.savefig(f'output/vision_conv/mnist_{y.item()}_{d}_rob.png')

            # break
        
        elif dataset == 'cifar':
            x, y = data
            d2l.plt.imshow(x.squeeze(0).permute(1,2,0))
            d2l.plt.savefig(f'output/vision_conv/cifar_{y.item()}_ori.png')
            
            if abstract:
                x_abstract = abstract_data(x, 2//d)
                conv_out = model.conv(x_abstract).squeeze(0)
            else:
                conv_out = model.conv(x).squeeze(0)

            fc = model.fc[1].weight.data.permute(1,0)  # [8192, 100]
            fc_sum = torch.sum(fc, dim=1)
            fs = torch.zeros(32)
            j = 0
            for i in range(32):
                for j in range(324):
                    fs[i] += fc_sum[i*j]
            sort_fs, indices = torch.sort(fs)
            img = torch.zeros((18,18))
            for i, indice in enumerate(indices[:32]):
                img += conv_out[indice] * sort_fs[i]


            d2l.plt.imshow(img.detach().numpy())
            d2l.plt.axis('off')
            d2l.plt.savefig(f'output/vision_conv/cifar_{y.item()}_{d}_rob.png')

            break

        elif dataset == 'imagenet':
            x, y = data
            d2l.plt.imshow(x.squeeze(0))
            d2l.plt.savefig(f'output/vision_conv/imagenet_{y.item()}_ori.png')
            x_abstract = abstract_data(x, 2//d)
            conv_out = model.conv(x_abstract).squeeze(0)  # 256*5*5

            d2l.show_images(conv_out, 80, 80)
            d2l.plt.savefig(f'output/vision_conv/imagenet_{y.item()}_{d}_features.png')

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
            d2l.plt.savefig(f'output/vision_conv/imagenet_{y.item()}_{d}_feature_map.png')

            break


if __name__ == '__main__':
    os.makedirs("output/vision_conv/", exist_ok=True)
    draw_vision_conv('mnist', 'DM_Small', 'output/models/mnist_dm_small_1.0.pt', 0.0, True)
    draw_vision_conv('mnist', 'DM_Small', 'output/models/mnist_dm_small_0.025.pt', 0.025, True)
    # draw_vision_conv('imagenet', 'AlexNet', 'output/models/imagenet_alexnet_1.0.pt', 1.0, True)
    # draw_vision_conv('imagenet', 'AlexNet', 'output/models/imagenet_alexnet_0.025.pt', 0.025, True)


