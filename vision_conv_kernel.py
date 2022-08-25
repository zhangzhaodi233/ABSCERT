import torch
import my_models_define
from d2l import torch as d2l
import os

# Visualization of convolution kernels for two adjacent channels
# We implement the process for DM_Small on MNIST

def draw_conv_kernel(model_path = "exp_results/mnist_dm_small_1.0.pt", 
                        dataset = 'MNIST', model_name = 'DM_Small'):
    if dataset == 'MNIST':
        if model_name == 'DM_Small':
            net = my_models_define.DM_Small(2, 28)
    net.load_state_dict(torch.load(model_path)['model_state_dict'])
    for layer in net.conv:
        if type(layer) == torch.nn.Conv2d:
            w = layer.weight.data.permute(1,0,2,3)
            x, y = w[0].reshape(-1), w[1].reshape(-1)
            idx = [i for i in range(x.shape[0])]
            figure, ax = d2l.plt.subplots(figsize=(10,3))
            A, = d2l.plt.plot(idx, x, '-', label='layer 0')
            B, = d2l.plt.plot(idx, y, 'm--', label='layer 1')
            legend = d2l.plt.legend(handles=[A,B], prop={'size':18})
            d2l.plt.tick_params(labelsize=20)     # Resize axis numbers
            d2l.plt.grid()  # Display grid lines
            model_name = model_path.split('/')[-1][:-3]
            d2l.plt.savefig(f'vision_conv_kernel_images/{model_name}_kernel.png')
        
            break

if __name__ == '__main__':

    if not os.path.exists('vision_conv_kernel_images'):
        os.mkdir('vision_conv_kernel_images')
    draw_conv_kernel()
    

