from d2l import torch as d2l
import json
import os

# Draw the verification error with the general increase of t
# Results are shown in experiment 3

def draw(file):
    record = []
    with open(file, 'r') as f:
        for line in f.readlines():
            if '[[' in line:
                record = json.loads(line)
                break

    
    if 'mnist' in file:
        d2l.plt.plot([data[0] for data in record], [1 - data[1] for data in record],
            color='green', linewidth=2.0, linestyle='-')
        d2l.plt.xlim(-0.1, 1.1)
        d2l.plt.ylim(0.003, 0.018)
        d2l.plt.xticks([0, 0.5, 1])
        d2l.plt.yticks([0.005, 0.01, 0.015])
    elif 'cifar' in file:
        d2l.plt.plot([data[0] for data in record], [1 - data[1] for data in record],
            color='blue', linewidth=2.0, linestyle='-')
        d2l.plt.xlim(-0.1, 1.1)
        d2l.plt.ylim(0.05, 0.45)
        d2l.plt.xticks([0, 0.5, 1])
        d2l.plt.yticks([0.1, 0.2, 0.3, 0.4])
    else:
        d2l.plt.plot([data[0] for data in record], [1 - data[1] for data in record],
            color='red', linewidth=2.0, linestyle='-')
        d2l.plt.xlim(-0.1, 1.1)
        d2l.plt.ylim(0.35, 0.65)
        d2l.plt.xticks([0, 0.5, 1])

    # font = {
    #     'family' : 'Times New Roman',
    #     'weight' : 'normal',
    #     'size'   : 27,
    # }
    # d2l.plt.xlabel('Abstract Granularity', font)
    # d2l.plt.ylabel('Err of DM-medium(%)', font)
    d2l.plt.tick_params(labelsize=30)
    # d2l.plt.grid()

if __name__ == '__main__':
    os.makedirs("output/vision_d/")

    d2l.set_figsize((5, 4))

    draw('output/log/mnist_dm_small.log')
    d2l.plt.tight_layout(pad=1.08)
    d2l.plt.savefig('output/vision_d/summary_mnist_dmsmall.png')
    d2l.plt.clf()

    draw('output/log/mnist_dm_medium.log')
    d2l.plt.tight_layout(pad=1.08)
    d2l.plt.savefig('output/vision_d/summary_mnist_dmmedium.png')
    d2l.plt.clf()

    draw('output/log/mnist_dm_large.log')
    d2l.plt.tight_layout(pad=1.08)
    d2l.plt.savefig('output/vision_d/summary_mnist_dmlarge.png')
    d2l.plt.clf()


    d2l.set_figsize((5, 4.2))

    draw('output/log/cifar_dm_small.log')
    d2l.plt.tight_layout(pad=1.08)
    d2l.plt.savefig('output/vision_d/summary_cifar_dmsmall.png')
    d2l.plt.clf()

    draw('output/log/cifar_dm_medium.log')
    d2l.plt.tight_layout(pad=1.08)
    d2l.plt.savefig('output/vision_d/summary_cifar_dmmedium.png')
    d2l.plt.clf()

    draw('output/log/cifar_dm_large.log')
    d2l.plt.tight_layout(pad=1.08)
    d2l.plt.savefig('output/vision_d/summary_cifar_dmlarge.png')
    d2l.plt.clf()

    draw('output/log/imagenet_alexnet.log')
    d2l.plt.tight_layout(pad=1.08)
    d2l.plt.savefig('output/vision_d/summary_imagenet_alexnet.png')
    d2l.plt.clf()

    draw('output/log/imagenet_vgg11.log')
    d2l.plt.tight_layout(pad=1.08)
    d2l.plt.savefig('output/vision_d/summary_imagenet_vgg11.png')
    d2l.plt.clf()

    draw('output/log/imagenet_resnet18.log')
    d2l.plt.tight_layout(pad=1.08)
    d2l.plt.savefig('output/vision_d/summary_imagenet_resnet18.png')
    d2l.plt.clf()

    draw('output/log/imagenet_resnet34.log')
    d2l.plt.tight_layout(pad=1.08)
    d2l.plt.savefig('output/vision_d/summary_imagenet_resnet34.png')
    d2l.plt.clf()

    draw('output/log/mnist_fc3.log')
    d2l.plt.tight_layout(pad=1.08)
    d2l.plt.savefig('output/vision_d/summary_mnist_fc3.png')
    d2l.plt.clf()

    draw('output/log/cifar_fc3.log')
    d2l.plt.tight_layout(pad=1.08)
    d2l.plt.savefig('output/vision_d/summary_cifar_fc3.png')
    d2l.plt.clf()

