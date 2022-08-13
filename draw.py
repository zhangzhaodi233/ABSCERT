from d2l import torch as d2l
import json

# 画图工具，对应实验3

def draw(file):
    record = []
    with open(file, 'r') as f:
        for line in f.readlines():
            if '[[' in line:
                record = json.loads(line)
                break

    d2l.plt.plot([data[0] for data in record], [data[1] for data in record],
            color='red', linewidth=1.0, linestyle='-')
    if 'mnist' in file:
        d2l.plt.ylim(0.95, 1)
    elif 'cifar' in file:
        d2l.plt.ylim(0.3, 1)
    d2l.plt.title(file.split('/')[-1].split('.')[0])
    d2l.plt.xlabel('Abstract Granularity')
    d2l.plt.ylabel('Test acc')

if __name__ == '__main__':
    d2l.set_figsize((6, 4))
    d2l.plt.subplot(2,3,1)
    draw('result/mnist_dm_small.log')
    d2l.plt.subplot(2,3,2)
    draw('result/mnist_dm_medium.log')
    d2l.plt.subplot(2,3,3)
    draw('result/mnist_dm_large.log')
    d2l.plt.subplot(2,3,4)
    draw('result/cifar_dm_small.log')
    d2l.plt.subplot(2,3,5)
    draw('result/cifar_dm_medium.log')
    d2l.plt.subplot(2,3,6)
    draw('result/cifar_dm_large.log')
    d2l.plt.tight_layout(pad=1.08)
    d2l.plt.savefig('summary.png')