# imports
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# helper function to show an image
# (used in the 'plot_classes_preds' function below)
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

# helper function
def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

def main():
    # transforms
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # datasets
    trainset = torchvision.datasets.MNIST('./data',
        download=True,
        train=True,
        transform=transform
    )
    testset = torchvision.datasets.MNIST('./data',
        download=True,
        train=False,
        transform=transform
    )

    # dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)

    # constant for classes
    classes = ('0', '1', '2', '3', '4',
            '5', '6', '7', '8', '9')
    
    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # this line alone creates a runs/fashion_mnist_experiment_1 folder
    writer = SummaryWriter('runs/fashion_mnist_experiment_1')

    # get some random training images
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()

    # create grid of images
    # img_grid = torchvision.utils.make_grid(images)

    # show images
    # matplotlib_imshow(img_grid, one_channel=True)

    # write to tensorboard
    # # show images
    # writer.add_image('four_fashion_mnist_images', img_grid)

    # # show graph structure
    # writer.add_graph(net, images)

    # visualize the lower dimensional representation of higher dimensianl data
    # select random images and their target indices
    images, labels = select_n_random(trainset.data, trainset.targets)

    # get the class labels for each image
    class_labels = [classes[lab] for lab in labels]

    # log embeddings
    features = images.view(-1, 28 * 28)
    writer.add_embedding(features, label_img=images.unsqueeze(1), metadata=class_labels)
    writer.close()

if __name__ == "__main__":
    main()
