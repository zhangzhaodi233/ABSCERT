import torch
import torchvision
import torchvision.transforms as transforms
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy


class TrainPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_root, img_size, use_gpu=False):
        if not use_gpu:
            super(TrainPipeline, self).__init__(batch_size, num_threads, device_id,
                                                exec_async=False,
                                                exec_pipelined=False)
            mode = 'cpu'
            self.decode = ops.decoders.Image(device='cpu')  # A module for decoding images is defined in pipeline. The output format sequence is RGB 
        else:
            super(TrainPipeline, self).__init__(batch_size, num_threads, device_id, prefetch_queue_depth=4)
            mode = 'gpu'
            self.decode = ops.decoders.Image(device='mixed')

        self.img_size = img_size
        self.use_gpu = use_gpu
        # readers.File, like torchvision.datasets.ImageFolder
        self.input = ops.readers.File(file_root=data_root, random_shuffle=True)
        # Resize
        self.resize = ops.Resize(device=mode, resize_x=256, resize_y=256)
        # Randomcrop, like torchvision.transforms.RandomCrop
        self.randomcrop = ops.RandomResizedCrop(device=mode, size=img_size, random_area=[0.3, 1.0])
        # CropMirrorNormalize realize normalize and random horizontal flip, like torchvision.transforms.Normalize & RandomHorizontalFlip
        self.normalize = ops.CropMirrorNormalize(device=mode, mean=[0.5*255, 0.5*255, 0.5*255],
                                                 std=[0.5*255, 0.5*255, 0.5*255])  # normalize to [-1, 1]

        # A class that changes the color of an image, like torchvision.transforms.ColorJitter
        self.colortwist = ops.ColorTwist(device=mode)
        # A class that rotating an image, like torchvision.transforms.RandomRotation
        self.rotate = ops.Rotate(device=mode, fill_value=0)
        # gridmask，an operation for random block occlusion
        self.gridmask = ops.GridMask(device=mode)

    # The function is how to actually operate the data when calling the pipeline, 
    # similar to torch.module.forward()
    def define_graph(self):
        jpegs, labels = self.input(name='Reader')  # load data
        images = self.decode(jpegs)
        images = self.resize(images)
        images = self.randomcrop(images)
        images = self.normalize(images, mirror=0.5)
        return images, labels


class TestPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_root, img_size):
        super(TestPipeline, self).__init__(batch_size, num_threads, device_id, prefetch_queue_depth=4)

        self.decode = ops.decoders.Image(device='mixed')
        self.input = ops.readers.File(file_root=data_root, random_shuffle=False)
        self.resize = ops.Resize(device='gpu', resize_x=img_size, resize_y=img_size)
        self.normalize = ops.CropMirrorNormalize(device='gpu', mean=[0.5 * 255, 0.5 * 255, 0.5 * 255],
                                                 std=[0.5 * 255, 0.5 * 255, 0.5 * 255])

    def define_graph(self):
        jpegs, labels = self.input(name='Reader')
        images = self.decode(jpegs)
        images = self.resize(images)
        images = self.normalize(images)

        return images, labels


def get_dali_iter(batch_size, num_threads, device_id, train_data_root, test_data_root, img_size, use_gpu):
    pipe_train = TrainPipeline(batch_size, num_threads, device_id, train_data_root, img_size, use_gpu=use_gpu)
    pipe_train.build()
    # DALIClassificationIterator: Returns 2 outputs (data and label) in the form of PyTorch’s Tensor, like DataLoader
    train_loader = DALIClassificationIterator(pipe_train, size=pipe_train.epoch_size('Reader'),
                                                last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)

    pipe_test = TestPipeline(batch_size, num_threads, device_id, test_data_root, img_size)
    pipe_test.build()
    """
    LastBatchPolicy.PARTIAL like drop_last=False, keep the sample of the last batch(the number of this batch<batch size)
    It is used for training or testing, and must be used for testing, otherwise the test results will be inaccurate
    """
    test_loader = DALIClassificationIterator(pipe_test, size=pipe_test.epoch_size('Reader'),
                                                 last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)
    return train_loader, test_loader



def load_dataset(batch_size=64, dataset='mnist'):
    """
    transforms.ToTensor() only normalize data to [0, 1]
    transforms.Normalize() 
    If mean and std standard deviation are calculated from the dataset itself, then after processing, the data will be standardized, 
    that is, the mean value is 0 and the standard deviation is 1, rather than being normalized to [-1,1];
    If mean and std standard deviation are both 0.5, then after normalizing, the data distribution is [-1, 1], 
    because the minimum value=(0-mean)/std=(0-0.5)/0.5=-1. 
    Similarly, the maximum value is equal to 1. Finally, the data will be normalized to [-1, 1]
    """
    if dataset == 'mnist':
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        
        data_train = torchvision.datasets.MNIST(root='data/MNIST', 
                                                train=True, download=True, 
                                                transform=trans)
        data_test = torchvision.datasets.MNIST(root='data/MNIST', 
                                                train=False, download=True, 
                                                transform=trans)
        train_iter = torch.utils.data.DataLoader(data_train,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=6) # number of threads to load data
        test_iter = torch.utils.data.DataLoader(data_test,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=6)
    elif dataset == 'cifar':
        std = [0.5, 0.5, 0.5]
        mean = [0.5, 0.5, 0.5]
        trans_train = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean = mean, std = std), 
            transforms.RandomHorizontalFlip(),                            # Random horizontal mirror
            transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),  # Random occlusion
            transforms.RandomCrop(32, padding=4),                         # Random center clipping
        ])
        trans_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = mean, std = std)])
        
        data_train = torchvision.datasets.CIFAR10(root='data/CIFAR10', 
                                                train=True, download=True, 
                                                transform=trans_train)
        data_test = torchvision.datasets.CIFAR10(root='data/CIFAR10', 
                                                train=False, download=True, 
                                                transform=trans_test)
        train_iter = torch.utils.data.DataLoader(data_train,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=8)
        test_iter = torch.utils.data.DataLoader(data_test,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=8)
    elif dataset == 'imagenet':
        # std = [0.5, 0.5, 0.5]
        # mean = [0.5, 0.5, 0.5]
        # trans_train = transforms.Compose([
        #     transforms.Resize([256, 256]), 
        #     transforms.ToTensor(), 
        #     transforms.Normalize(mean = mean, std = std), 
        #     transforms.RandomHorizontalFlip(),
        #     # transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),
        #     transforms.RandomCrop(224,padding=4),
        #     # PCA
        # ])
        # trans_test = transforms.Compose([transforms.Resize([224, 224]),
        #     transforms.ToTensor(), transforms.Normalize(mean = mean, std = std)])
        
        # data_train = torchvision.datasets.ImageFolder(root='data/ImageNet/train',
        #     transform=trans_train)
        # data_test = torchvision.datasets.ImageFolder(root='data/ImageNet/valid',
        #     transform=trans_test)

        train_iter, test_iter = get_dali_iter(batch_size, num_threads=16, device_id=0, 
            train_data_root='data/ImageNet/train', test_data_root='data/ImageNet/valid', 
            img_size=224, use_gpu=True)

    return train_iter, test_iter

