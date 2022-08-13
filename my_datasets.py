from re import L
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
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
            self.decode = ops.decoders.Image(device='cpu')  # pipeline中定义了一个解码图像的模块，输出的格式为RGB顺序
        else:
            super(TrainPipeline, self).__init__(batch_size, num_threads, device_id, prefetch_queue_depth=4)
            mode = 'gpu'
            self.decode = ops.decoders.Image(device='mixed')

        self.img_size = img_size
        self.use_gpu = use_gpu
        # readers.File类似torchvision.datasets.ImageFolder
        self.input = ops.readers.File(file_root=data_root, random_shuffle=True)
        # Resize
        self.resize = ops.Resize(device=mode, resize_x=256, resize_y=256)
        # Randomcrop，类似于torchvision.transforms.RandomCrop
        self.randomcrop = ops.RandomResizedCrop(device=mode, size=img_size, random_area=[0.3, 1.0])
        # CropMirrorNormalize可以实现normalize和随机水平翻转，类似于torchvision.transforms.Normalize & RandomHorizontalFlip
        self.normalize = ops.CropMirrorNormalize(device=mode, mean=[0.5*255, 0.5*255, 0.5*255],
                                                 std=[0.5*255, 0.5*255, 0.5*255])  # 归一化到[-1, 1]

        # 实例化改变图片色彩的类，类似于torchvision.transforms.ColorJitter
        self.colortwist = ops.ColorTwist(device=mode)
        # 实例化旋转图像的类，类似于torchvision.transforms.RandomRotation
        self.rotate = ops.Rotate(device=mode, fill_value=0)
        # gridmask，类似于cutout这种随机遮挡块操作
        self.gridmask = ops.GridMask(device=mode)
        """
        自定义cutout预处理, 缺点是全部需要用cpu进行
        需要设exec_async=False和exec_pipelined=False
        """
        # self.mask = ops.PythonFunction(device=mode, function=CUTOUT(n_holes, length), num_outputs=1)

    # 作用是在调用该pipeline时，应该如何对数据进行实际的操作，可以理解为pytorch的module的forward函数
    def define_graph(self):
        jpegs, labels = self.input(name='Reader')  # 加载数据
        images = self.decode(jpegs)
        images = self.resize(images)  # 数据增强
        # images = self.rotate(images, angle=rng4)
        images = self.randomcrop(images)
        # images = self.colortwist(images, brightness=rng1, contrast=rng1, saturation=rng2, hue=rng3)
        # if self.custom_cutout:
        #     images = self.mask(images)
        # else:
        #     images = self.gridmask(images, ratio=rng6, shift_x=rng7, shift_y=rng7, tile=int(self.img_size*0.25))
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
    # DALIClassificationIterator: Returns 2 outputs (data and label) in the form of PyTorch’s Tensor, 即DataLoader
    train_loader = DALIClassificationIterator(pipe_train, size=pipe_train.epoch_size('Reader'),
                                                last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)

    pipe_test = TestPipeline(batch_size, num_threads, device_id, test_data_root, img_size)
    pipe_test.build()
    """
    LastBatchPolicy.PARTIAL的作用等同于drop_last=False,保留最后一个batch的样本(该batch的样本数<batch size)
    用于训练或测试，测试的话一定要用这个，不然得到的测试结果会有不准确
    """
    test_loader = DALIClassificationIterator(pipe_test, size=pipe_test.epoch_size('Reader'),
                                                 last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)
    return train_loader, test_loader



def load_dataset(batch_size=64, dataset='mnist'):

    # transforms.ToTensor() 只是将数据归一化到 [0, 1]
    # transforms.Normalize() 
    # 如果 mean 均值和 std 标准差是通过数据集本身求出来的，那么经过处理后，数据被标准化，即均值为0，标准差为1，而并非归一化到 [-1, 1];
    # 如果 mean 均值和 std 标准差 都为 0.5，那么 Normalize之后，数据分布是 [-1，1], 因为最小值 =（0-mean）/std=(0-0.5)/0.5=-1。同理最大值的等于1。最终则是将数据归一化到 [-1, 1]
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
                                             num_workers= 6) # num_workers 工作者数量，默认是0。使用多少个子进程来导入数据。设置为0，就是使用主进程来导入数据
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
            transforms.RandomHorizontalFlip(),                            # 随机水平镜像
            transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),  # 随机遮挡
            transforms.RandomCrop(32, padding=4),                         # 随机中心裁剪
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
                                             num_workers= 6) # num_workers 工作者数量，默认是0。使用多少个子进程来导入数据。设置为0，就是使用主进程来导入数据
        test_iter = torch.utils.data.DataLoader(data_test,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=6)
    elif dataset == 'imagenet':
        # std = [0.5, 0.5, 0.5]
        # mean = [0.5, 0.5, 0.5]
        # trans_train = transforms.Compose([
        #     transforms.Resize([256, 256]), 
        #     transforms.ToTensor(), 
        #     transforms.Normalize(mean = mean, std = std), 
        #     transforms.RandomHorizontalFlip(),                            # 随机水平镜像
        #     # transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),  # 随机遮挡
        #     transforms.RandomCrop(224,padding=4),                         # 随机中心裁剪
        #     # PCA，暂时不实现
        # ])
        # trans_test = transforms.Compose([transforms.Resize([224, 224]),
        #     transforms.ToTensor(), transforms.Normalize(mean = mean, std = std)])
        
        # data_train = torchvision.datasets.ImageFolder(root='data/ImageNet/train',
        #     transform=trans_train)
        # data_test = torchvision.datasets.ImageFolder(root='data/ImageNet/valid',
        #     transform=trans_test)

        train_iter, test_iter = get_dali_iter(batch_size, num_threads=8, device_id=0, 
            train_data_root='data/ImageNet/train', test_data_root='data/ImageNet/valid', 
            img_size=224, use_gpu=True)

    return train_iter, test_iter

def abstract_data(x, interval_num):

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

    x_result = torch.cat((x_upper, x_lower), dim=1)
    return x_result

# 如何将扰动区间映射到抽象区间
def abstract_disturbed_data(x, interval_num, epsilon):

    # 比如 x = (-0.55, 0.55), eps = 0.1, 则 x 的扰动区间为 ([-0.65, -0.45][0.45, 0.65])
    # 那么每个像素的扰动区间分别映射为 [-1, -0.5], [-0.5, 0] 和 [0, 0.5], [0.5, 1]
    # 这时，将每个像素的抽象子区间进行合并，合并为 [-1, 0] 和 [0, 1]
    # 这样既解决了空间爆炸问题，验证的时候又包含了所有情况。只不过可能对分类精确度和鲁棒精确度有所影响，后期根据实验结果再进行调整。

    # 计算被扰动之后下界的抽象区间
    x_lower = x - epsilon
    x_lower = torch.clamp(x_lower, -1, 1)
    
    step = (1-(-1))/interval_num
    k = torch.div((x_lower - (-1)), step, rounding_mode='floor')
    x_lower_abstract_lower = -1 + k * step 
    x_lower_abstract_lower = torch.clamp(x_lower_abstract_lower, -1, 1)

    # 计算扰动之后上界的抽象区间
    x_upper = x + epsilon
    x_upper = torch.clamp(x_upper, -1, 1)
    
    k = torch.div((x_upper - (-1)), step, rounding_mode='floor')
    x_upper_abstract_lower = -1 + k * step
    x_upper_abstract_lower = torch.clamp(x_upper_abstract_lower, -1, 1)
    x_upper_abstract_upper = x_upper_abstract_lower + step 
    x_upper_abstract_upper = torch.clamp(x_upper_abstract_upper, -1, 1)

    # 取扰动区间下界的抽象区间的下界, 取扰动区间上界的抽象区间的上界
    x_result = torch.cat((x_lower_abstract_lower, x_upper_abstract_upper), dim=1)

    return x_result

if __name__ == '__main__':
    x = torch.Tensor([[0.6], [0.79]])
    x_result = abstract_data(x, 6)
    print(x_result)