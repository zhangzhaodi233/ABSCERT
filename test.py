from turtle import end_fill
import torch
import torchvision
import torchvision.transforms as transforms
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from nvidia.dali.plugin.base_iterator import LastBatchPolicy
import time

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



if __name__ == "__main__":
    train_iter, test_iter = get_dali_iter(128, 8, 0, 'data/ImageNet/train', 'data/ImageNet/valid', 224, True)
    # print(len(train_iter))
    # start = time.time()
    # for batch_idx, data in enumerate(train_iter):
    #     # 图像和标签导入部分跟torchvision不一样，其余都一样
    #     img = data[0]["data"]
    #     label = data[0]["label"].squeeze(-1).long().to(img.device)
    # end = time.time() - start
    # print(end)
    for batch_idx, data in enumerate(train_iter):
        img = data[0]['data']
        print(img[0][0])
        exit(0)




