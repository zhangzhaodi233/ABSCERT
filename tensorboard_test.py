from importlib_metadata import metadata
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torchvision
import torch

def add_graph(writer):
    img = torch.rand([1, 3, 64, 64], dtype=torch.float32)
    model = torchvision.models.AlexNet(num_classes=10)
    writer.add_graph(model, input_to_model=img)

# 在一张图中绘制多个曲线
def add_scalars(writer):
    r = 5
    for i in range(100):
        writer.add_scalars(main_tag='scalars1/P1',
                            tag_scalar_dict={
                                'xsinx': i * np.sin(i / r),
                                'xcosx': i * np.cos(i / r),
                                'tanx': np.tan(i / r)
                            },
                            global_step=i)
        writer.add_scalars('scalars1/P2',
                            {
                                'xsinx': i * np.sin(i / (2 * r)),
                                'xcosx': i * np.cos(i / (2 * r)),
                                'tanx': np.tan(i / (2 * r))
                            }, i)
        writer.add_scalars(main_tag='scalars2/Q1',
                            tag_scalar_dict={
                                'xsinx': i * np.sin((2 * i) / r),
                                'xcosx': i * np.cos((2 * i) / r),
                                'tanx': np.tan((2 * i) / r)
                            },
                            global_step=i)
        writer.add_scalars('scalars2/Q2',
                            {
                                'xsinx': i * np.sin(i / (0.5 * r)),
                                'xcosx': i * np.cos(i / (0.5 * r)),
                                'tanx': np.tan(i / (0.5 * r))
                            }, i)

# 直方图
def add_histogram(writer):
    for i in range(10):
        x = np.random.random(1000)
        writer.add_histogram('distribution centers/p1', x + i, i)
        writer.add_histogram('distribution centers/p2', x + i * 2, i)

# 可视化相应的像素矩阵，例如本地图片，或者特征图等
def add_image(writer):
    
    img1 = np.random.randn(1, 100, 100)
    # 生成一个形状为 [C, H, W] 的三维矩阵并进行可视化
    writer.add_image('img/imag1', img1) 
    img2 = np.random.randn(100, 100, 3)
    # 生成一个形状为 [H, W, C] 的三维矩阵并进行可视化，需要使用 dataformats 参数指定矩阵的维度信息
    # 默认格式是 [C, H, W]
    writer.add_image('img/imag2', img2, dataformats='HWC')
    # # 从本地读取一张图片，再进行可视化
    # from PIL import Image
    # img = Image.open('xxx.png')
    # img_array = np.array(img)
    # writer.add_image('local/xxx', img_array, dataformats='HWC')

# 一次性可视化多张像素图
def add_images(writer):
    img1 = np.random.randn(8, 100, 100, 1)
    writer.add_images('imgs/images1', img1, dataformats='NHWC')
    img2 = np.zeros((16, 3, 100, 100))
    for i in range(16):
        img2[i, 0] = np.arange(0, 10000).reshape(100, 100) / 10000 / 16 * i
        img2[i, 1] = (1 - np.arange(0, 10000).reshape(100, 100) / 10000) / 16 * i
    writer.add_images('imgs/images2', img2)

# 在三维空间中对高维向量进行可视化，默认情况下是对高维向量以 PCA 方法进行降维处理
def add_embedding(writer):
    # 解决版本的兼容性问题
    import tensorflow as tf
    import tensorboard as tb
    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

    import keyword
    
    # 随机生成100个标签信息
    meta = []
    while len(meta) < 100:
        meta = meta + keyword.kwlist
    meta = meta[:100]
    for i, v in enumerate(meta):
        meta[i] = v + str(i)
    # 随机生成100个标签图片
    label_img = torch.rand(100, 3, 10, 32)
    for i in range(100):
        label_img[i] *= i / 100.0
    # 随机生成100个点
    data_points = torch.randn(100, 5)

    # mat: 用来指定可视化结果中每个点的坐标，形状为 (N,D)，不能为空， 例如对词向量可视化时mat就是词向量矩阵，图片分类时mat可以是分类层的输出结果
    # metadata: 用来指定每个点对应的标签信息，是一个包含N个元素的字符串列表，为空时则默认为['1','2',...,'N']
    # label_img: 用来指定每个点对应可视化信息，形状为（N,C,H,W），可以为空，例如图片分类是 label_img 就是每一张真实图片的可视化结果
    writer.add_embedding(mat=data_points,
                         metadata=meta,
                         label_img=label_img,
                         global_step=1)





if __name__ == '__main__':
    writer = SummaryWriter(log_dir='runs/result_2', flush_secs=120)
    for n_iter in range(100):
        writer.add_scalar(tag='Loss/train', scalar_value=np.random.random(), global_step=n_iter)
        writer.add_scalar('Loss/test', np.random.random(), n_iter)
    
    add_scalars(writer)
    add_histogram(writer)
    add_image(writer)
    add_images(writer)
    add_embedding(writer)

    writer.close()