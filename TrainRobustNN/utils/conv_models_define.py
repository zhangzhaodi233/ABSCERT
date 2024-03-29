import torch.nn as nn
from torch.nn import functional as F

class LeNet5(nn.Module):
    def __init__(self, in_ch, in_dim):
        super(LeNet5, self).__init__()
        self.conv = nn.Sequential(  # [n, 3, 32, 32]
            nn.Conv2d(in_ch, 6, 5, padding=2), # in_channels, out_channels, kernel_size
            nn.ReLU(),                     
            nn.MaxPool2d(2, 2),            # 32/2=16
            nn.Conv2d(6, 16, 5),           # (16-5+1)=12
            nn.ReLU(),
            nn.MaxPool2d(2, 2)             # 12/2=6
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * int((in_dim/2-4)/2) * int((in_dim/2-4)/2), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )
    
    def forward(self, img, labels=None):
        output = self.conv(img)
        logits = self.fc(output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

class IBP_2_layer(nn.Module):
    def __init__(self, ):
        super(IBP_2_layer, self).__init__()
        self.conv = nn.Sequential(  # [n, 1, 28, 28]
            nn.Conv2d(1, 64, 3, stride=1, padding=1), # in_channels, out_channels, kernel_size
            nn.ReLU(),                    
            nn.Conv2d(64, 64, 3, stride=2, padding=1),        
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, img, labels=None):
        output = self.conv(img)
        logits = self.fc(output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

class mlp_4_layer(nn.Module):
    def __init__(self, ):
        super(mlp_4_layer, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, img, labels=None):
        x = img.view(-1, 784)
        logits = self.fc(x)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

class mlp_4_layer_robust(nn.Module):
    def __init__(self, ):
        super(mlp_4_layer_robust, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(784*2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, img, labels=None, abstract=False, interval_num=0):
        #x = img.view(-1, 784)
        logits = self.fc(img)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits
        
class DM_Small_Relu(nn.Module):  # relu
    def __init__(self, in_ch, in_dim, linear_size=100):
        super(DM_Small_Relu, self).__init__()
        self.conv = nn.Sequential(  # [n, in_ch, in_dim, in_dim]
            nn.Conv2d(in_ch, 16, 4, stride=2, padding=2), # in_channels, out_channels, kernel_size
            nn.ReLU(),  # [4*width, ((in_dim-kernel_size+2*padding) // stride) + 1, ((in_dim-kernel_size+2*padding) // stride) + 1]                  
            nn.Conv2d(16, 32, 4, stride=1, padding=2),        
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8*4*(in_dim // 2+2)*(in_dim // 2+2),linear_size),
            nn.ReLU(),
            nn.Linear(linear_size, 10)
        )
    
    def forward(self, img, labels=None):
        output = self.conv(img)
        logits = self.fc(output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

class DM_Small_Sigmoid(nn.Module):  # sigmoid
    def __init__(self, in_ch, in_dim, linear_size=100):
        super(DM_Small_Sigmoid, self).__init__()
        self.conv = nn.Sequential(  # [n, in_ch, in_dim, in_dim]
            nn.Conv2d(in_ch, 16, 4, stride=2, padding=2), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),  # [4*width, ((in_dim-kernel_size+2*padding) // stride) + 1, ((in_dim-kernel_size+2*padding) // stride) + 1]                  
            nn.Conv2d(16, 32, 4, stride=1, padding=2),        
            nn.Sigmoid(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8*4*(in_dim // 2+2)*(in_dim // 2+2),linear_size),
            nn.Sigmoid(),
            nn.Linear(linear_size, 10)
        )
    
    def forward(self, img, labels=None):
        output = self.conv(img)
        logits = self.fc(output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

class DM_Small_Tanh(nn.Module):  # tanh
    def __init__(self, in_ch, in_dim, linear_size=100):
        super(DM_Small_Tanh, self).__init__()
        self.conv = nn.Sequential(  # [n, in_ch, in_dim, in_dim]
            nn.Conv2d(in_ch, 16, 4, stride=2, padding=2), # in_channels, out_channels, kernel_size
            nn.Tanh(),  # [4*width, ((in_dim-kernel_size+2*padding) // stride) + 1, ((in_dim-kernel_size+2*padding) // stride) + 1]                  
            nn.Conv2d(16, 32, 4, stride=1, padding=2),        
            nn.Tanh(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8*4*(in_dim // 2+2)*(in_dim // 2+2),linear_size),
            nn.Tanh(),
            nn.Linear(linear_size, 10)
        )
    
    def forward(self, img, labels=None):
        output = self.conv(img)
        logits = self.fc(output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

class DM_Medium_Relu(nn.Module):  # relu
    def __init__(self, in_ch, in_dim, linear_size=512):
        super(DM_Medium_Relu, self).__init__()
        self.conv = nn.Sequential(  # [n, in_ch, in_dim, in_dim]
            nn.Conv2d(in_ch, 32, 3, stride=1, padding=1), # in_channels, out_channels, kernel_size
            nn.ReLU(),                    
            nn.Conv2d(32, 32, 4, stride=2, padding=2),        
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=2),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8*8*(in_dim // 4+1)*(in_dim // 4+1), linear_size),
            nn.ReLU(),
            nn.Linear(linear_size, linear_size),
            nn.ReLU(),
            nn.Linear(linear_size, 10)
        )
    
    def forward(self, img, labels=None):
        output = self.conv(img)
        logits = self.fc(output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

class DM_Medium_Sigmoid(nn.Module):  # sigmoid
    def __init__(self, in_ch, in_dim, linear_size=512):
        super(DM_Medium_Sigmoid, self).__init__()
        self.conv = nn.Sequential(  # [n, in_ch, in_dim, in_dim]
            nn.Conv2d(in_ch, 32, 3, stride=1, padding=1), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),                    
            nn.Conv2d(32, 32, 4, stride=2, padding=2),        
            nn.Sigmoid(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(64, 64, 4, stride=2, padding=2),
            nn.Sigmoid(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8*8*(in_dim // 4+1)*(in_dim // 4+1), linear_size),
            nn.Sigmoid(),
            nn.Linear(linear_size, linear_size),
            nn.Sigmoid(),
            nn.Linear(linear_size, 10)
        )
    
    def forward(self, img, labels=None):
        output = self.conv(img)
        logits = self.fc(output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

class DM_Medium_Tanh(nn.Module):  # tanh
    def __init__(self, in_ch, in_dim, linear_size=512):
        super(DM_Medium_Tanh, self).__init__()
        self.conv = nn.Sequential(  # [n, in_ch, in_dim, in_dim]
            nn.Conv2d(in_ch, 32, 3, stride=1, padding=1), # in_channels, out_channels, kernel_size
            nn.Tanh(),                    
            nn.Conv2d(32, 32, 4, stride=2, padding=2),        
            nn.Tanh(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(64, 64, 4, stride=2, padding=2),
            nn.Tanh(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8*8*(in_dim // 4+1)*(in_dim // 4+1), linear_size),
            nn.Tanh(),
            nn.Linear(linear_size, linear_size),
            nn.Tanh(),
            nn.Linear(linear_size, 10)
        )
    
    def forward(self, img, labels=None):
        output = self.conv(img)
        logits = self.fc(output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

class DM_Large_Relu(nn.Module):  # relu
    def __init__(self, in_ch, in_dim, linear_size=512):
        super(DM_Large_Relu, self).__init__()
        self.conv = nn.Sequential(  # [n, in_ch, in_dim, in_dim]
            nn.Conv2d(in_ch, 64, 3, stride=1, padding=1), # in_channels, out_channels, kernel_size
            nn.ReLU(),                    
            nn.Conv2d(64, 64, 3, stride=1, padding=1),        
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear((in_dim//2) * (in_dim//2) * 128, linear_size),
            nn.ReLU(),
            nn.Linear(linear_size, 10)
        )

    def forward(self, img, labels=None):
        output = self.conv(img)
        logits = self.fc(output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits
        

class DM_Large_Sigmoid(nn.Module):  # sigmoid
    def __init__(self, in_ch, in_dim, linear_size=512):
        super(DM_Large_Sigmoid, self).__init__()
        self.conv = nn.Sequential(  # [n, in_ch, in_dim, in_dim]
            nn.Conv2d(in_ch, 64, 3, stride=1, padding=1), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),                    
            nn.Conv2d(64, 64, 3, stride=1, padding=1),        
            nn.Sigmoid(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.Sigmoid(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.Sigmoid(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear((in_dim//2) * (in_dim//2) * 128, linear_size),
            nn.Sigmoid(),
            nn.Linear(linear_size, 10)
        )

    def forward(self, img, labels=None):
        output = self.conv(img)
        logits = self.fc(output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


class DM_Large_Tanh(nn.Module):  # tanh
    def __init__(self, in_ch, in_dim, linear_size=512):
        super(DM_Large_Tanh, self).__init__()
        self.conv = nn.Sequential(  # [n, in_ch, in_dim, in_dim]
            nn.Conv2d(in_ch, 64, 3, stride=1, padding=1), # in_channels, out_channels, kernel_size
            nn.Tanh(),                    
            nn.Conv2d(64, 64, 3, stride=1, padding=1),        
            nn.Tanh(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.Tanh(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear((in_dim//2) * (in_dim//2) * 128, linear_size),
            nn.Tanh(),
            nn.Linear(linear_size, 10)
        )

    def forward(self, img, labels=None):
        output = self.conv(img)
        logits = self.fc(output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits




class AlexNet(nn.Module):
    def __init__(self, in_ch):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 96, kernel_size=(11, 11), stride=(4, 4), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6400, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1000)
        )
    def forward(self, x, y=None):  # x: (batch_size, 3, 256, 256)
        output = self.conv(x)
        y_hat = self.fc(output)
        if y is not None:
            loss = nn.CrossEntropyLoss(reduction='mean')
            l = loss(y_hat, y)
            return l, y_hat
        else:
            return y_hat

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=(3,3), padding=(1,1)))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

conv_arch = ((1,64),(1,128),(2,256),(2,512),(2,512))

class VGG11(nn.Module):
    def __init__(self, in_ch):
        super(VGG11, self).__init__()
        conv_blks = []
        in_channels = in_ch
        for (num_convs, out_channels) in conv_arch:
            conv_blks.append(vgg_block(
                num_convs, in_channels, out_channels
            ))
            in_channels = out_channels

        self.conv = nn.Sequential(
            *conv_blks,
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_channels * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1000)
        )

    def forward(self, x, y=None):  # x: (batch_size, 3, 224, 224)
        output = self.conv(x)
        y_hat = self.fc(output)
        if y is not None:
            loss = nn.CrossEntropyLoss(reduction='mean')
            l = loss(y_hat, y)
            return l, y_hat
        else:
            return y_hat

class Residual(nn.Module):
    """residual block"""
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=(3,3), padding=(1,1), stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=(3,3), padding=(1,1))
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=(1,1), stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y+=x
        return F.relu(y)

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    """
    If first_block is True, then the height and width will not be halved and the channel will not be doubled. 
    Else, half height and width, double channel
    """
    blk = []
    for i in range(num_residuals):
        if i==0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True,
                                strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

class ResNet18(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        b1 = nn.Sequential(nn.Conv2d(in_ch, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3)),
                           nn.BatchNorm2d(64),
                           nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))
        self.conv = nn.Sequential(
            b1,b2,b3,b4,b5,
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1000)
        )
    def forward(self, x, y=None):  # x: (batch_size, 3, 224, 224)
        output = self.conv(x)
        y_hat = self.fc(output)
        if y is not None:
            loss = nn.CrossEntropyLoss(reduction='mean')
            l = loss(y_hat, y)
            return l, y_hat
        else:
            return y_hat


class ResNet34(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        b1 = nn.Sequential(nn.Conv2d(in_ch, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3)),
                           nn.BatchNorm2d(64),
                           nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        b2 = nn.Sequential(*resnet_block(64, 64, 3, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 4))
        b4 = nn.Sequential(*resnet_block(128, 256, 6))
        b5 = nn.Sequential(*resnet_block(256, 512, 3))
        self.conv = nn.Sequential(
            b1,b2,b3,b4,b5,
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1000)
        )
    
    def forward(self, x, y=None):  # x: (batch_size, 3, 224, 224)
        output = self.conv(x)
        y_hat = self.fc(output)
        if y is not None:
            loss = nn.CrossEntropyLoss(reduction='mean')
            l = loss(y_hat, y)
            return l, y_hat
        else:
            return y_hat




class Residual_50(nn.Module):
    """residual block"""
    def __init__(self, input_channels, middle_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, middle_channels,
                               kernel_size=(1,1), padding=0)
        self.conv2 = nn.Conv2d(middle_channels, middle_channels,
                               kernel_size=(3,3), padding=(1,1), stride=strides)
        self.conv3 = nn.Conv2d(middle_channels, num_channels,
                                kernel_size=(1,1), padding=0)

        if use_1x1conv:
            self.conv4 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=(1,1), stride=strides)
        else:
            self.conv4 = None
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.bn2 = nn.BatchNorm2d(middle_channels)
        self.bn3 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU()

    def forward(self,x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        if self.conv4:
            x = self.conv4(x)
        y+=x
        return F.relu(y)


def resnet_block_50(input_channels, middle_channels, num_channels, num_residuals,
                 first_block=False):
    """
    If first_block is True, then the height and width will not be halved and the channel will not be doubled. 
    Else, half height and width, double channel
    """
    blk = []
    for i in range(num_residuals):
        if i==0: 
            if first_block:
                blk.append(Residual_50(input_channels, middle_channels, num_channels, use_1x1conv=True,
                                strides=1))
            else:
                blk.append(Residual_50(input_channels, middle_channels, num_channels, use_1x1conv=True,
                                strides=2))
        else:
            blk.append(Residual_50(num_channels, middle_channels, num_channels))
    return blk


class ResNet50(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        b1 = nn.Sequential(nn.Conv2d(in_ch, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3)),
                           nn.BatchNorm2d(64),
                           nn.ReLU(),
                           nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        b2 = nn.Sequential(*resnet_block_50(64, 64, 256, 3, first_block=True))
        b3 = nn.Sequential(*resnet_block_50(256, 128, 512, 4))
        b4 = nn.Sequential(*resnet_block_50(512, 256, 1024, 6))
        b5 = nn.Sequential(*resnet_block_50(1024, 512, 2048, 3))
        self.conv = nn.Sequential(
            b1,b2,b3,b4,b5,
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1000)
        )
    
    def forward(self, x, y=None):  # x: (batch_size, 3, 224, 224)
        output = self.conv(x)
        y_hat = self.fc(output)
        if y is not None:
            loss = nn.CrossEntropyLoss(reduction='mean')
            l = loss(y_hat, y)
            return l, y_hat
        else:
            return y_hat

class Inception_block(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super().__init__(**kwargs)
        # path 1, 1*1 convolution layer
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # path 2, 1*1 + 3*3 convolution layer
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # path 3, 1*1 + 5*5 convolution layer
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # path 4, 3*3 maxpool layer + 1*1 convolution layer
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
    
    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)


class Inception_v1(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        b1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        b2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        b3 = nn.Sequential(
            Inception_block(192, 64, (96, 128), (16, 32), 32),
            Inception_block(256, 128, (128, 192), (32, 96), 64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        b4 = nn.Sequential(
            Inception_block(480, 192, (96, 208), (16, 48), 64),
            Inception_block(512, 160, (112, 224), (24, 64), 64),
            Inception_block(512, 128, (128, 256), (24, 64), 64),
            Inception_block(512, 112, (144, 288), (32, 64), 64),
            Inception_block(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        b5 = nn.Sequential(
            Inception_block(832, 256, (160, 320), (32, 128), 128),
            Inception_block(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.conv = nn.Sequential(
            b1,b2,b3,b4,b5
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(1024, 1000)
        )
    

    def forward(self, x, y=None):  # x(batch_size, 3, 224, 224)
        output = self.conv(x)
        y_hat = self.fc(output)
        if y is not None:
            loss = nn.CrossEntropyLoss(reduction='mean')
            l = loss(y_hat, y)
            return l, y_hat
        else:
            return y_hat

