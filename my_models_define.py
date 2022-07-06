from re import L
import torch.nn as nn
from my_datasets import abstract_data

class LeNet5(nn.Module):
    def __init__(self, ):
        super(LeNet5, self).__init__()
        self.conv = nn.Sequential(  # [n, 1, 28, 28]
            nn.Conv2d(1, 6, 5, padding=2), # in_channels, out_channels, kernel_size
            nn.ReLU(),                     # [n, 6, 24, 24]
            nn.MaxPool2d(2, 2),            # kernel_size, stride  [n, 6, 14, 14]
            nn.Conv2d(6, 16, 5),           # [n, 16, 10, 10]
            nn.ReLU(),
            nn.MaxPool2d(2, 2)             # [n, 16, 5, 5]
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
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