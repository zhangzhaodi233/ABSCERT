# TrainRobustNN

TrainRobustNN is the official implementation for paper ["Abstraction-based Training Verified Robust Neural Networks"](). In this project, we propose an elegant method to train verifiable robust neural networks through abstracting input images. The experiment shows that our method is much better than other verifiable robust methods in both accuracy and efficiency.

## Start with the code

All the scripts and code were tested on a workstation running Ubuntu 18.04.

1. Download the code  
	```
	git clone https://github.com/zhangzhaodi233/TrainRobustNN
	```
2. Install the following necessary dependencies:  
	> torch, torchvision, transformers, nvidia-dali, d2l  

	If prefer, you may create a new conda environment by running:

	```
	./install.sh
	```

GPU is indispensiable for training models

## Train and Verify

We implement a method of training while verifying. 

The project is tested on Python 3.9.12 and Pytorch 1.12.1.

We provide example training hyper-parameters with json format under the **config** directory. You can also apply different hyper-parameters to train the verifiable robust neural network.

To train and verify model DM_Small on MNIST with predefined hyper-parameters, run:

	python train_robust.py --config mnist_dm_small.json

To train and verify model DM_Small on CIFAR with predefined hyper-parameters, run:

	python train_robust.py --config cifar_dm_small.json

To train and verify model on ImageNet, you need to download the Imagenet dataset first. After that, put the training set to **"./data/ImageNet/train"** and put the verification set to **"./data/ImageNet/valid"**. Note that you need to reform the verification set just like the training set.   
To train and verify model AlexNet on ImageNet with predefined hyper-parameters, run:

	python train_robust.py --config imagenet_alexnet.json


## Main Result

We show part of our state-of-art verification results below：

| Dataset      | Model     | l∞ Perturbation(ε) | Abstract Granularity(η) | Acc    | Time(s)   |
| :----------: | :-------: | :----------------: | :------------------: | :----: | :----: |
| MNIST        | DM_Small  | 0.1                | 0.200                  |  99.10%   |  2.18  |
| MNIST        | DM_Medium | 0.1                | 0.222                  |  99.36%   |  2.32   |
| MNIST        | DM_Large  | 0.1                | 0.285                  |  99.43%   |  4.28   |
| CIFAR10      | DM_Small  | 2/255              | 0.064                  |  74.48%   |  3.31   |
| CIFAR10      | DM_Medium | 2/255              | 0.035                  |  83.60%   |  3.58   |
| CIFAR10      | DM_Large  | 2/255              | 0.045                  |  86.19%   |  5.29   |
| ImageNet     | AlexNet   | 2/255              | 0.1                  |  55.04%   |  481.35   |
| ImageNet     | VGG11     | 2/255              | 0.1                  |  99%   |  10s   |
| ImageNet     | ResNet18  | 2/255              | 0.1                  |  99%   |  10s   |
| ImageNet     | ResNet34  | 2/255              | 0.1                  |  99%   |  10s   |
| ImageNet     | ResNet50  | 2/255              | 0.1                  |  99%   |  10s   |

Note that we set η >= 2 · ε for each experiemnt. 