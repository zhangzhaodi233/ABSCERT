# ABSCERT

ABSCERT is the official implementation for paper ["Boosting Verified Training for Robust Image Classifications via Abstraction"](). In this project, we propose an elegant method to train verifiable robust neural networks through abstracting input images. The experiment shows that our method is much better than other verifiable robust methods in both accuracy and efficiency.


## Project Structure
> Tool's code: 
> - TrainRobustNN/train/
> 	- train.py
> - TrainRobustNN/verify/
> 	- verify.py
> - TrainRobustNN/tuning/
> 	- tuning.py
> - TrainRobustNN/utils/
> 	- params.py
> 	- datasets.py
> 	- mapping_func.py
> 	- conv_models_define.py
> 	- fc_models_define.py
> - TrainRobustNN/main.py

> Configuration: 
> - TrainRobustNN/config/

> Reproduce figures in the paper:
> - TrainRobustNN/support/

> Train and Verify results produced when the code is run: 
> - TrainRobustNN/output/

> Scripts to install and run:
> - setup.py
> - install.sh
> - TrainRobustNN/run.sh


## Install ABSCERT

### Start from docker

We provide a docker image to run:
1. Download the docker image ***abscert.tar*** from [https://figshare.com/articles/software/abscert_tar/21571533](https://figshare.com/articles/software/abscert_tar/21571533). 
2. Load the docker image:
	
		docker load -i abscert.tar
3. Start a container with the image:
	
		docker run -it abscert:v1 /bin/bash
4. Navigate to the project directory

		cd /root/ABSCERT-main/TrainRobustNN



## Run ABSCERT and reproduce the results

1. Activate the virtual environment:
		
		conda activate abscert

2. Reproduce the result: 

	We provide example training hyper-parameters with json format under the **config** directory. You can also apply different hyper-parameters to train the verifiable robust neural network.

	To train and verify model DM_Small on MNIST with predefined hyper-parameters, run:

		python main.py --config mnist_dm_small.json

	To train and verify model DM_Small on CIFAR with predefined hyper-parameters, run:

		python main.py --config cifar_dm_small.json

	To train and verify model on ImageNet, you need to download the Imagenet dataset first(As the ImageNet training dataset is more than 130G, we don't provide it). After that, put the training set to **"./data/ImageNet/train"** and put the verification set to **"./data/ImageNet/valid"**. Note that you need to reform the verification set just like the training set.   
	To train and verify model AlexNet on ImageNet with predefined hyper-parameters, run:

		python main.py --config imagenet_alexnet.json

	Complete commands to train 35 neural networks on 3 datasets is provided in **"run.sh"**

3. The trained models will be saved in **./output/models/**. The log during training and verification will be saved in **./output/log/**.

## Experiment

We show part of our state-of-art verification results below：

| Dataset      | Model     | l∞ Perturbation(ε) | Abstract Granularity(η) | Acc    | Time(s)   |
| :----------: | :-------: | :----------------: | :------------------: | :----: | :----: |
| MNIST        | DM_Small  | 0.1                | 0.200                  |  99.10%   |  2.18  |
| MNIST        | DM_Medium | 0.1                | 0.222                  |  99.36%   |  2.32   |
| MNIST        | DM_Large  | 0.1                | 0.285                  |  99.43%   |  4.28   |
| CIFAR10      | DM_Small  | 2/255              | 0.064                  |  74.48%   |  3.31   |
| CIFAR10      | DM_Medium | 2/255              | 0.035                  |  83.60%   |  3.58   |
| CIFAR10      | DM_Large  | 2/255              | 0.045                  |  86.19%   |  5.29   |
| ImageNet     | AlexNet   | 2/255              | 0.016                  |  55.04%   |  508.2   |
| ImageNet     | VGG11     | 2/255              | 0.016                  |  63.71%   |  2530.3   |
| ImageNet     | Inception V1  | 2/255              | 0.016                  |  58.33%   |  2183.7   |
| ImageNet     | ResNet18  | 2/255              | 0.016                  |  67.85%   |  212.6   |

Note that we set η >= 2 · ε for each experiemnt. 