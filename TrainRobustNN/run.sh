#!/bin/bash

# run command: nohup ./run.sh >/dev/null 2>train_error.log &

python main.py --config mnist_fc3.json
python main.py --config mnist_fc3_sigmoid.json
python main.py --config mnist_fc3_tanh.json
python main.py --config mnist_fc5.json
python main.py --config mnist_fc5_sigmoid.json
python main.py --config mnist_fc5_tanh.json
python main.py --config cifar_fc3.json
python main.py --config cifar_fc3_sigmoid.json
python main.py --config cifar_fc3_tanh.json
python main.py --config cifar_fc5.json
python main.py --config cifar_fc5_sigmoid.json
python main.py --config cifar_fc5_tanh.json


python main.py --config mnist_dm_small.json
python main.py --config mnist_dm_medium.json
python main.py --config mnist_dm_large.json
python main.py --config mnist_dm_small_tanh.json
python main.py --config mnist_dm_medium_tanh.json
python main.py --config mnist_dm_large_tanh.json
python main.py --config mnist_dm_small_sigmoid.json
python main.py --config mnist_dm_medium_sigmoid.json
python main.py --config mnist_dm_large_sigmoid.json

python main.py --config cifar_dm_small.json
python main.py --config cifar_dm_medium.json
python main.py --config cifar_dm_large.json
python main.py --config cifar_dm_small_tanh.json
python main.py --config cifar_dm_medium_tanh.json
python main.py --config cifar_dm_large_tanh.json
python main.py --config cifar_dm_small_sigmoid.json
python main.py --config cifar_dm_medium_sigmoid.json
python main.py --config cifar_dm_large_sigmoid.json

python main.py --config imagenet_alexnet.json
python main.py --config imagenet_vgg11.json
python main.py --config imagenet_inceptionv1.json
python main.py --config imagenet_resnet18.json
python main.py --config imagenet_resnet34.json
