#!/bin/bash

# run command: nohup ./run.sh >/dev/null 2>train_error.log &

python main.py --config mnist_dm_small.json
python main.py --config mnist_dm_medium.json
python main.py --config mnist_dm_large.json

python main.py --config cifar_dm_small.json
python main.py --config cifar_dm_medium.json
python main.py --config cifar_dm_large.json

python main.py --config imagenet_alexnet.json
python main.py --config imagenet_vgg11.json
python main.py --config imagenet_resnet18.json
python main.py --config imagenet_resnet34.json
