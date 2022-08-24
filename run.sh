#!/bin/bash

# run command: nohup ./run.sh >/dev/null 2>train_error.log &

python train_robust.py --config mnist_dm_small.json
python train_robust.py --config mnist_dm_medium.json
python train_robust.py --config mnist_dm_large.json

python train_robust.py --config cifar_dm_small.json
python train_robust.py --config cifar_dm_medium.json
python train_robust.py --config cifar_dm_large.json

python train_robust.py --config imagenet_alexnet.json
python train_robust.py --config imagenet_vgg11.json
python train_robust.py --config imagenet_resnet18.json
python train_robust.py --config imagenet_resnet34.json
