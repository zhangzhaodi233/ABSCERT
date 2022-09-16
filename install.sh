#!/bin/bash

conda create -n trainrobustnn python=3.9
conda activate trainrobustnn
conda install cudatoolkit=11.7
conda install cudnn=8.4.1
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install d2l
pip install transformers
pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/ nvidia-dali-cuda110==1.6.0
pip install -e .
