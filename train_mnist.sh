#!/bin/bash

# run command: nohup ./train_mnist.sh >/dev/null 2>runs_v2_txt/train.log &

python train_robust.py --structure DM_Small --model_name small_0_005 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 0.05
python train_robust.py --structure DM_Small --model_name small_0_008 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 0.08
python train_robust.py --structure DM_Small --model_name small_0_016 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 0.16
python train_robust.py --structure DM_Small --model_name small_0_032 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 0.32
python train_robust.py --structure DM_Small --model_name small_0_05 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 0.5
python train_robust.py --structure DM_Small --model_name small_0_064 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 0.64
python train_robust.py --structure DM_Small --model_name small_0_1 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 1
python train_robust.py --structure DM_Small --model_name small_0_128 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 1.28
python train_robust.py --structure DM_Small --model_name small_0_2 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 2
python train_robust.py --structure DM_Small --model_name small_0_256 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 2.56
python train_robust.py --structure DM_Small --model_name small_0_4 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 4
python train_robust.py --structure DM_Small --model_name small_0_5 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 5
python train_robust.py --structure DM_Small --model_name small_0_512 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 5.12
python train_robust.py --structure DM_Small --model_name small_0_6 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 6
python train_robust.py --structure DM_Small --model_name small_0_8 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 8
python train_robust.py --structure DM_Small --model_name small_1_0 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 10
python train_robust.py --structure DM_Small --model_name small_1_024 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 10.24

python train_robust.py --structure DM_Medium --model_name medium_0_005 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 0.05
python train_robust.py --structure DM_Medium --model_name medium_0_008 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 0.08
python train_robust.py --structure DM_Medium --model_name medium_0_016 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 0.16
python train_robust.py --structure DM_Medium --model_name medium_0_032 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 0.32
python train_robust.py --structure DM_Medium --model_name medium_0_05 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 0.5
python train_robust.py --structure DM_Medium --model_name medium_0_064 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 0.64
python train_robust.py --structure DM_Medium --model_name medium_0_1 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 1
python train_robust.py --structure DM_Medium --model_name medium_0_128 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 1.28
python train_robust.py --structure DM_Medium --model_name medium_0_2 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 2
python train_robust.py --structure DM_Medium --model_name medium_0_256 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 2.56
python train_robust.py --structure DM_Medium --model_name medium_0_4 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 4
python train_robust.py --structure DM_Medium --model_name medium_0_5 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 5
python train_robust.py --structure DM_Medium --model_name medium_0_512 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 5.12
python train_robust.py --structure DM_Medium --model_name medium_0_6 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 6
python train_robust.py --structure DM_Medium --model_name medium_0_8 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 8
python train_robust.py --structure DM_Medium --model_name medium_1_0 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 10
python train_robust.py --structure DM_Medium --model_name medium_1_024 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 10.24

python train_robust.py --structure DM_Large --model_name large_0_005 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 0.05
python train_robust.py --structure DM_Large --model_name large_0_008 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 0.08
python train_robust.py --structure DM_Large --model_name large_0_016 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 0.16
python train_robust.py --structure DM_Large --model_name large_0_032 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 0.32
python train_robust.py --structure DM_Large --model_name large_0_05 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 0.5
python train_robust.py --structure DM_Large --model_name large_0_064 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 0.64
python train_robust.py --structure DM_Large --model_name large_0_1 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 1
python train_robust.py --structure DM_Large --model_name large_0_128 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 1.28
python train_robust.py --structure DM_Large --model_name large_0_2 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 2
python train_robust.py --structure DM_Large --model_name large_0_256 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 2.56
python train_robust.py --structure DM_Large --model_name large_0_4 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 4
python train_robust.py --structure DM_Large --model_name large_0_5 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 5
python train_robust.py --structure DM_Large --model_name large_0_512 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 5.12
python train_robust.py --structure DM_Large --model_name large_0_6 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 6
python train_robust.py --structure DM_Large --model_name large_0_8 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 8
python train_robust.py --structure DM_Large --model_name large_1_0 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 10
python train_robust.py --structure DM_Large --model_name large_1_024 --epochs 100 --learning_rate 0.001 --epsilon 0.1 --k 10.24

