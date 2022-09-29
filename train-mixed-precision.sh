#!/bin/bash

python3 train.py \
        --name cifar10-100_500 \
        --dataset cifar10 \
        --model_type ViT-B_16 \
        --num_steps 10 \
        --train_batch_size 32 \
        --pretrained_dir checkpoint/ViT-B_16.npz \
        --fp16 --fp16_opt_level O2
