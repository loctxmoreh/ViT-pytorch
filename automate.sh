#!/usr/bin/env bash

# Automated testing script for HAC machine

mkdir -p checkpoint
if [[ ! -f checkpoint/ViT-B_16.npz ]]; then
    wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz -P ./checkpoint/
fi

#env_file="hacenv.yml"
env_file="hacenv2.yml"
env_name=$(grep ^name: $env_file | awk '{ print $NF }')

conda env create -f $env_file
conda run -n $env_name update-moreh --force --nightly

# install apex
[[ -d apex ]] && rm -rf apex
git clone https://github.com/NVIDIA/apex
conda run -n $env_name pip install -v --disable-pip-version-check --no-cache-dir apex/

echo "Train with single precision:"
conda run -n $env_name ./train-single-precision.sh 2>&1 | tee single-precision.log

echo "Train with mixed precision:"
conda run -n $env_name ./train-mixed-precision.sh 2>&1 | tee mixed-precision.log

conda env remove -n $env_name
