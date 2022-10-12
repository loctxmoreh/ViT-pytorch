# [MOREH] Running on HAC VM - Moreh AI Framework
![](https://badgen.net/badge/Moreh-HAC/fail/red) ![](https://badgen.net/badge/Nvidia-A100/passed/green)

## Prepare

### Code
```bash
git clone https://github.com/loctxmoreh/ViT-pytorch
cd ViT-pytorch
```

#### Download pretrained model
Download the pretrained model by Google to serve as a checkpoint. This checkpoint
is used when training.
```bash
mkdir -p ./checkpoint
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz -P ./checkpoint/
```

### Environment
```bash
conda create -n vit python=3.8
conda activate vit
```

#### Installing `torch`

##### On A100 machine
With `torch==1.7.1` and CUDA 11.0:
```bash
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

With `torch==1.12.1` and CUDA 11.3:
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

##### On HAC machine
Install `torch=1.7.1`:
```bash
conda install -y torchvision torchaudio numpy protobuf==3.13.0 pytorch==1.7.1 cpuonly -c pytorch
```
Then, force update Moreh framework (version 22.9.2 at the time of writing)
```bash
update-moreh --force --version 22.9.2
```

#### Installing `apex`
Currently, `apex` fails to install in HAC VM. These instructions is for A100 VM.

Still inside the conda env, clone the `apex` repo:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
```

##### Install in full mode with CUDA
To be able to install in full mode, `apex` requires the *exact* same version
of CUDA on the machine and the CUDA version `torch` is compiled with. On A100 VM,
this is possible with `torch==1.7.1+cu110` and CUDA version 11.0.

```bash
# explicitly pointing CUDA_HOME to CUDA 11.0,
# since normally, CUDA_HOME point to CUDA 11.2 on A100 VM
CUDA_HOME=/usr/local/cuda-11.0 pip install -v --disable-pipp-version-check --no-cache-dir --global-option="--cpp_ext" --global-optionion="--cuda_ext" ./
```

##### Install in python-only mode
With `torch=1.12.1` and CUDA 11.3, `apex` can only be installed in python-only mode
```bash
pip install -v --disable-pip-version-check --no-cache-dir ./
```

#### The rest of requirements
```bash
pip install -r requirements.txt
```

## Run

### Single-precision training
With single-precision training, `apex` is not required. It runs fine in both
HAC VM and A100 VM.

Run the example script:
```bash
./train-single-precision.sh
```

### Mixed-precision training
Mixed-precision training requires `apex`. It failed on HAC VM. On A100 VM, both
full-mode and python-only mode of `apex` work fine.

Run the example script:
```bash
./train-mixed-precision.sh
```
