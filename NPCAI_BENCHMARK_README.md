# NPCAI Exllama Benchmark

This directory contains a benchmark script for Llama models via Exllama. Please refer to this document for how to install a Llama model and run the benchmark script against it. There are two different methods for setting up the instance:
- Via a setup script
- Manual setup

# Setup Script Method
This section outlines how to set up the instance using a shell script.

## Step 1: Create the setup script 

Run the following command:

```
sudo vim setup.sh
```

Paste the following code:

**NOTE 1:** This script downloads the `Llama-2-13B-chat-GPTQ` model files. If you would like to change that, update the URL in Step 5 of this script

**NOTE 2:** Check [https://developer.nvidia.com](https://developer.nvidia.com/cuda-downloads) for the latest version of the cuda keyring file and update Step 2 of this script accordingly
```
#!/bin/bash

# Step 1: Update apt repository and upgrade packages
sudo apt-get update
sudo apt-get -y upgrade

# Step 2: Install CUDA keyring and CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda

# Step 3: Install python3-pip & git-lfs
sudo apt install -y python3-pip git-lfs

# Step 4: Clone & configure exllama repo
git clone --progress --verbose https://github.com/NPCAI-Studio/exllama.git
cd exllama
git-lfs install
pip install -r requirements.txt
sudo apt install -y ninja-build

# Step 5: Install a GPTQ model from HuggingFace
mkdir models
cd ./models
git clone --progress --verbose https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ

# Step 6: Run the benchmark
cd ..
echo ""
echo "Setup completed succussfully. Run the benchmark with the following command:"
echo ""
echo "python3 npcai_benchmark.py --model_dir ./models/Llama-2-13B-chat-GPTQ/"
echo ""
echo ""
```

Save and close the file.

## Step 2: Make the file executable

Run the following command:

```
sudo chmod u+x setup.sh
```

## Step 3: Run the script

Run the following command:

```
sudo ./setup.sh
```

## Step 4: Run the benchmark

Run the following commands:

```
cd exllama
python3 npcai_benchmark.py --model_dir <path to model dir>
```

# Manual Setup

## Step 1: Update apt repository and upgrade packages

Run the following commands:

```
sudo apt update
sudo apt -y upgrade
```

## Step 2: Install CUDA
Navigate to [https://developer.nvidia.com](https://developer.nvidia.com/cuda-downloads) and configure the download option.

Follow the installation instructions depending on the install method selected (recommended: **deb (network)**)


## Step 3: Install python3-pip & git-lfs

Run the following command:

```
sudo apt install python3-pip git-lfs
```

## Step 4: Clone & configure this repo

Run the following commands:

```
git clone https://github.com/NPCAI-Studio/exllama.git
cd exllama
git-lfs install
pip install -r requirements.txt
sudo apt install ninja-build
```

## Step 5: Install a GPTQ model from HuggingFace

Navigate to the `exllama` repo root directory. Find the binary you want to install on HuggingFace. Copy the link and run the following commands (Llama-2-13b-chat-GPTQ is the default):

```
mkdir models
cd models
git clone https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ
```

## Step 6: Run the benchmark

Run the following command from the root directory of this repository (`path-to-model-dir` should be the cloned repository from the previous step):

```
python3 npcai_benchmark.py --model_dir <path-to-model-dir>
```
