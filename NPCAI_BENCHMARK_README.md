# NPCAI Exllama Benchmark

This directory contains a benchmark script for Llama models via Exllama. Please refer to this document for how to install a Llama model and run the benchmark script against it.

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
