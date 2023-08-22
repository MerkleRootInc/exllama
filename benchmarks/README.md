# NPCAI Exllama Benchmark

This directory contains a benchmark script for Llama models via Exllama. Please refer to this document for how to install a Llama model and run the benchmark script against it.

## Step 1: Install Python

Run the following commands:

```
sudo apt-get update
sudo apt-get install python3.10
sudo apt install python3-pip
```

## Step 2: Install git

Run the follow command:

```
sudo apt-get install git
```

## Step 3: Clone and configure this repository

Run the following commands:

```
cd ..
git clone https://github.com/NPCAI-Studio/exllama.git
cd exllama
pip install -e .
```

## Step 4: Install git-lfs

Run the following command:

```
sudo apt-get install git-lfs
```

## Step 4: Install a GPTQ model from HuggingFace

Navigate to the `exllama` repo root directory. Find the binary you want to install on HuggingFace. Copy the link and run the following commands:

```
mkdir models
cd models
git lfs install
git clone https://huggingface.co/TheBloke/Llama-2-13B-chat-GPTQ
```

## Step 5: Run the benchmark

Run the following command from the root directory of this repository (`path-to-model-dir` should be the cloned repository from the previous step):

```
torchrun --nproc_per_node 1 benchmarks/npcai_benchmark.py --model_dir <path-to-model-dir>
```
