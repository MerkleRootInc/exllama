import os, glob, psutil, time, fire, subprocess

from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator

def main(
    model_dir: str,
    token_repetition_penalty_max: float = 1.2,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 100,
    typical: float = 0.5,
    max_seq_len: int = 4096,
    max_gen_len: int = 512,
):
    # Locate files we need within the directory containing the model, tokenizer, and generator
    tokenizer_path = os.path.join(model_dir, "tokenizer.model")
    model_config_path = os.path.join(model_dir, "config.json")
    st_pattern = os.path.join(model_dir, "*.safetensors")
    model_path = glob.glob(st_pattern)[0]

    # Create config, model, tokenizer and generator
    config = ExLlamaConfig(model_config_path)               # create config from config.json
    config.model_path = model_path                          # supply path to model weights file

    model = ExLlama(config)                                 # create ExLlama instance and load the weights
    tokenizer = ExLlamaTokenizer(tokenizer_path)            # create tokenizer from tokenizer model file

    cache = ExLlamaCache(model)                             # create cache for inference
    generator = ExLlamaGenerator(model, tokenizer, cache)   # create generator

    # Configure generator
    generator.disallow_tokens([tokenizer.eos_token_id])
    generator.settings.token_repetition_penalty_max = token_repetition_penalty_max
    generator.settings.temperature = temperature
    generator.settings.top_p = top_p
    generator.settings.top_k = top_k
    generator.settings.typical = typical

    input_sequences = [
        "What's 2+2?"
    ]

    print(f"Benchmarking inference for {len(input_sequences)} input sequences...\n")

    for sequence in input_sequences:
        tokens = tokenizer.encode(sequence, max_seq_len = max_seq_len)
        start_time = time.time()
        gpu_percent = "N/A"
        vram_usage = "N/A"

        gen_text = generator.generate_simple(sequence, max_new_tokens = max_gen_len)

        end_time = time.time()
        cpu_percent = psutil.cpu_percent(interval=None)
        if has_nvidia_gpu():
            gpu_percent = get_gpu_utilization()
            vram_usage = get_vram_usage()

        gen_tokens = tokenizer.encode(gen_text, max_seq_len = max_gen_len)
        gen_speed = len(gen_tokens) / (end_time - start_time)

        print()
        print(f"Input sequence length: {len(tokens)}")
        print(f"Total inference time (sec.): {round(end_time - start_time, 2)}")
        print(f"Tokens generated: {len(gen_tokens)}")
        print(f"Inference-adjusted rate (tokens/sec.): {round(gen_speed, 2)}")
        print(f"CPU utilization (%): {cpu_percent}")
        print(f"GPU utilization (%): {gpu_percent}")
        print(f"vRAM usage (MB): {vram_usage}")

        print()
        print("Generated Text:")
        print(gen_text)

        print()
        print("=" * 40)

def has_nvidia_gpu():
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if "NVIDIA-SMI" in result.stdout:
            return True
        return False
    except Exception as e:
        return False

def get_gpu_utilization():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        return float(result.stdout.strip())
    except Exception as e:
        print("Error:", e)
        return "N/A"

def get_vram_usage():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        return int(result.stdout.strip())
    except Exception as e:
        print("Error:", e)
        return "N/A"

if __name__ == "__main__":
    fire.Fire(main)