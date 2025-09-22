"""
Download and test Llama 3 8B model with authentication
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from huggingface_hub import login

# Your token
TOKEN = "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"
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

print("=" * 60)
print("LLAMA 3 8B DOWNLOAD")
print("=" * 60)

# Authenticate
print("\n1. Authenticating with HuggingFace...")
login(token=TOKEN)
print("   Authentication successful!")

# Test access
print("\n2. Testing Llama model access...")
try:
    config = AutoConfig.from_pretrained(MODEL_ID, token=TOKEN)
    print("   Access confirmed!")
    print(f"   Model type: {config.model_type}")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Layers: {config.num_hidden_layers}")
    print(f"   Max length: {config.max_position_embeddings}")
except Exception as e:
    print(f"   Error: {e}")
    print("\n   Your token may not have access to Llama yet.")
    print("   Please request access at: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct")
    exit(1)

# Download tokenizer
print("\n3. Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=TOKEN)
print("   Tokenizer downloaded!")

# Download model (this will be large ~16GB)
print("\n4. Downloading model (this will take several minutes)...")
print("   Model size: ~16GB")
print("   Downloading to cache: ~/.cache/huggingface/hub/")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=TOKEN,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

print("\n" + "=" * 60)
print("SUCCESS! Llama 3 8B is downloaded and ready!")
print("=" * 60)

# Quick test
print("\nQuick test - generating text...")
inputs = tokenizer("Hello, I am", return_tensors="pt")
outputs = model.generate(inputs.input_ids, max_new_tokens=20)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Response: {response}")

print("\nModel is ready for training!")