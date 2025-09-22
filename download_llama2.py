"""
Download and test Llama 2 7B model with authentication
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from huggingface_hub import login

# Your token
TOKEN = "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"

# Try different Llama 2 models
MODELS_TO_TRY = [
    "meta-llama/Llama-2-7b-hf",           # Base Llama 2 7B
    "meta-llama/Llama-2-7b-chat-hf",      # Chat version
    "NousResearch/Llama-2-7b-hf",         # Community version (no auth)
    "NousResearch/Llama-2-7b-chat-hf",    # Community chat version
]

print("=" * 60)
print("LLAMA 2 MODEL ACCESS TEST")
print("=" * 60)

# Authenticate
print("\n1. Authenticating with HuggingFace...")
login(token=TOKEN)
print("   Authentication successful!")

# Test each model
successful_model = None
for model_id in MODELS_TO_TRY:
    print(f"\n2. Testing access to: {model_id}")
    try:
        config = AutoConfig.from_pretrained(model_id, token=TOKEN)
        print(f"   ✓ ACCESS GRANTED!")
        print(f"   Model type: {config.model_type}")
        print(f"   Hidden size: {config.hidden_size}")
        print(f"   Layers: {config.num_hidden_layers}")
        print(f"   Max length: {config.max_position_embeddings}")
        successful_model = model_id
        break
    except Exception as e:
        if "403" in str(e) or "restricted" in str(e).lower():
            print(f"   ✗ No access (needs approval)")
        else:
            print(f"   ✗ Error: {str(e)[:100]}")

if not successful_model:
    print("\n" + "=" * 60)
    print("No Llama 2 models accessible with current token.")
    print("\nTo get access:")
    print("1. Go to: https://huggingface.co/meta-llama/Llama-2-7b-hf")
    print("2. Click 'Request access'")
    print("3. Fill out Meta's form")
    print("=" * 60)
    exit(1)

print("\n" + "=" * 60)
print(f"SUCCESS! Can access: {successful_model}")
print("=" * 60)

# Download the accessible model
MODEL_ID = successful_model

print(f"\n3. Downloading tokenizer for {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("   Tokenizer downloaded!")

print("\n4. Downloading model (this will take time)...")
print("   Model size: ~13GB for Llama 2 7B")
print("   Downloading to cache...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=TOKEN,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

print("\n" + "=" * 60)
print(f"SUCCESS! {MODEL_ID} is downloaded and ready!")
print("=" * 60)

# Quick test
print("\nQuick test - generating text...")
inputs = tokenizer("The code smell in this function is", return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Response: {response}")

print(f"\nModel {MODEL_ID} is ready for training!")
print("Saved to HuggingFace cache for future use.")