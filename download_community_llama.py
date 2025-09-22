"""
Download community Llama models that don't require Meta approval
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# Community Llama models - no Meta approval needed
MODELS = [
    "NousResearch/Llama-2-7b-hf",         # Community Llama 2 7B
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", # Tiny version (1.1B)
]

print("=" * 60)
print("COMMUNITY LLAMA MODELS (No Meta approval needed)")
print("=" * 60)

for model_id in MODELS:
    print(f"\nTesting: {model_id}")
    print("-" * 40)
    
    try:
        # Test access
        config = AutoConfig.from_pretrained(model_id)
        print(f"[OK] Model accessible!")
        print(f"     Type: {config.model_type}")
        print(f"     Hidden size: {config.hidden_size}")
        print(f"     Layers: {config.num_hidden_layers}")
        print(f"     Max length: {config.max_position_embeddings}")
        
        # Ask to download
        print(f"\nDownload {model_id}? (y/n): ", end="")
        choice = "y"  # Auto-yes for now
        print(choice)
        
        if choice.lower() == "y":
            print(f"\nDownloading {model_id}...")
            
            # Download tokenizer
            print("1. Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Download model
            print("2. Downloading model...")
            if "TinyLlama" in model_id:
                print("   Size: ~2.2GB")
            else:
                print("   Size: ~13GB")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            
            print(f"\n[SUCCESS] {model_id} downloaded!")
            
            # Quick test
            print("\nQuick test...")
            inputs = tokenizer("The code exhibits", return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=10, temperature=0.7)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Response: {response}")
            
            print(f"\n{model_id} is ready for training!")
            break
            
    except Exception as e:
        print(f"[ERROR] {str(e)[:200]}")

print("\n" + "=" * 60)