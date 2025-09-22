from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys

# You need to be logged in to HuggingFace to download Llama models
# Run: huggingface-cli login
# Or set HF_TOKEN environment variable

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
local_path = os.path.abspath("../models/llama_8b_base")

print(f"[INFO] Downloading {model_id}...")
print(f"[INFO] Target path: {local_path}")

try:
    # Download tokenizer
    print("[INFO] Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN")  # Use token from environment if available
    )
    
    # Download model
    print("[INFO] Downloading model (this will take a while, ~16GB)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True,
        token=os.getenv("HF_TOKEN")  # Use token from environment if available
    )
    
    # Save locally
    print(f"[INFO] Saving model to {local_path}...")
    os.makedirs(local_path, exist_ok=True)
    tokenizer.save_pretrained(local_path)
    model.save_pretrained(local_path)
    
    print(f"[INFO] Model successfully downloaded and saved to: {local_path}")
    
except Exception as e:
    print(f"[ERROR] Failed to download model: {e}")
    print("\n[INFO] To download Llama models, you need to:")
    print("  1. Request access at: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct")
    print("  2. Login with: huggingface-cli login")
    print("  3. Or set HF_TOKEN environment variable with your token")
    sys.exit(1)