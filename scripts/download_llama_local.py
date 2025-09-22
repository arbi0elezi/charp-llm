from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import sys

# Llama 8B model - will be saved locally for offline use
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
local_path = os.path.abspath("../models/llama_8b_baseline")

print("=" * 80)
print("LLAMA 8B LOCAL DOWNLOAD")
print("=" * 80)
print(f"[INFO] Model: {model_id}")
print(f"[INFO] Target path: {local_path}")
print("[INFO] Once downloaded, this model can be used offline without authentication")

# Check for token
token = os.getenv("HF_TOKEN")
if not token:
    print("\n[WARNING] HF_TOKEN not set.")
    print("To download Llama models, you need to:")
    print("1. Request access at: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct")
    print("2. Set HF_TOKEN environment variable with your token")
    print("\nAlternatively, you can use the Qwen model which is already downloaded.")
    response = input("\nDo you have a HuggingFace token with Llama access? (y/n): ")
    if response.lower() != 'y':
        print("\nSkipping Llama download. You can use Qwen instead.")
        sys.exit(0)
    token = input("Enter your HuggingFace token: ").strip()
    os.environ["HF_TOKEN"] = token

try:
    # Download tokenizer
    print("\n[INFO] Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=token)
    
    # Download model
    print("[INFO] Downloading model (this will take a while, ~16GB)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True,
        token=token
    )
    
    # Save locally
    print(f"\n[INFO] Saving model to {local_path}...")
    os.makedirs(local_path, exist_ok=True)
    tokenizer.save_pretrained(local_path)
    model.save_pretrained(local_path)
    
    print(f"\n✓ Model successfully downloaded and saved to: {local_path}")
    print("\n[INFO] You can now use this model locally without authentication:")
    print(f"  - For training: python scripts/train_llama_balanced.py --base_model {local_path}")
    print(f"  - For evaluation: python scripts/eval_llama.py baseline --model_path {local_path}")
    
except Exception as e:
    print(f"\n[ERROR] Failed to download model: {e}")
    print("\nIf you don't have Llama access, you can use the Qwen model instead:")
    print("  - Qwen is already downloaded at: models/qwen_7b_base")
    print("  - It's a powerful 7B model that doesn't require authentication")
    sys.exit(1)