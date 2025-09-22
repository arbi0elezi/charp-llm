from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Qwen2.5 7B Instruct - publicly available, no authentication required
# High-performance model comparable to Llama 8B
model_id = "Qwen/Qwen2.5-7B-Instruct"
local_path = os.path.abspath("../models/qwen_7b_base")

print(f"[INFO] Downloading {model_id}...")
print(f"[INFO] Target path: {local_path}")
print("[INFO] This is a publicly available model, no authentication required")

try:
    # Download tokenizer
    print("[INFO] Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Download model
    print("[INFO] Downloading model (this will take a while, ~15GB)...")
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    
    # Save locally
    print(f"[INFO] Saving model to {local_path}...")
    os.makedirs(local_path, exist_ok=True)
    tokenizer.save_pretrained(local_path)
    model.save_pretrained(local_path)
    
    print(f"[INFO] Model successfully downloaded and saved to: {local_path}")
    print("[INFO] You can now use this model for evaluation and fine-tuning")
    print("[INFO] Qwen2.5 7B is a state-of-the-art model with excellent code understanding")
    
except Exception as e:
    print(f"[ERROR] Failed to download model: {e}")
    import traceback
    traceback.print_exc()