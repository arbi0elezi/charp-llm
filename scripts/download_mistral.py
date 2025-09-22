from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Mistral 7B - publicly available, no authentication required
# Similar performance to Llama 8B for code smell detection
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
local_path = os.path.abspath("../models/mistral_7b_base")

print(f"[INFO] Downloading {model_id}...")
print(f"[INFO] Target path: {local_path}")
print("[INFO] This is a publicly available model, no authentication required")

try:
    # Download tokenizer
    print("[INFO] Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Download model
    print("[INFO] Downloading model (this will take a while, ~14GB)...")
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    
    # Save locally
    print(f"[INFO] Saving model to {local_path}...")
    os.makedirs(local_path, exist_ok=True)
    tokenizer.save_pretrained(local_path)
    model.save_pretrained(local_path)
    
    print(f"[INFO] Model successfully downloaded and saved to: {local_path}")
    print("[INFO] You can now use this model for evaluation and fine-tuning")
    
except Exception as e:
    print(f"[ERROR] Failed to download model: {e}")
    import traceback
    traceback.print_exc()