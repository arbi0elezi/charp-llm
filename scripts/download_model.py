from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_id = "deepseek-ai/DeepSeek-Coder-V2-Lite-Base"
local_path = os.path.abspath("../models/dclb")  # Adjust if needed, e.g., to "models/dcb" for your original script

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

tokenizer.save_pretrained(local_path)
model.save_pretrained(local_path)

print(f"[INFO] Model downloaded and saved to: {local_path}")