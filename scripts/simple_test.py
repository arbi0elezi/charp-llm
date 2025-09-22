import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path

# Configuration
MODEL_BASE = "models/dclb"
MODEL_FINETUNED = "models/tff_rag"

print("=" * 80)
print("SIMPLE MODEL TEST")
print("=" * 80)

# Device check
if not torch.cuda.is_available():
    print("[ERROR] CUDA is not available")
    exit(1)

device = torch.device("cuda")
print(f"[INFO] Using device: {device}")
print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

# Load tokenizer
print("\n[INFO] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_FINETUNED, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"[INFO] Tokenizer loaded. Vocab size: {len(tokenizer)}")

# Load model with quantization
print("\n[INFO] Loading model...")
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_BASE,
    quantization_config=quant_config,
    trust_remote_code=True,
    device_map="auto"
)

# Resize embeddings to match fine-tuned tokenizer
base_model.resize_token_embeddings(len(tokenizer))

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, MODEL_FINETUNED)
model.config.use_cache = False
model.eval()
print("[INFO] Model loaded successfully!")

# Test prompt
test_prompt = """You are a code smell classifier. Given the following C# snippet, classify it.

Code:
public int Calculate(int a, int b, int c, int d, int e, int f, int g) {
    if (a > 0) {
        if (b > 0) {
            if (c > 0) {
                return a + b + c + d + e + f + g;
            }
        }
    }
    return 0;
}

In your answer strictly follow this format
FINAL ANSWER: followed by the label, example: FINAL ANSWER: ComplexMethod"""

print("\n[INFO] Testing model with sample input...")
print("-" * 80)
print("Input prompt (truncated):")
print(test_prompt[:200] + "...")

# Tokenize
inputs = tokenizer(
    test_prompt,
    return_tensors="pt",
    truncation=True,
    max_length=512,
    padding=True
).to(device)

print("\n[INFO] Generating response...")
# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.1,
        do_sample=True,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=False
    )

# Decode
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("-" * 80)
print("Model response:")
print(response)

# Extract prediction
if "FINAL ANSWER:" in response:
    prediction = response.split("FINAL ANSWER:")[-1].strip().split()[0]
    print(f"\n[INFO] Extracted prediction: {prediction}")
else:
    print("\n[WARNING] Could not extract prediction from response")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)