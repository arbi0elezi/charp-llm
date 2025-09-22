import os
import json
import torch
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

# Configuration
MODEL_BASE = "models/dclb"
MODEL_FINETUNED = "models/tff_rag"
TEST_DATA = "dataset/test/tmf.jsonl"
SAMPLE_SIZE = 50  # Evaluate on 50 samples for quick testing

print("=" * 80)
print("QUICK MODEL EVALUATION")
print("=" * 80)

# Device setup
device = torch.device("cuda")
print(f"[INFO] Using device: {device}")
print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

# Load test data
print(f"\n[INFO] Loading test data from: {TEST_DATA}")
test_samples = []
with open(TEST_DATA, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            test_samples.append(json.loads(line))

# Limit samples
test_samples = test_samples[:SAMPLE_SIZE]
print(f"[INFO] Using {len(test_samples)} samples for evaluation")

# Get label distribution
label_counts = Counter(s['label'] for s in test_samples)
print("\n[INFO] Label distribution:")
for label, count in sorted(label_counts.items()):
    print(f"  - {label}: {count} samples")

# Load tokenizer
print("\n[INFO] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_FINETUNED, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
print("\n[INFO] Loading model...")
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_BASE,
    quantization_config=quant_config,
    trust_remote_code=True,
    device_map="auto"
)
base_model.resize_token_embeddings(len(tokenizer))

model = PeftModel.from_pretrained(base_model, MODEL_FINETUNED)
model.config.use_cache = False
model.eval()
print("[INFO] Model loaded successfully!")

# Evaluation template
EVAL_TEMPLATE = """You are a code smell classifier. Given the following C# snippet, classify it.

Code:
{text}

In your answer strictly follow this format
FINAL ANSWER: followed by the label, example: FINAL ANSWER: ComplexMethod"""

def predict(text):
    """Make prediction for a single sample."""
    prompt = EVAL_TEMPLATE.replace("{text}", text[:600])  # Limit text length
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,  # Reduced to prevent repetition
            temperature=0.1,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract prediction
    if "FINAL ANSWER:" in response:
        parts = response.split("FINAL ANSWER:")[-1].strip()
        # Take first word after FINAL ANSWER
        label = parts.split()[0] if parts else "Unknown"
        return label.strip()
    
    return "Unknown"

# Run evaluation
print("\n" + "=" * 80)
print("STARTING EVALUATION")
print("=" * 80)

true_labels = []
predicted_labels = []
correct = 0
start_time = time.time()

for i, sample in enumerate(test_samples):
    if i % 10 == 0:
        print(f"Progress: {i}/{len(test_samples)} ({i/len(test_samples)*100:.1f}%)")
    
    true_label = sample['label']
    predicted_label = predict(sample['text'])
    
    true_labels.append(true_label)
    predicted_labels.append(predicted_label)
    
    if true_label == predicted_label:
        correct += 1

elapsed = time.time() - start_time
print(f"Progress: {len(test_samples)}/{len(test_samples)} (100.0%)")
print(f"\n[INFO] Evaluation completed in {elapsed:.1f} seconds")

# Calculate metrics
accuracy = accuracy_score(true_labels, predicted_labels)
unique_labels = sorted(list(set(true_labels + predicted_labels)))

print("\n" + "=" * 80)
print("EVALUATION RESULTS")
print("=" * 80)
print(f"Accuracy: {accuracy:.4f} ({correct}/{len(test_samples)})")
print(f"Average time per sample: {elapsed/len(test_samples):.2f} seconds")

# Show confusion
print("\n" + "-" * 80)
print("PREDICTIONS BREAKDOWN")
print("-" * 80)
confusion = {}
for true, pred in zip(true_labels, predicted_labels):
    key = f"{true} -> {pred}"
    confusion[key] = confusion.get(key, 0) + 1

for key, count in sorted(confusion.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {key:40} : {count:3} samples")

# Classification report
print("\n" + "-" * 80)
print("CLASSIFICATION REPORT")
print("-" * 80)
try:
    report = classification_report(
        true_labels, predicted_labels,
        labels=[l for l in unique_labels if l != "Unknown"],
        zero_division=0
    )
    print(report)
except:
    print("[WARNING] Could not generate full classification report")

# Save results
results = {
    "accuracy": accuracy,
    "correct": correct,
    "total": len(test_samples),
    "time_seconds": elapsed,
    "confusion": confusion
}

results_file = "evaluation_results/quick_eval_results.json"
os.makedirs("evaluation_results", exist_ok=True)
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[INFO] Results saved to: {results_file}")

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)