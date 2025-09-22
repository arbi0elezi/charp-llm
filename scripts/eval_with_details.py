import os
import json
import torch
import time
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter

# Configuration
MODEL_BASE = "models/dclb"
MODEL_FINETUNED = "models/tff_rag"
TEST_DATA = "dataset/test/tmf.jsonl"
OUTPUT_DIR = "evaluation_results"
DETAILED_OUTPUT = "evaluation_results/detailed_predictions.json"

print("=" * 80)
print("DETAILED MODEL EVALUATION")
print("=" * 80)
print(f"[INFO] Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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

print(f"[INFO] Loaded {len(test_samples)} test samples")

# Get label distribution
label_counts = Counter(s['label'] for s in test_samples)
print("\n[INFO] Label distribution in test set:")
for label, count in sorted(label_counts.items()):
    print(f"  - {label}: {count} samples ({count/len(test_samples)*100:.1f}%)")

# Load tokenizer
print("\n[INFO] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_FINETUNED, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
print("\n[INFO] Loading model (this may take a minute)...")
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

# Clear GPU cache
torch.cuda.empty_cache()

# Evaluation template
EVAL_TEMPLATE = """You are a code smell classifier. Given the following C# snippet, classify it.

Code:
{text}

In your answer strictly follow this format
FINAL ANSWER: followed by the label, example: FINAL ANSWER: ComplexMethod"""

def predict(text):
    """Make prediction for a single sample."""
    # Truncate text if too long
    text_truncated = text[:600] if len(text) > 600 else text
    prompt = EVAL_TEMPLATE.replace("{text}", text_truncated)
    
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
    
    # Extract prediction from response
    prediction = "Unknown"
    if "FINAL ANSWER:" in response:
        parts = response.split("FINAL ANSWER:")[-1].strip()
        # Take first word after FINAL ANSWER
        if parts:
            prediction = parts.split()[0].strip()
    
    return prompt, response, prediction

# Run evaluation
print("\n" + "=" * 80)
print("STARTING EVALUATION")
print("=" * 80)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Store detailed results
detailed_results = []
true_labels = []
predicted_labels = []
correct = 0
start_time = time.time()

print(f"\n[INFO] Evaluating {len(test_samples)} samples...")
print("[INFO] Progress will be shown every 10 samples")

for i, sample in enumerate(test_samples):
    if i % 10 == 0:
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1) if i > 0 else 0
        remaining = avg_time * (len(test_samples) - i)
        print(f"  [{i:3}/{len(test_samples)}] {i/len(test_samples)*100:5.1f}% | "
              f"Correct: {correct}/{i if i > 0 else 1} | "
              f"Time: {elapsed:.1f}s | ETA: {remaining:.1f}s")
    
    # Get true label
    true_label = sample['label']
    
    # Make prediction
    prompt, response, predicted_label = predict(sample['text'])
    
    # Store results
    true_labels.append(true_label)
    predicted_labels.append(predicted_label)
    
    # Track correctness
    is_correct = (true_label == predicted_label)
    if is_correct:
        correct += 1
    
    # Store detailed result
    detailed_results.append({
        "index": i,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "correct": is_correct,
        "original_code": sample['text'],
        "prompt": prompt,
        "model_response": response
    })
    
    # Clear GPU cache periodically
    if i % 50 == 0 and i > 0:
        torch.cuda.empty_cache()

# Final timing
elapsed = time.time() - start_time
print(f"\n[INFO] Evaluation completed in {elapsed:.1f} seconds")
print(f"[INFO] Average time per sample: {elapsed/len(test_samples):.2f} seconds")

# Calculate metrics
print("\n[INFO] Calculating metrics...")
accuracy = accuracy_score(true_labels, predicted_labels)
unique_labels = sorted(list(set(true_labels + predicted_labels)))

# Confusion matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=[l for l in unique_labels if l != "Unknown"])

# Generate classification report
try:
    report = classification_report(
        true_labels, predicted_labels,
        labels=[l for l in unique_labels if l != "Unknown"],
        zero_division=0,
        output_dict=True
    )
except:
    report = {}

# Print results summary
print("\n" + "=" * 80)
print("EVALUATION RESULTS")
print("=" * 80)
print(f"Overall Accuracy: {accuracy:.4f} ({correct}/{len(test_samples)})")
print(f"Total evaluation time: {elapsed:.1f} seconds")

# Show per-class performance
print("\n" + "-" * 80)
print("PER-CLASS PERFORMANCE")
print("-" * 80)
for label in unique_labels:
    if label != "Unknown" and label in report:
        metrics = report[label]
        print(f"{label:25} | "
              f"Precision: {metrics['precision']:.3f} | "
              f"Recall: {metrics['recall']:.3f} | "
              f"F1: {metrics['f1-score']:.3f} | "
              f"Support: {int(metrics['support'])}")

# Show confusion breakdown
print("\n" + "-" * 80)
print("TOP PREDICTION PATTERNS")
print("-" * 80)
confusion = {}
for true, pred in zip(true_labels, predicted_labels):
    key = f"{true} -> {pred}"
    confusion[key] = confusion.get(key, 0) + 1

for key, count in sorted(confusion.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {key:40} : {count:3} samples")

# Save detailed results
print("\n[INFO] Saving detailed results...")
with open(DETAILED_OUTPUT, 'w', encoding='utf-8') as f:
    json.dump(detailed_results, f, indent=2, ensure_ascii=False)
print(f"[INFO] Detailed predictions saved to: {DETAILED_OUTPUT}")

# Save summary results
summary = {
    "timestamp": datetime.now().isoformat(),
    "model_path": MODEL_FINETUNED,
    "test_data": TEST_DATA,
    "total_samples": len(test_samples),
    "correct_predictions": correct,
    "accuracy": accuracy,
    "evaluation_time_seconds": elapsed,
    "avg_time_per_sample": elapsed/len(test_samples),
    "label_distribution": dict(label_counts),
    "confusion_patterns": confusion,
    "classification_report": report,
    "confusion_matrix": cm.tolist() if 'cm' in locals() else None,
    "unique_labels": unique_labels
}

summary_file = os.path.join(OUTPUT_DIR, "evaluation_summary.json")
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"[INFO] Summary saved to: {summary_file}")

# Create a human-readable report
report_file = os.path.join(OUTPUT_DIR, "evaluation_report.txt")
with open(report_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("C# CODE SMELL DETECTION - EVALUATION REPORT\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model: {MODEL_FINETUNED}\n")
    f.write(f"Test Dataset: {TEST_DATA}\n")
    f.write(f"Total Samples: {len(test_samples)}\n")
    f.write(f"Evaluation Time: {elapsed:.1f} seconds\n\n")
    
    f.write("-" * 80 + "\n")
    f.write("OVERALL PERFORMANCE\n")
    f.write("-" * 80 + "\n")
    f.write(f"Accuracy: {accuracy:.4f} ({correct}/{len(test_samples)})\n")
    f.write(f"Incorrect: {len(test_samples) - correct}\n")
    f.write(f"Average time per sample: {elapsed/len(test_samples):.2f} seconds\n\n")
    
    if report:
        f.write("-" * 80 + "\n")
        f.write("DETAILED CLASSIFICATION REPORT\n")
        f.write("-" * 80 + "\n")
        report_text = classification_report(
            true_labels, predicted_labels,
            labels=[l for l in unique_labels if l != "Unknown"],
            zero_division=0
        )
        f.write(report_text)
        f.write("\n")
    
    f.write("-" * 80 + "\n")
    f.write("CONFUSION PATTERNS\n")
    f.write("-" * 80 + "\n")
    for key, count in sorted(confusion.items(), key=lambda x: x[1], reverse=True):
        f.write(f"{key:40} : {count:3} samples\n")

print(f"[INFO] Report saved to: {report_file}")

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
print(f"[INFO] Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nResults saved to:")
print(f"  1. Detailed predictions: {DETAILED_OUTPUT}")
print(f"  2. Summary: {summary_file}")
print(f"  3. Report: {report_file}")