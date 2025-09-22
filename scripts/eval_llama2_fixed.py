"""
Fixed evaluation script for Llama 2 models (baseline and fine-tuned)
Fixes prompt format and generation parameters
"""

import os
import sys
import json
import torch
import time
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

def evaluate_model(model_path, model_name, output_dir, is_baseline=False):
    """
    Evaluate a Llama 2 model on the test dataset
    
    Args:
        model_path: Path to model or HuggingFace model ID
        model_name: Name for output directory
        output_dir: Base output directory
        is_baseline: If True, load from HuggingFace; if False, load local fine-tuned
    """
    
    print("=" * 80)
    print(f"MODEL EVALUATION: {model_name}")
    print("=" * 80)
    print(f"[INFO] Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] Model path: {model_path}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Using device: cuda" if torch.cuda.is_available() else "[INFO] Using device: cpu")
    
    if torch.cuda.is_available():
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
    
    # Load test data
    TEST_DATA_PATH = "dataset/test/ds.jsonl"
    print(f"\n[INFO] Loading test data from: {TEST_DATA_PATH}")
    
    test_samples = []
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            test_samples.append(json.loads(line.strip()))
    
    print(f"[INFO] Loaded {len(test_samples)} test samples")
    
    # Show label distribution
    label_counts = Counter(s['label'] for s in test_samples)
    print("\n[INFO] Label distribution in test set:")
    for label, count in sorted(label_counts.items()):
        print(f"  - {label}: {count} samples ({count/len(test_samples)*100:.1f}%)")
    
    # Load tokenizer
    print(f"\n[INFO] Loading tokenizer...")
    if is_baseline:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load model
    print(f"\n[INFO] Loading model {model_name}...")
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    if is_baseline:
        # Load baseline model
        print("[INFO] Loading baseline model with 4-bit quantization")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
    else:
        # Load fine-tuned model (with LoRA adapters)
        print("[INFO] Loading fine-tuned model with LoRA adapters")
        
        # First load base model
        base_model_id = "NousResearch/Llama-2-7b-hf"
        model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        # Load LoRA adapters
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()  # Merge LoRA weights
    
    model.eval()
    print(f"[INFO] Model {model_name} loaded successfully!")
    
    # Create output directory
    model_output_dir = Path(output_dir) / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Files for saving results
    DETAILED_OUTPUT = model_output_dir / "detailed_predictions.json"
    PROGRESS_FILE = model_output_dir / "progress.json"
    SUMMARY_FILE = model_output_dir / "evaluation_summary.json"
    
    print("\n" + "=" * 80)
    print(f"STARTING EVALUATION: {model_name}")
    print("=" * 80)
    
    # Evaluation loop
    detailed_results = []
    predictions = []
    true_labels = []
    
    print(f"\n[INFO] Evaluating {len(test_samples)} samples...")
    print("[INFO] Results will be saved after each prediction")
    
    start_time = time.time()
    
    for i, sample in enumerate(test_samples):
        # Progress update every 5 samples
        if i % 5 == 0:
            elapsed = time.time() - start_time
            if i > 0:
                avg_time = elapsed / i
                remaining = avg_time * (len(test_samples) - i)
                eta_str = f"ETA: {remaining:.1f}s"
            else:
                eta_str = "Calculating ETA..."
            
            # Calculate current accuracy
            if len(predictions) > 0:
                current_acc = accuracy_score(true_labels[:len(predictions)], predictions)
                acc_str = f"Acc: {current_acc:.3f}"
            else:
                acc_str = "Acc: N/A"
            
            print(f"  [{model_name}] [{i:3}/{len(test_samples)}] {i/len(test_samples)*100:5.1f}% | {acc_str} | {eta_str}")
        
        # Get code and true label
        code = sample.get('code', sample.get('text', ''))
        true_label = sample['label']
        
        # Create prompt based on model type
        if is_baseline:
            # Simple prompt for baseline
            prompt = f"""Classify the following C# code snippet as one of these code smells: ComplexConditional, ComplexMethod, or FeatureEnvy.

Code:
```csharp
{code[:800]}
```

The code smell is:"""
        else:
            # Training format for fine-tuned model
            prompt = f"""<s>[INST] <<SYS>>
You are a code smell classifier. Analyze C# code and identify the specific code smell present.
<</SYS>>

Classify this C# code snippet and identify the code smell:

```csharp
{code[:800]}
```

What code smell does this code exhibit? [/INST]

This code exhibits the"""
        
        # Generate prediction
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode only the generated part
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Extract predicted label from response
            predicted_label = "Unknown"
            response_lower = response.lower().strip()
            
            # Check for exact matches first
            if "complexconditional" in response_lower or "complex conditional" in response_lower:
                predicted_label = "ComplexConditional"
            elif "complexmethod" in response_lower or "complex method" in response_lower:
                predicted_label = "ComplexMethod"  
            elif "featureenvy" in response_lower or "feature envy" in response_lower:
                predicted_label = "FeatureEnvy"
            # Check for partial matches
            elif "conditional" in response_lower:
                predicted_label = "ComplexConditional"
            elif "method" in response_lower:
                predicted_label = "ComplexMethod"
            elif "envy" in response_lower or "feature" in response_lower:
                predicted_label = "FeatureEnvy"
            
        except Exception as e:
            print(f"  [ERROR] Failed on sample {i}: {e}")
            response = f"ERROR: {str(e)}"
            predicted_label = "Unknown"
        
        # Track results
        predictions.append(predicted_label)
        true_labels.append(true_label)
        
        # Store detailed result
        result = {
            "index": i,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "correct": (true_label == predicted_label),
            "original_code": code,
            "prompt": prompt[-500:],  # Save last 500 chars of prompt
            "model_response": response,
            "model_name": model_name,
            "model_type": "baseline" if is_baseline else "fine-tuned"
        }
        detailed_results.append(result)
        
        # Save progress periodically
        if i % 10 == 0:
            with open(DETAILED_OUTPUT, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, indent=2, ensure_ascii=False)
            
            progress = {
                "model_name": model_name,
                "model_path": model_path,
                "model_type": "baseline" if is_baseline else "fine-tuned",
                "total_samples": len(test_samples),
                "completed": i + 1,
                "status": "evaluating",
                "current_accuracy": accuracy_score(true_labels[:i+1], predictions[:i+1]) if i > 0 else 0,
                "last_prediction": predicted_label,
                "last_true_label": true_label,
                "last_update": datetime.now().isoformat()
            }
            with open(PROGRESS_FILE, 'w') as f:
                json.dump(progress, f, indent=2)
    
    # Final evaluation
    elapsed = time.time() - start_time
    accuracy = accuracy_score(true_labels, predictions)
    
    print(f"\n[INFO] {model_name} evaluation completed in {elapsed:.1f} seconds")
    
    # Calculate metrics
    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS: {model_name}")
    print("=" * 80)
    print(f"Model: {model_name} ({'baseline' if is_baseline else 'fine-tuned'})")
    print(f"Accuracy: {accuracy:.4f} ({sum(p == t for p, t in zip(predictions, true_labels))}/{len(test_samples)})")
    print(f"Evaluation time: {elapsed:.1f} seconds")
    
    # Confusion patterns
    confusion = Counter(f"{t} -> {p}" for t, p in zip(true_labels, predictions))
    print("\n" + "-" * 80)
    print("TOP PREDICTION PATTERNS")
    print("-" * 80)
    for pattern, count in confusion.most_common(10):
        print(f"  {pattern:40} : {count:3} samples")
    
    # Classification report
    unique_labels = sorted(set(true_labels + predictions))
    report = classification_report(true_labels, predictions, labels=unique_labels, output_dict=True, zero_division=0)
    
    # Save final results
    with open(DETAILED_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "model_path": model_path,
        "model_type": "baseline" if is_baseline else "fine-tuned",
        "test_data": TEST_DATA_PATH,
        "total_samples": len(test_samples),
        "correct_predictions": sum(p == t for p, t in zip(predictions, true_labels)),
        "accuracy": accuracy,
        "evaluation_time_seconds": elapsed,
        "avg_time_per_sample": elapsed/len(test_samples),
        "label_distribution": dict(label_counts),
        "confusion_patterns": dict(confusion),
        "classification_report": report,
        "unique_labels": unique_labels
    }
    
    with open(SUMMARY_FILE, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Update final progress
    progress = {
        "model_name": model_name,
        "model_path": model_path,
        "model_type": "baseline" if is_baseline else "fine-tuned",
        "total_samples": len(test_samples),
        "completed": len(test_samples),
        "status": "completed",
        "final_accuracy": accuracy,
        "last_update": datetime.now().isoformat()
    }
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)
    
    print(f"\n[INFO] Results saved to:")
    print(f"  - Directory: {model_output_dir}")
    print(f"  - Detailed: {DETAILED_OUTPUT}")
    print(f"  - Summary: {SUMMARY_FILE}")
    print(f"  - Progress: {PROGRESS_FILE}")
    
    return accuracy, summary

def main():
    """Main function to evaluate both baseline and fine-tuned Llama 2 models"""
    
    OUTPUT_BASE = "evaluation_results"
    
    # Test 1: Baseline Llama 2
    print("\n" + "=" * 80)
    print("TESTING LLAMA 2 BASELINE MODEL")
    print("=" * 80)
    
    baseline_acc, baseline_summary = evaluate_model(
        model_path="NousResearch/Llama-2-7b-hf",
        model_name="llama2_baseline_fixed",
        output_dir=OUTPUT_BASE,
        is_baseline=True
    )
    
    print("\n" + "=" * 80)
    print("BASELINE EVALUATION COMPLETE")
    print(f"Accuracy: {baseline_acc:.4f}")
    print("=" * 80)
    
    # Test 2: Fine-tuned Llama 2
    print("\n" + "=" * 80)
    print("TESTING LLAMA 2 FINE-TUNED MODEL V2")
    print("=" * 80)
    
    finetuned_acc, finetuned_summary = evaluate_model(
        model_path="models/llama2_finetuned_v2",
        model_name="llama2_finetuned_v2",
        output_dir=OUTPUT_BASE,
        is_baseline=False
    )
    
    print("\n" + "=" * 80)
    print("FINE-TUNED EVALUATION COMPLETE")
    print(f"Accuracy: {finetuned_acc:.4f}")
    print("=" * 80)
    
    # Final comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    print(f"Baseline Llama 2:   {baseline_acc:.4f}")
    print(f"Fine-tuned Llama 2: {finetuned_acc:.4f}")
    print(f"Improvement:        {(finetuned_acc - baseline_acc):.4f} ({(finetuned_acc - baseline_acc)*100:.2f}%)")
    print("=" * 80)

if __name__ == "__main__":
    main()