import os
import json
import torch
import time
import sys
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

def evaluate_model(model_path, base_model_path=None, test_data_path="dataset/test/ds.jsonl"):
    """
    Unified evaluation function for any model.
    
    Args:
        model_path: Path to the model to evaluate
        base_model_path: Path to base model (for LoRA adapters)
        test_data_path: Path to test dataset
    """
    
    # Get model name for folder organization
    model_name = Path(model_path).name
    
    # Setup output directories
    OUTPUT_BASE = "evaluation_results"
    OUTPUT_DIR = os.path.join(OUTPUT_BASE, model_name)
    DETAILED_OUTPUT = os.path.join(OUTPUT_DIR, "detailed_predictions.json")
    PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress.json")
    SUMMARY_FILE = os.path.join(OUTPUT_DIR, "evaluation_summary.json")
    
    print("=" * 80)
    print(f"MODEL EVALUATION: {model_name}")
    print("=" * 80)
    print(f"[INFO] Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] Model path: {model_path}")
    print(f"[INFO] Output directory: {OUTPUT_DIR}")
    
    # Device setup
    device = torch.device("cuda")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load test data
    print(f"\n[INFO] Loading test data from: {test_data_path}")
    test_samples = []
    with open(test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_samples.append(json.loads(line))
    
    print(f"[INFO] Loaded {len(test_samples)} test samples")
    
    # Get label distribution
    label_counts = Counter(s['label'] for s in test_samples)
    print("\n[INFO] Label distribution in test set:")
    for label, count in sorted(label_counts.items()):
        print(f"  - {label}: {count} samples ({count/len(test_samples)*100:.1f}%)")
    
    # Check for existing progress
    start_index = 0
    existing_results = []
    
    if os.path.exists(DETAILED_OUTPUT):
        print(f"\n[INFO] Found existing results for {model_name}, resuming from last position...")
        with open(DETAILED_OUTPUT, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)
            start_index = len(existing_results)
            print(f"[INFO] Resuming from sample {start_index}")
    
    # Save initial progress
    progress = {
        "model_name": model_name,
        "model_path": model_path,
        "total_samples": len(test_samples),
        "completed": start_index,
        "status": "loading_model",
        "last_update": datetime.now().isoformat()
    }
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)
    
    # Load tokenizer
    print(f"\n[INFO] Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print(f"\n[INFO] Loading model {model_name} (this may take a minute)...")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    
    # Check if this is a LoRA adapter
    adapter_config_path = Path(model_path) / "adapter_config.json"
    is_lora = adapter_config_path.exists()
    
    if is_lora and base_model_path:
        print(f"[INFO] Detected LoRA adapter, loading base model + adapter")
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=quant_config,
            trust_remote_code=True,
            device_map="auto"
        )
        # Resize embeddings to match tokenizer
        base_model.resize_token_embeddings(len(tokenizer))
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, model_path)
        model_type = "fine-tuned"
    else:
        print(f"[INFO] Loading as standard model")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            trust_remote_code=True,
            device_map="auto"
        )
        model_type = "baseline"
    
    model.config.use_cache = False
    model.eval()
    print(f"[INFO] Model {model_name} loaded successfully!")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Update progress
    progress["status"] = "evaluating"
    progress["model_type"] = model_type
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)
    
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
                max_new_tokens=30,
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
            if parts:
                prediction = parts.split()[0].strip()
        
        return prompt, response, prediction
    
    # Run evaluation
    print("\n" + "=" * 80)
    print(f"STARTING EVALUATION: {model_name}")
    print("=" * 80)
    
    # Initialize or use existing results
    detailed_results = existing_results.copy()
    start_time = time.time()
    
    print(f"\n[INFO] Evaluating samples {start_index} to {len(test_samples)}...")
    print("[INFO] Results will be saved after each prediction")
    
    for i in range(start_index, len(test_samples)):
        sample = test_samples[i]
        
        # Show progress
        if i % 5 == 0:
            elapsed = time.time() - start_time
            if i > start_index:
                avg_time = elapsed / (i - start_index)
                remaining = avg_time * (len(test_samples) - i)
                eta_str = f"ETA: {remaining:.1f}s"
            else:
                eta_str = "Calculating ETA..."
            
            # Calculate current accuracy
            if len(detailed_results) > 0:
                current_correct = sum(1 for r in detailed_results if r['correct'])
                current_acc = current_correct / len(detailed_results)
                acc_str = f"Acc: {current_acc:.3f}"
            else:
                acc_str = "Acc: N/A"
                
            print(f"  [{model_name}] [{i:3}/{len(test_samples)}] {i/len(test_samples)*100:5.1f}% | {acc_str} | {eta_str}")
        
        # Get true label
        true_label = sample['label']
        
        # Make prediction
        try:
            prompt, response, predicted_label = predict(sample['text'])
        except Exception as e:
            print(f"  [ERROR] Failed on sample {i}: {e}")
            prompt = EVAL_TEMPLATE.replace("{text}", sample['text'][:600])
            response = f"ERROR: {str(e)}"
            predicted_label = "Unknown"
        
        # Track correctness
        is_correct = (true_label == predicted_label)
        
        # Store detailed result
        result = {
            "index": i,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "correct": is_correct,
            "original_code": sample['text'],
            "prompt": prompt,
            "model_response": response,
            "model_name": model_name,
            "model_type": model_type
        }
        detailed_results.append(result)
        
        # Save after each prediction
        with open(DETAILED_OUTPUT, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        # Update progress file
        progress = {
            "model_name": model_name,
            "model_path": model_path,
            "model_type": model_type,
            "total_samples": len(test_samples),
            "completed": i + 1,
            "status": "evaluating",
            "current_accuracy": sum(1 for r in detailed_results if r['correct']) / len(detailed_results),
            "last_prediction": predicted_label,
            "last_true_label": true_label,
            "last_update": datetime.now().isoformat()
        }
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f, indent=2)
        
        # Clear GPU cache periodically
        if i % 20 == 0 and i > 0:
            torch.cuda.empty_cache()
    
    # Final calculations
    elapsed = time.time() - start_time
    true_labels = [r['true_label'] for r in detailed_results]
    predicted_labels = [r['predicted_label'] for r in detailed_results]
    accuracy = accuracy_score(true_labels, predicted_labels)
    unique_labels = sorted(list(set(true_labels + predicted_labels)))
    
    # Classification report
    try:
        report = classification_report(
            true_labels, predicted_labels,
            labels=[l for l in unique_labels if l != "Unknown"],
            zero_division=0,
            output_dict=True
        )
    except:
        report = {}
    
    # Print results
    print(f"\n[INFO] {model_name} evaluation completed in {elapsed:.1f} seconds")
    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS: {model_name}")
    print("=" * 80)
    correct = sum(1 for r in detailed_results if r['correct'])
    print(f"Model: {model_name} ({model_type})")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(detailed_results)})")
    print(f"Evaluation time: {elapsed:.1f} seconds")
    
    # Confusion patterns
    confusion = {}
    for r in detailed_results:
        key = f"{r['true_label']} -> {r['predicted_label']}"
        confusion[key] = confusion.get(key, 0) + 1
    
    print("\n" + "-" * 80)
    print("TOP PREDICTION PATTERNS")
    print("-" * 80)
    for key, count in sorted(confusion.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {key:40} : {count:3} samples")
    
    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "model_path": model_path,
        "model_type": model_type,
        "test_data": test_data_path,
        "total_samples": len(detailed_results),
        "correct_predictions": correct,
        "accuracy": accuracy,
        "evaluation_time_seconds": elapsed,
        "avg_time_per_sample": elapsed/len(detailed_results),
        "label_distribution": dict(label_counts),
        "confusion_patterns": confusion,
        "classification_report": report,
        "unique_labels": unique_labels
    }
    
    with open(SUMMARY_FILE, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Update final progress
    progress = {
        "model_name": model_name,
        "model_path": model_path,
        "model_type": model_type,
        "total_samples": len(test_samples),
        "completed": len(test_samples),
        "status": "completed",
        "final_accuracy": accuracy,
        "last_update": datetime.now().isoformat()
    }
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)
    
    print(f"\n[INFO] Results saved to:")
    print(f"  - Directory: {OUTPUT_DIR}")
    print(f"  - Detailed: {DETAILED_OUTPUT}")
    print(f"  - Summary: {SUMMARY_FILE}")
    print(f"  - Progress: {PROGRESS_FILE}")
    
    return accuracy, summary

def main():
    """Main function with command line argument support."""
    if len(sys.argv) < 2:
        print("Usage: python eval_unified.py <model_path> [base_model_path]")
        print("Examples:")
        print("  python eval_unified.py models/dclb")
        print("  python eval_unified.py models/tff_rag models/dclb")
        return
    
    model_path = sys.argv[1]
    base_model_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(model_path):
        print(f"Error: Model path {model_path} does not exist!")
        return
    
    accuracy, summary = evaluate_model(model_path, base_model_path)
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Final Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()