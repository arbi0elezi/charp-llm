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

def evaluate_llama_model(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", model_path=None, is_finetuned=False, test_data_path="dataset/test/ds.jsonl"):
    """
    Evaluation function for Large Language Models (Mistral 7B as alternative to Llama 8B).
    
    Args:
        model_name: HuggingFace model name or local path
        model_path: Path to fine-tuned adapter (if applicable)
        is_finetuned: Whether this is a fine-tuned model
        test_data_path: Path to test dataset
    """
    
    # Setup model identifier for folders
    if is_finetuned:
        model_id = "llama_8b_finetuned"
    else:
        model_id = "llama_8b_baseline"
    
    # Setup output directories
    OUTPUT_BASE = "evaluation_results"
    OUTPUT_DIR = os.path.join(OUTPUT_BASE, model_id)
    DETAILED_OUTPUT = os.path.join(OUTPUT_DIR, "detailed_predictions.json")
    PROGRESS_FILE = os.path.join(OUTPUT_DIR, "progress.json")
    SUMMARY_FILE = os.path.join(OUTPUT_DIR, "evaluation_summary.json")
    
    print("=" * 80)
    print(f"LLAMA 8B MODEL EVALUATION: {model_id}")
    print("=" * 80)
    print(f"[INFO] Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] Model: {model_name}")
    if model_path:
        print(f"[INFO] Adapter path: {model_path}")
    print(f"[INFO] Output directory: {OUTPUT_DIR}")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    if torch.cuda.is_available():
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
        print(f"\n[INFO] Found existing results for {model_id}, resuming from last position...")
        with open(DETAILED_OUTPUT, 'r', encoding='utf-8') as f:
            existing_results = json.load(f)
            start_index = len(existing_results)
            print(f"[INFO] Resuming from sample {start_index}")
    
    # Save initial progress
    progress = {
        "model_name": model_id,
        "model_path": model_name,
        "adapter_path": model_path if model_path else None,
        "total_samples": len(test_samples),
        "completed": start_index,
        "status": "loading_model",
        "last_update": datetime.now().isoformat()
    }
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)
    
    # Check if using local model or HuggingFace model
    token = None
    if model_name.startswith("meta-llama/") or model_name.startswith("mistralai/"):
        # Only need token for downloading from HuggingFace
        token = os.getenv("HF_TOKEN")
        if not token and "llama" in model_name.lower():
            print("\n[WARNING] Using HuggingFace Llama model but HF_TOKEN not set.")
            print("If model is not cached locally, download will fail.")
    
    # Load tokenizer
    print(f"\n[INFO] Loading tokenizer for Llama 8B...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load model with 4-bit quantization
    print(f"\n[INFO] Loading Llama 8B model with 4-bit quantization...")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=compute_dtype,
        attn_implementation="eager",  # Avoid Flash Attention issues on Windows
        token=token  # Pass token for Llama models
    )
    
    # Load fine-tuned adapter if provided
    if model_path and is_finetuned:
        print(f"[INFO] Loading fine-tuned LoRA adapter from {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        model_type = "fine-tuned"
    else:
        model_type = "baseline"
    
    model.eval()
    print(f"[INFO] Llama 8B {model_type} model loaded successfully!")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Update progress
    progress["status"] = "evaluating"
    progress["model_type"] = model_type
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)
    
    # Evaluation prompt template for Llama
    EVAL_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a code smell classifier. Analyze C# code snippets and classify them into one of three categories: ComplexMethod, ComplexConditional, or FeatureEnvy.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
Classify the following C# code snippet:

{text}

Respond with ONLY the classification label (ComplexMethod, ComplexConditional, or FeatureEnvy).<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""
    
    def predict(text):
        """Make prediction for a single sample."""
        # Truncate text if too long
        text_truncated = text[:800] if len(text) > 800 else text
        prompt = EVAL_TEMPLATE.replace("{text}", text_truncated)
        
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the generated tokens
        generated_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Extract prediction from response
        prediction = "Unknown"
        response_lower = response.lower()
        
        if "complexmethod" in response_lower or "complex method" in response_lower:
            prediction = "ComplexMethod"
        elif "complexconditional" in response_lower or "complex conditional" in response_lower:
            prediction = "ComplexConditional"
        elif "featureenvy" in response_lower or "feature envy" in response_lower:
            prediction = "FeatureEnvy"
        else:
            # Try first word as label
            words = response.split()
            if words:
                first_word = words[0].strip(".,!?")
                if first_word in ["ComplexMethod", "ComplexConditional", "FeatureEnvy"]:
                    prediction = first_word
        
        return prompt, response, prediction
    
    # Run evaluation
    print("\n" + "=" * 80)
    print(f"STARTING EVALUATION: {model_id}")
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
                eta_str = f"ETA: {remaining/60:.1f}min"
            else:
                eta_str = "Calculating ETA..."
            
            # Calculate current accuracy
            if len(detailed_results) > 0:
                current_correct = sum(1 for r in detailed_results if r['correct'])
                current_acc = current_correct / len(detailed_results)
                acc_str = f"Acc: {current_acc:.3f}"
            else:
                acc_str = "Acc: N/A"
                
            print(f"  [{model_id}] [{i:3}/{len(test_samples)}] {i/len(test_samples)*100:5.1f}% | {acc_str} | {eta_str}")
        
        # Get true label
        true_label = sample['label']
        
        # Make prediction
        try:
            prompt, response, predicted_label = predict(sample.get('text', sample.get('code', '')))
        except Exception as e:
            print(f"  [ERROR] Failed on sample {i}: {e}")
            prompt = EVAL_TEMPLATE.replace("{text}", sample.get('text', sample.get('code', ''))[:800])
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
            "original_code": sample.get('text', sample.get('code', '')),
            "prompt": prompt,
            "model_response": response,
            "model_name": model_id,
            "model_type": model_type
        }
        detailed_results.append(result)
        
        # Save after each prediction
        with open(DETAILED_OUTPUT, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        
        # Update progress file
        progress = {
            "model_name": model_id,
            "model_path": model_name,
            "adapter_path": model_path if model_path else None,
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
        if i % 20 == 0 and i > 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Final calculations
    elapsed = time.time() - start_time
    true_labels = [r['true_label'] for r in detailed_results]
    predicted_labels = [r['predicted_label'] for r in detailed_results]
    
    # Filter out Unknown predictions for accuracy calculation
    valid_indices = [i for i, p in enumerate(predicted_labels) if p != "Unknown"]
    if valid_indices:
        valid_true = [true_labels[i] for i in valid_indices]
        valid_pred = [predicted_labels[i] for i in valid_indices]
        accuracy = accuracy_score(valid_true, valid_pred)
    else:
        accuracy = 0.0
    
    unique_labels = sorted(list(set(true_labels + [p for p in predicted_labels if p != "Unknown"])))
    
    # Classification report
    try:
        report = classification_report(
            true_labels, predicted_labels,
            labels=unique_labels,
            zero_division=0,
            output_dict=True
        )
    except:
        report = {}
    
    # Print results
    print(f"\n[INFO] {model_id} evaluation completed in {elapsed/60:.1f} minutes")
    print("\n" + "=" * 80)
    print(f"EVALUATION RESULTS: {model_id}")
    print("=" * 80)
    correct = sum(1 for r in detailed_results if r['correct'])
    print(f"Model: {model_id} ({model_type})")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(detailed_results)})")
    print(f"Evaluation time: {elapsed/60:.1f} minutes")
    
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
        "model_name": model_id,
        "model_path": model_name,
        "adapter_path": model_path if model_path else None,
        "model_type": model_type,
        "test_data": test_data_path,
        "total_samples": len(detailed_results),
        "correct_predictions": correct,
        "accuracy": accuracy,
        "evaluation_time_seconds": elapsed,
        "avg_time_per_sample": elapsed/len(detailed_results) if detailed_results else 0,
        "label_distribution": dict(label_counts),
        "confusion_patterns": confusion,
        "classification_report": report,
        "unique_labels": unique_labels
    }
    
    with open(SUMMARY_FILE, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Update final progress
    progress = {
        "model_name": model_id,
        "model_path": model_name,
        "adapter_path": model_path if model_path else None,
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
        print("Usage:")
        print("  python eval_llama.py baseline              # Evaluate baseline Llama 8B")
        print("  python eval_llama.py finetuned <adapter>  # Evaluate fine-tuned Llama 8B")
        return
    
    mode = sys.argv[1]
    
    if mode == "baseline":
        print("Evaluating baseline Llama 8B model...")
        # Use local model path
        model_path = "models/llama_8b_baseline" if os.path.exists("models/llama_8b_baseline") else "meta-llama/Meta-Llama-3.1-8B-Instruct"
        accuracy, summary = evaluate_llama_model(
            model_name=model_path,
            is_finetuned=False
        )
    elif mode == "finetuned":
        adapter_path = sys.argv[2] if len(sys.argv) > 2 else "models/llama_8b_finetuned"
        # Use local baseline model
        base_model = "models/llama_8b_baseline" if os.path.exists("models/llama_8b_baseline") else "meta-llama/Meta-Llama-3.1-8B-Instruct"
        print(f"Evaluating fine-tuned Llama 8B model with adapter: {adapter_path}")
        accuracy, summary = evaluate_llama_model(
            model_name=base_model,
            model_path=adapter_path,
            is_finetuned=True
        )
    else:
        print(f"Unknown mode: {mode}")
        return
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Final Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()