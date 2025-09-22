import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import time
from datetime import datetime
import gc
import os

def clear_gpu_memory():
    """Clear GPU memory between test cases"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def load_model_fresh(model_path, base_model_path=None):
    """Load model with maximum context window and optimized settings"""
    print(f"Loading model from {model_path}...")
    
    # Clear any existing GPU memory
    clear_gpu_memory()
    
    # Determine if this is a LoRA model
    is_lora = (Path(model_path) / "adapter_config.json").exists()
    
    if is_lora:
        if not base_model_path:
            # Try to read base model from adapter config
            config_path = Path(model_path) / "adapter_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    base_model_path = config.get("base_model_name_or_path", "models/dclb")
        
        print(f"Loading base model: {base_model_path}")
        
        # Configure for maximum GPU utilization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        # Set maximum memory fraction for GPU
        torch.cuda.set_per_process_memory_fraction(0.95, 0)
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",  # Flash attention not available on Windows
            max_memory={0: "23GB"}  # Use almost all of RTX A5000's 24GB
        )
        
        print(f"Loading LoRA adapter from {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
    else:
        # Load standard model with maximum settings
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",  # Flash attention not available on Windows
            max_memory={0: "23GB"}
        )
    
    # Load tokenizer with maximum context
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path if is_lora else model_path,
        trust_remote_code=True,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set model to evaluation mode
    model.eval()
    model.config.use_cache = False  # Disable cache to save memory
    
    # Additional cache handling for DeepSeek models
    if hasattr(model.config, 'past_key_values'):
        model.config.past_key_values = None
    
    # Maximize context window - DeepSeek supports up to 16K
    max_context = min(16384, model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else 4096)
    print(f"Using maximum context window: {max_context} tokens")
    
    return model, tokenizer, max_context

def evaluate_single_case(code, model, tokenizer, max_context):
    """Evaluate a single test case with maximum context"""
    prompt = f"""You are a code smell detector. Analyze the following C# code and classify it.
Respond with ONLY one of these labels: ComplexMethod, ComplexConditional, FeatureEnvy

Code:
{code}

Label:"""
    
    # Tokenize with maximum length
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=max_context - 100,  # Leave room for generation
        truncation=True,
        padding=True
    )
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate with optimized settings
    with torch.no_grad():
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False  # Explicitly disable cache
            )
        except Exception as e:
            # Fallback generation without cache
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
    
    # Extract label
    response_lower = response.lower()
    if "complexmethod" in response_lower or "complex method" in response_lower:
        return "ComplexMethod"
    elif "complexconditional" in response_lower or "complex conditional" in response_lower:
        return "ComplexConditional"
    elif "featureenvy" in response_lower or "feature envy" in response_lower:
        return "FeatureEnvy"
    else:
        return response.split()[0] if response else "Unknown"

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/balanced_model", 
                        help="Path to model (LoRA or full)")
    parser.add_argument("--base_model_path", type=str, default=None,
                        help="Base model path for LoRA (optional)")
    parser.add_argument("--test_data", type=str, default="dataset/test/ds.jsonl",
                        help="Path to test data")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size (keep at 1 for session restart)")
    args = parser.parse_args()
    
    # Load test data
    print(f"Loading test data from {args.test_data}")
    test_samples = []
    with open(args.test_data, 'r', encoding='utf-8') as f:
        for line in f:
            test_samples.append(json.loads(line))
    
    print(f"Loaded {len(test_samples)} test samples")
    
    # Setup output directory
    model_name = Path(args.model_path).name
    output_dir = Path(args.output_dir) / model_name / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluation metrics
    correct = 0
    total = 0
    predictions = []
    label_counts = {"ComplexMethod": 0, "ComplexConditional": 0, "FeatureEnvy": 0}
    confusion_matrix = {
        "ComplexMethod": {"ComplexMethod": 0, "ComplexConditional": 0, "FeatureEnvy": 0},
        "ComplexConditional": {"ComplexMethod": 0, "ComplexConditional": 0, "FeatureEnvy": 0},
        "FeatureEnvy": {"ComplexMethod": 0, "ComplexConditional": 0, "FeatureEnvy": 0}
    }
    
    print("\nStarting evaluation with session restart for each test case...")
    print("=" * 80)
    
    start_time = time.time()
    
    for idx, sample in enumerate(test_samples):
        print(f"\nTest case {idx+1}/{len(test_samples)}")
        
        # Load fresh model for each test case (ensures clean session)
        model, tokenizer, max_context = load_model_fresh(args.model_path, args.base_model_path)
        
        code = sample.get('code', sample.get('text', ''))  # Handle both 'code' and 'text' fields
        true_label = sample['label']
        
        # Evaluate
        try:
            predicted_label = evaluate_single_case(code, model, tokenizer, max_context)
            
            # Update metrics
            if predicted_label == true_label:
                correct += 1
            total += 1
            
            if true_label in confusion_matrix and predicted_label in confusion_matrix[true_label]:
                confusion_matrix[true_label][predicted_label] += 1
            
            predictions.append({
                "index": idx,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "correct": predicted_label == true_label,
                "code_preview": code[:200] + "..." if len(code) > 200 else code
            })
            
            print(f"  True: {true_label}, Predicted: {predicted_label}, Correct: {predicted_label == true_label}")
            
            # Save incremental results every 10 samples
            if (idx + 1) % 10 == 0:
                accuracy = correct / total * 100
                print(f"\nProgress: {total}/{len(test_samples)}, Accuracy: {accuracy:.2f}%")
                
                # Save intermediate results
                results = {
                    "progress": f"{total}/{len(test_samples)}",
                    "current_accuracy": accuracy,
                    "predictions_so_far": predictions
                }
                with open(output_dir / "intermediate_results.json", 'w') as f:
                    json.dump(results, f, indent=2)
        
        except Exception as e:
            print(f"  Error processing sample {idx}: {e}")
            predictions.append({
                "index": idx,
                "error": str(e),
                "true_label": true_label
            })
        
        finally:
            # Clean up model and memory after each test case
            del model
            del tokenizer
            clear_gpu_memory()
    
    # Calculate final metrics
    accuracy = correct / total * 100 if total > 0 else 0
    
    # Calculate per-class metrics
    class_metrics = {}
    for label in ["ComplexMethod", "ComplexConditional", "FeatureEnvy"]:
        tp = confusion_matrix[label][label]
        fp = sum(confusion_matrix[other][label] for other in confusion_matrix if other != label)
        fn = sum(confusion_matrix[label][other] for other in confusion_matrix[label] if other != label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion_matrix[label].values())
        }
    
    # Save final results
    elapsed_time = time.time() - start_time
    final_results = {
        "model_path": args.model_path,
        "test_data": args.test_data,
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(test_samples),
        "correct_predictions": correct,
        "accuracy": accuracy,
        "elapsed_time_seconds": elapsed_time,
        "avg_time_per_sample": elapsed_time / len(test_samples) if test_samples else 0,
        "confusion_matrix": confusion_matrix,
        "class_metrics": class_metrics,
        "predictions": predictions
    }
    
    # Save to JSON
    with open(output_dir / "final_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Test samples: {len(test_samples)}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct}/{total}")
    print(f"Time: {elapsed_time:.2f} seconds")
    print(f"\nPer-class metrics:")
    for label, metrics in class_metrics.items():
        print(f"  {label}:")
        print(f"    Precision: {metrics['precision']:.3f}")
        print(f"    Recall: {metrics['recall']:.3f}")
        print(f"    F1: {metrics['f1']:.3f}")
    print(f"\nResults saved to: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()