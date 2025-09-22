import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import time
from datetime import datetime
import gc

def load_model(model_path="models/balanced_model", base_model_path="models/dclb"):
    """Load model once with maximum settings"""
    print(f"Loading model from {model_path}...")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    
    # Check if this is a LoRA model
    is_lora = (Path(model_path) / "adapter_config.json").exists()
    
    if is_lora:
        print(f"Loading base model: {base_model_path}")
        
        # Configure for maximum GPU utilization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        # Set maximum memory fraction
        torch.cuda.set_per_process_memory_fraction(0.9, 0)
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            max_memory={0: "22GB"}
        )
        
        print(f"Loading LoRA adapter from {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        print("Merging LoRA weights...")
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
            max_memory={0: "22GB"}
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path if is_lora else model_path,
        trust_remote_code=True,
        padding_side="left"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set model to evaluation mode
    model.eval()
    
    # Disable cache to avoid issues
    model.config.use_cache = False
    
    print("Model loaded successfully!")
    return model, tokenizer

def evaluate_batch(model, tokenizer, test_samples, batch_size=4):
    """Evaluate samples in batches for efficiency"""
    results = []
    correct = 0
    total = 0
    
    confusion_matrix = {
        "ComplexMethod": {"ComplexMethod": 0, "ComplexConditional": 0, "FeatureEnvy": 0},
        "ComplexConditional": {"ComplexMethod": 0, "ComplexConditional": 0, "FeatureEnvy": 0},
        "FeatureEnvy": {"ComplexMethod": 0, "ComplexConditional": 0, "FeatureEnvy": 0}
    }
    
    for i in range(0, len(test_samples), batch_size):
        batch = test_samples[i:i+batch_size]
        batch_prompts = []
        true_labels = []
        
        for sample in batch:
            code = sample.get('text', sample.get('code', ''))
            true_label = sample['label']
            true_labels.append(true_label)
            
            prompt = f"""You are a code smell detector. Analyze the following C# code and classify it.
Respond with ONLY one of these labels: ComplexMethod, ComplexConditional, FeatureEnvy

Code:
{code}

Label:"""
            batch_prompts.append(prompt)
        
        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            max_length=2048,  # Use reasonable context size
            truncation=True,
            padding=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate predictions
        with torch.no_grad():
            try:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            except Exception as e:
                print(f"Generation error: {e}")
                continue
        
        # Process outputs
        for j, output in enumerate(outputs):
            response = tokenizer.decode(output[inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
            
            # Extract label
            response_lower = response.lower()
            if "complexmethod" in response_lower or "complex method" in response_lower:
                predicted = "ComplexMethod"
            elif "complexconditional" in response_lower or "complex conditional" in response_lower:
                predicted = "ComplexConditional"
            elif "featureenvy" in response_lower or "feature envy" in response_lower:
                predicted = "FeatureEnvy"
            else:
                # Try to extract first word as label
                predicted = response.split()[0] if response else "Unknown"
            
            true = true_labels[j]
            
            if predicted == true:
                correct += 1
            total += 1
            
            if true in confusion_matrix and predicted in confusion_matrix[true]:
                confusion_matrix[true][predicted] += 1
            
            results.append({
                "index": i + j,
                "true_label": true,
                "predicted_label": predicted,
                "correct": predicted == true
            })
            
            print(f"Sample {i+j+1}/{len(test_samples)}: True={true}, Predicted={predicted}, Correct={predicted==true}")
        
        # Clear GPU cache periodically
        if (i + batch_size) % 20 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            accuracy = (correct / total * 100) if total > 0 else 0
            print(f"\nProgress: {total}/{len(test_samples)}, Current Accuracy: {accuracy:.2f}%\n")
    
    return results, correct, total, confusion_matrix

def main():
    # Load test data
    test_data_path = "dataset/test/ds.jsonl"
    print(f"Loading test data from {test_data_path}")
    test_samples = []
    with open(test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_samples.append(json.loads(line))
    
    print(f"Loaded {len(test_samples)} test samples")
    
    # Load model once
    model, tokenizer = load_model()
    
    # Setup output directory
    output_dir = Path("evaluation_results") / "balanced_model_fast" / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nStarting batch evaluation...")
    print("=" * 80)
    
    start_time = time.time()
    
    # Evaluate in batches
    results, correct, total, confusion_matrix = evaluate_batch(model, tokenizer, test_samples, batch_size=4)
    
    # Calculate metrics
    accuracy = (correct / total * 100) if total > 0 else 0
    
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
    
    elapsed_time = time.time() - start_time
    
    # Save results
    final_results = {
        "model_path": "models/balanced_model",
        "test_data": test_data_path,
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(test_samples),
        "correct_predictions": correct,
        "accuracy": accuracy,
        "elapsed_time_seconds": elapsed_time,
        "confusion_matrix": confusion_matrix,
        "class_metrics": class_metrics,
        "predictions": results
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Model: balanced_model (fine-tuned on 5,100 balanced samples)")
    print(f"Test samples: {len(test_samples)}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct}/{total}")
    print(f"Time: {elapsed_time:.2f} seconds")
    print(f"\nConfusion Matrix:")
    print(f"{'True/Predicted':<20} {'ComplexMethod':<15} {'ComplexConditional':<20} {'FeatureEnvy':<15}")
    for true_label in confusion_matrix:
        row = confusion_matrix[true_label]
        print(f"{true_label:<20} {row['ComplexMethod']:<15} {row['ComplexConditional']:<20} {row['FeatureEnvy']:<15}")
    print(f"\nPer-class metrics:")
    for label, metrics in class_metrics.items():
        print(f"  {label}:")
        print(f"    Precision: {metrics['precision']:.3f}")
        print(f"    Recall: {metrics['recall']:.3f}")
        print(f"    F1: {metrics['f1']:.3f}")
        print(f"    Support: {metrics['support']}")
    print(f"\nResults saved to: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()