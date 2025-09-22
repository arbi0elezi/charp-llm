"""
Quick evaluation script for Llama 2 v2 fine-tuned model only
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

def evaluate_model(model_path, model_name, output_dir):
    """Evaluate the Llama 2 v2 fine-tuned model"""
    
    print("=" * 80)
    print(f"EVALUATING: {model_name}")
    print("=" * 80)
    print(f"Model path: {model_path}")
    print(f"Output directory: {output_dir}")
    
    # Load test data
    TEST_DATA_PATH = "dataset/test/ds.jsonl"
    print(f"\nLoading test data from: {TEST_DATA_PATH}")
    
    test_samples = []
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            test_samples.append(json.loads(line.strip()))
    
    print(f"Loaded {len(test_samples)} test samples")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load model
    print("\nLoading model with LoRA adapters...")
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Load base model
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
    model = model.merge_and_unload()
    model.eval()
    
    print("Model loaded successfully!")
    
    # Create output directory
    model_output_dir = Path(output_dir) / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluation loop
    predictions = []
    true_labels = []
    
    print(f"\nEvaluating {len(test_samples)} samples...")
    start_time = time.time()
    
    for i, sample in enumerate(test_samples):
        if i % 10 == 0:
            elapsed = time.time() - start_time
            if i > 0:
                avg_time = elapsed / i
                remaining = avg_time * (len(test_samples) - i)
                eta_str = f"ETA: {remaining:.1f}s"
            else:
                eta_str = "Calculating..."
            
            if len(predictions) > 0:
                current_acc = accuracy_score(true_labels[:len(predictions)], predictions)
                acc_str = f"Acc: {current_acc:.3f}"
            else:
                acc_str = "Acc: N/A"
            
            print(f"  [{i:3}/{len(test_samples)}] {i/len(test_samples)*100:5.1f}% | {acc_str} | {eta_str}")
        
        # Get code and true label
        code = sample.get('code', sample.get('text', ''))
        true_label = sample['label']
        
        # Create prompt using training format
        prompt = f"""### Task: Identify the code smell in this C# code.

### Code:
```csharp
{code[:800]}
```

### Answer:"""
        
        # Generate prediction
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode only the generated part
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Extract predicted label
            predicted_label = "Unknown"
            response_lower = response.lower()
            
            if "complexconditional" in response_lower:
                predicted_label = "ComplexConditional"
            elif "complexmethod" in response_lower:
                predicted_label = "ComplexMethod"
            elif "featureenvy" in response_lower:
                predicted_label = "FeatureEnvy"
            
        except Exception as e:
            print(f"  Error on sample {i}: {e}")
            response = f"ERROR: {str(e)}"
            predicted_label = "Unknown"
        
        predictions.append(predicted_label)
        true_labels.append(true_label)
    
    # Calculate final metrics
    elapsed = time.time() - start_time
    accuracy = accuracy_score(true_labels, predictions)
    
    print(f"\nEvaluation completed in {elapsed:.1f} seconds")
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Accuracy: {accuracy:.4f} ({sum(p == t for p, t in zip(predictions, true_labels))}/{len(test_samples)})")
    print(f"Time: {elapsed:.1f}s ({elapsed/len(test_samples):.2f}s per sample)")
    
    # Save results
    summary = {
        "model_name": model_name,
        "model_path": model_path,
        "accuracy": accuracy,
        "correct": sum(p == t for p, t in zip(predictions, true_labels)),
        "total": len(test_samples),
        "time_seconds": elapsed,
        "timestamp": datetime.now().isoformat()
    }
    
    summary_file = model_output_dir / "evaluation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {summary_file}")
    
    return accuracy

if __name__ == "__main__":
    evaluate_model(
        model_path="models/llama2_finetuned_v2",
        model_name="llama2_finetuned_v2",
        output_dir="evaluation_results"
    )