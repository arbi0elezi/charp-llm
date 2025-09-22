import os
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
import warnings
warnings.filterwarnings('ignore')

# === Configuration ===
MODEL_BASE = os.path.abspath("models/dclb")
MODEL_FINETUNED = os.path.abspath("models/tff_rag")
TEST_DATA = os.path.abspath("dataset/test/tmf.jsonl")
RESULTS_DIR = os.path.abspath("evaluation_results")

# === Device Setup ===
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for evaluation")

device = torch.device("cuda")
print(f"[INFO] Using device: {device}")
print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# === Evaluation Template ===
EVAL_TEMPLATE = """You are a code smell classifier. Given the following C# snippet, classify it.

Code:
{text}

In your answer strictly follow this format
FINAL ANSWER: followed by the label, example: FINAL ANSWER: ComplexMethod"""

class ModelEvaluator:
    def __init__(self, model_path: str, base_model_path: str = None):
        """Initialize the evaluator with model paths."""
        self.model_path = model_path
        self.base_model_path = base_model_path or model_path
        self.tokenizer = None
        self.model = None
        self.results = {}
        
    def load_model(self):
        """Load the fine-tuned model."""
        print(f"\n[INFO] Loading model from: {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Quantization config for memory efficiency
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        
        # Check if this is a LoRA model by looking for adapter_config.json
        adapter_config_path = Path(self.model_path) / "adapter_config.json"
        
        if adapter_config_path.exists():
            print("[INFO] Detected LoRA adapter, loading base model + adapter")
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                quantization_config=quant_config,
                trust_remote_code=True,
                device_map="auto"
            )
            
            # First, load the tokenizer from the fine-tuned model to get vocab size
            base_tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            
            # Resize embeddings to match the fine-tuned model's vocabulary
            base_model.resize_token_embeddings(len(base_tokenizer))
            
            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            # Disable cache for compatibility
            self.model.config.use_cache = False
            print("[INFO] LoRA adapter loaded successfully")
        else:
            print("[INFO] Loading as standard model")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=quant_config,
                trust_remote_code=True,
                device_map="auto"
            )
        
        self.model.eval()
        print("[INFO] Model loaded and set to evaluation mode")
        
    def load_test_data(self) -> List[Dict]:
        """Load test dataset."""
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
        
        return test_samples
    
    def predict_single(self, text: str, max_length: int = 512) -> str:
        """Make prediction for a single sample."""
        # Format prompt
        prompt = EVAL_TEMPLATE.replace("{text}", text[:800])  # Limit text length
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        ).to(device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,  # Low temperature for more deterministic output
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=False  # Disable cache to avoid compatibility issues
            )
        
        # Decode output
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract prediction from response
        prediction = self.extract_label(response)
        return prediction
    
    def extract_label(self, response: str) -> str:
        """Extract the predicted label from model response."""
        # Look for "FINAL ANSWER:" pattern
        if "FINAL ANSWER:" in response:
            parts = response.split("FINAL ANSWER:")[-1].strip()
            # Extract the first word after FINAL ANSWER
            label = parts.split()[0] if parts else "Unknown"
            return label.strip()
        
        # Fallback: look for common labels
        common_labels = [
            "ComplexMethod", "FeatureEnvy", "MultifacetedAbstraction",
            "ComplexConditional", "LongParameterList", "DeadCode",
            "LongMethod", "LargeClass", "DataClass", "RefusedBequest"
        ]
        
        response_lower = response.lower()
        for label in common_labels:
            if label.lower() in response_lower:
                return label
        
        return "Unknown"
    
    def evaluate(self, test_samples: List[Dict], sample_size: int = None) -> Dict:
        """Evaluate model on test samples."""
        if sample_size:
            test_samples = test_samples[:sample_size]
            print(f"\n[INFO] Evaluating on {sample_size} samples")
        else:
            print(f"\n[INFO] Evaluating on all {len(test_samples)} samples")
        
        true_labels = []
        predicted_labels = []
        
        print("\n[INFO] Starting evaluation...")
        for i, sample in enumerate(test_samples):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(test_samples)} ({i/len(test_samples)*100:.1f}%)")
            
            true_label = sample['label']
            predicted_label = self.predict_single(sample['text'])
            
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)
            
            # Clear CUDA cache periodically
            if i % 50 == 0:
                torch.cuda.empty_cache()
        
        print(f"  Progress: {len(test_samples)}/{len(test_samples)} (100.0%)")
        
        # Calculate metrics
        self.results = self.calculate_metrics(true_labels, predicted_labels)
        return self.results
    
    def calculate_metrics(self, true_labels: List[str], predicted_labels: List[str]) -> Dict:
        """Calculate evaluation metrics."""
        print("\n[INFO] Calculating metrics...")
        
        # Get unique labels
        unique_labels = sorted(list(set(true_labels + predicted_labels)))
        
        # Basic accuracy
        accuracy = accuracy_score(true_labels, predicted_labels)
        
        # Precision, recall, F1 per class
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predicted_labels, 
            labels=unique_labels,
            average=None,
            zero_division=0
        )
        
        # Macro averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            true_labels, predicted_labels,
            average='macro',
            zero_division=0
        )
        
        # Weighted averages
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            true_labels, predicted_labels,
            average='weighted',
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=unique_labels)
        
        # Classification report
        report = classification_report(
            true_labels, predicted_labels,
            labels=unique_labels,
            zero_division=0
        )
        
        results = {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'per_class_metrics': {
                label: {
                    'precision': precision[i],
                    'recall': recall[i],
                    'f1': f1[i],
                    'support': int(support[i])
                }
                for i, label in enumerate(unique_labels)
            },
            'confusion_matrix': cm.tolist(),
            'labels': unique_labels,
            'classification_report': report,
            'total_samples': len(true_labels),
            'correct_predictions': sum(1 for t, p in zip(true_labels, predicted_labels) if t == p)
        }
        
        return results
    
    def save_results(self):
        """Save evaluation results to file."""
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        results_file = os.path.join(RESULTS_DIR, f"evaluation_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save summary report
        report_file = os.path.join(RESULTS_DIR, f"report_{timestamp}.txt")
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Model Path: {self.model_path}\n")
            f.write(f"Test Data: {TEST_DATA}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total Samples: {self.results['total_samples']}\n")
            f.write(f"Correct Predictions: {self.results['correct_predictions']}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("OVERALL METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Accuracy: {self.results['accuracy']:.4f}\n\n")
            
            f.write("Macro Averages:\n")
            f.write(f"  Precision: {self.results['macro_precision']:.4f}\n")
            f.write(f"  Recall: {self.results['macro_recall']:.4f}\n")
            f.write(f"  F1-Score: {self.results['macro_f1']:.4f}\n\n")
            
            f.write("Weighted Averages:\n")
            f.write(f"  Precision: {self.results['weighted_precision']:.4f}\n")
            f.write(f"  Recall: {self.results['weighted_recall']:.4f}\n")
            f.write(f"  F1-Score: {self.results['weighted_f1']:.4f}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("CLASSIFICATION REPORT\n")
            f.write("-" * 80 + "\n")
            f.write(self.results['classification_report'])
            f.write("\n")
            
            f.write("-" * 80 + "\n")
            f.write("PER-CLASS PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            
            for label, metrics in sorted(self.results['per_class_metrics'].items()):
                f.write(f"\n{label}:\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {metrics['f1']:.4f}\n")
                f.write(f"  Support: {metrics['support']}\n")
        
        print(f"\n[INFO] Results saved to:")
        print(f"  - JSON: {results_file}")
        print(f"  - Report: {report_file}")
        
        return results_file, report_file
    
    def print_summary(self):
        """Print evaluation summary to console."""
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Overall Accuracy: {self.results['accuracy']:.4f} ({self.results['correct_predictions']}/{self.results['total_samples']})")
        print(f"Macro F1-Score: {self.results['macro_f1']:.4f}")
        print(f"Weighted F1-Score: {self.results['weighted_f1']:.4f}")
        
        print("\n" + "-" * 80)
        print("TOP PERFORMING CLASSES (by F1-Score):")
        print("-" * 80)
        
        # Sort classes by F1 score
        class_performance = [
            (label, metrics['f1']) 
            for label, metrics in self.results['per_class_metrics'].items()
        ]
        class_performance.sort(key=lambda x: x[1], reverse=True)
        
        for i, (label, f1) in enumerate(class_performance[:5], 1):
            metrics = self.results['per_class_metrics'][label]
            print(f"{i}. {label:25} F1: {f1:.4f} (P: {metrics['precision']:.4f}, R: {metrics['recall']:.4f})")
        
        if len(class_performance) > 5:
            print("\n" + "-" * 80)
            print("LOWEST PERFORMING CLASSES (by F1-Score):")
            print("-" * 80)
            
            for i, (label, f1) in enumerate(class_performance[-3:], 1):
                metrics = self.results['per_class_metrics'][label]
                print(f"{i}. {label:25} F1: {f1:.4f} (P: {metrics['precision']:.4f}, R: {metrics['recall']:.4f})")

def main():
    """Main evaluation function."""
    print("=" * 80)
    print("C# CODE SMELL DETECTION MODEL EVALUATION")
    print("=" * 80)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=MODEL_FINETUNED,
        base_model_path=MODEL_BASE
    )
    
    # Load model
    evaluator.load_model()
    
    # Load test data
    test_samples = evaluator.load_test_data()
    
    # Run evaluation (start with small sample for testing)
    print("\n" + "=" * 80)
    print("STARTING EVALUATION")
    print("=" * 80)
    
    # First, test with small sample
    print("\n[PHASE 1] Quick test with 20 samples...")
    quick_results = evaluator.evaluate(test_samples, sample_size=20)
    print(f"Quick test accuracy: {quick_results['accuracy']:.4f}")
    
    # Ask user if they want full evaluation
    print("\n" + "-" * 80)
    user_input = input("Run full evaluation on all test samples? (y/n): ").strip().lower()
    
    if user_input == 'y':
        print("\n[PHASE 2] Full evaluation on all samples...")
        full_results = evaluator.evaluate(test_samples)
        
        # Save results
        evaluator.save_results()
        
        # Print summary
        evaluator.print_summary()
    else:
        print("\n[INFO] Saving quick test results only...")
        evaluator.save_results()
        evaluator.print_summary()
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()