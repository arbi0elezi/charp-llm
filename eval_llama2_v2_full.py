"""
Full evaluation script for Llama 2 v2 fine-tuned model
"""

import sys
sys.path.append('scripts')
from eval_llama2_fixed import evaluate_model

print("=" * 80)
print("FULL EVALUATION: Llama 2 Fine-tuned v2")
print("=" * 80)

# Evaluate the new v2 model
accuracy, summary = evaluate_model(
    model_path="models/llama2_finetuned_v2",
    model_name="llama2_finetuned_v2_final",
    output_dir="evaluation_results",
    is_baseline=False
)

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
print(f"Final Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
print("Results saved to: evaluation_results/llama2_finetuned_v2_final/")
print("=" * 80)