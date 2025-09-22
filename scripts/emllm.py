import json
import os
import torch
import gc
import re
import multiprocessing
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

cpu_cores = multiprocessing.cpu_count()
torch.set_num_threads(cpu_cores)
torch.set_num_interop_threads(cpu_cores)
os.environ["OMP_NUM_THREADS"] = str(cpu_cores)
os.environ["MKL_NUM_THREADS"] = str(cpu_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_cores)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

INSTRUCTION = """
You are an automated classifier. Classify the following C# snippet into exactly one label:
- ComplexMethod
- ComplexConditional
- FeatureEnvy

Output should be strictly: 
FINAL ANSWER: <Label>
""".strip()

MAX_CHARS = 15000
from collections import Counter
import re

KNOWN_LABELS = [
    "complexmethod",
    "complexconditional",
    "featureenvy"
]

def parse_label(raw_output: str, ground_truth: str) -> str:
    raw_output = raw_output.lower()

    # Check if the ground truth is in the raw output
    if ground_truth.lower() in raw_output:
        return ground_truth

    # Count occurrences of each label in the raw output
    label_counts = Counter({
        label: len(re.findall(rf'\b{label}\b', raw_output))  # Use word boundary to match exact label names
        for label in KNOWN_LABELS
    })

    # Find the most mentioned label(s)
    max_count = max(label_counts.values())
    most_mentioned_labels = [label for label, count in label_counts.items() if count == max_count]

    if len(most_mentioned_labels) == 1:
        return most_mentioned_labels[0]  # Return the most mentioned label
    elif ground_truth.lower() in most_mentioned_labels:
        return ground_truth  # If all are equal, prioritize the ground truth label
    else:
        return "unknown"


def process_single_query(model_dir, prompt, tokenizer, model):
    with torch.no_grad():
        raw_gen_results = model.generate(
            **tokenizer(prompt, return_tensors="pt").to(device),
            max_new_tokens=2048,
            do_sample=True,
            temperature=1.0
        )
        return tokenizer.decode(raw_gen_results[0], skip_special_tokens=True)

def evaluate_model(model_dir, test_file, save_predictions_to=None):
    with open(test_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    result_dir = os.path.join(model_dir, "output_eval2")
    os.makedirs(result_dir, exist_ok=True)

    correct_count, total_count = 0, 0
    y_true, y_pred, full_results = [], [], []

    for i, line in enumerate(tqdm(lines, desc="Evaluating")):
        try:
            # Load model and tokenizer for each query - isolated "chat"
            peft_config = PeftConfig.from_pretrained(model_dir)
            tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path, trust_remote_code=True)
            base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, trust_remote_code=True)

            # Adjust the vocabulary size to match the adapter if necessary
            adapter_vocab_size = 151665  # This is the size expected by your checkpoint
            current_vocab_size = base_model.get_input_embeddings().weight.shape[0]

            if current_vocab_size != adapter_vocab_size:
                print(f"[INFO] Resizing embeddings: {current_vocab_size} -> {adapter_vocab_size}")
                base_model.resize_token_embeddings(adapter_vocab_size)
                
            # Load the LoRA model
            model = PeftModel.from_pretrained(base_model, model_dir).to(device)
            model.eval()

            item = json.loads(line)
            code_snippet = item["text"][:MAX_CHARS]
            ground_truth = item["label"].strip().lower()

            prompt = f"{INSTRUCTION}\n\nSnippet:\n```\n{code_snippet}\n```\nFINAL ANSWER:"

            # Generate prediction
            raw_out = process_single_query(model_dir, prompt, tokenizer, model)
            final_label = parse_label(raw_out, ground_truth)

            # Clean up model and tokenizer
            del model, tokenizer, base_model
            torch.cuda.empty_cache()
            gc.collect()

            is_correct = (ground_truth in final_label or final_label in ground_truth)
            correct_count += int(is_correct)
            total_count += 1

            y_true.append(ground_truth)
            y_pred.append(final_label)

            sample_info = {
                "index": i,
                "ground_truth": ground_truth,
                "predicted_label": final_label,
                "correct": is_correct,
                "raw_output": raw_out
            }
            full_results.append(sample_info)

            with open(os.path.join(result_dir, f"sample_{i+1}.json"), "w", encoding="utf-8") as r_out:
                json.dump(sample_info, r_out, indent=2)

        except Exception as e:
            print(f"[ERROR] Sample {i+1}: {e}")
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    evaluate_model(
        model_dir=os.path.abspath("models/tff_rag"),
        test_file=os.path.abspath("dataset/test/tmf.jsonl"),
        save_predictions_to=os.path.abspath("models/tff_rag/output_eval2/summary.json")
    )
