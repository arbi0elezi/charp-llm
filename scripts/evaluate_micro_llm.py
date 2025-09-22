import json
import os
import torch
import gc
import re
import multiprocessing
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

if not torch.cuda.is_available():
    raise RuntimeError("[ERROR] CUDA is not available. This script requires CUDA.")

cpu_cores = multiprocessing.cpu_count()
torch.set_num_threads(cpu_cores)
torch.set_num_interop_threads(cpu_cores)
os.environ["OMP_NUM_THREADS"] = str(cpu_cores)
os.environ["MKL_NUM_THREADS"] = str(cpu_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_cores)

device = torch.device("cuda")
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

def parse_label(raw_output: str) -> str:
    match = re.search(r"FINAL ANSWER:\s*([\w]+)", raw_output, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return "unknown"

def evaluate_model(model_dir, test_file, save_predictions_to=None):
    # Normalize path to use forward slashes to avoid backslash issues
    model_dir = os.path.normpath(os.path.abspath(model_dir)).replace("\\", "/")
    print(f"[INFO] Normalized model directory: {model_dir}")

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"[ERROR] Model directory not found: {model_dir}")

    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[ERROR] config.json not found in {model_dir}. Directory contents: {os.listdir(model_dir)}")

    print(f"[INFO] Loading baseline model from local directory: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
    
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quant_config,
        trust_remote_code=True,
        device_map="auto",
        local_files_only=True
    )
    
    # Resize embeddings if necessary
    current_vocab_size = model.get_input_embeddings().weight.shape[0]
    tokenizer_vocab_size = len(tokenizer)
    if current_vocab_size != tokenizer_vocab_size:
        print(f"[INFO] Resizing embeddings: {current_vocab_size} -> {tokenizer_vocab_size}")
        model.resize_token_embeddings(tokenizer_vocab_size)

    with open(test_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    result_dir = os.path.join(model_dir, "output_eval")
    os.makedirs(result_dir, exist_ok=True)

    correct_count, total_count = 0, 0
    y_true, y_pred, full_results = [], [], []
    unknown_count = 0

    for i, line in enumerate(tqdm(lines, desc="Evaluating")):
        try:
            item = json.loads(line)
            code_snippet = item["text"][:MAX_CHARS]
            ground_truth = item["label"].strip().lower()

            prompt = f"{INSTRUCTION}\n\nSnippet:\n```\n{code_snippet}\n```\nFINAL ANSWER:"

            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id
                )
                generated_tokens = outputs[0][len(inputs["input_ids"][0]):]
                raw_out = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            final_label = parse_label(raw_out)
            if final_label == "unknown":
                unknown_count += 1

            is_correct = (final_label == ground_truth)
            correct_count += int(is_correct)
            total_count += 1

            y_true.append(ground_truth)
            y_pred.append(final_label)

            sample_info = {
                "index": i,
                "ground_truth": ground_truth,
                "predicted_label": final_label,
                "correct": is_correct,
                "raw_output": raw_out,
                "full_generation": outputs.tolist()
            }
            full_results.append(sample_info)

            with open(os.path.join(result_dir, f"sample_{i+1}.json"), "w", encoding="utf-8") as r_out:
                json.dump(sample_info, r_out, indent=2)

            if i % 20 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"[ERROR] Sample {i+1}: {e}")

    acc = accuracy_score(y_true, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")

    print("\n==== EVALUATION RESULTS ====")
    print(f"Total Samples: {total_count}")
    print(f"Unknown Predictions: {unknown_count}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall: {r:.4f}")
    print(f"F1 Score: {f:.4f}")

    if save_predictions_to:
        with open(save_predictions_to, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2)
        print(f"[INFO] Results saved to: {save_predictions_to}")

if __name__ == "__main__":
    evaluate_model(
        model_dir=os.path.abspath("../models/dclb"),
        test_file=os.path.abspath("../dataset/test/tmf.jsonl"),
        save_predictions_to=os.path.abspath("../models/dclb/output_eval/summary.json")
    )