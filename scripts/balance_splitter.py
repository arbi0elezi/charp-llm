import json
import random
from collections import defaultdict
from tqdm import tqdm

def stratified_split(input_path, train_path, test_path, test_ratio=0.2, seed=42):
    label_to_entries = defaultdict(list)
    print("[INFO] Reading and grouping data by label...")

    # Step 1: Read and group by label
    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading JSONL"):
            try:
                obj = json.loads(line)
                label_to_entries[obj["label"]].append(obj)
            except Exception as e:
                print(f"[WARN] Skipping malformed line: {e}")

    train_set, test_set = [], []

    # Step 2: For each label, do 80/20 split
    for label, entries in label_to_entries.items():
        random.seed(seed)
        random.shuffle(entries)

        split_idx = int(len(entries) * (1 - test_ratio))
        train_entries = entries[:split_idx]
        test_entries = entries[split_idx:]

        train_set.extend(train_entries)
        test_set.extend(test_entries)

        print(f"  - {label}: {len(train_entries)} train, {len(test_entries)} test")

    # Step 3: Write output files
    print(f"[INFO] Writing training set to {train_path}")
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_set:
            f.write(json.dumps(item) + "\n")

    print(f"[INFO] Writing test set to {test_path}")
    with open(test_path, "w", encoding="utf-8") as f:
        for item in test_set:
            f.write(json.dumps(item) + "\n")

    print(f"[✅ DONE] Total: {len(train_set)} train, {len(test_set)} test")

# Example usage:
stratified_split("processed_data.jsonl", "train_b.jsonl", "test_b.jsonl")
