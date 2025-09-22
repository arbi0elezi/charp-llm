import json
import random
from collections import defaultdict
from tqdm import tqdm

def split_dataset(input_path, train_path, test_path, seed=42):
    label_to_entries = defaultdict(list)
    print("[INFO] Reading data and grouping by label...")

    # Read all lines and group by label
    with open(input_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading JSONL"):
            try:
                obj = json.loads(line)
                label = obj["label"]
                label_to_entries[label].append(obj)
            except Exception as e:
                print(f"[WARN] Skipping malformed line: {e}")

    # Calculate 20% of the smallest label count
    label_counts = {label: len(entries) for label, entries in label_to_entries.items()}
    min_count = min(label_counts.values())
    test_size = int(min_count * 0.2)

    print(f"[INFO] Minimum label count: {min_count}")
    print(f"[INFO] Selecting {test_size} samples per label for testing")

    test_set = []
    train_set = []

    # Stratified balanced sampling
    for label, entries in label_to_entries.items():
        random.seed(seed)
        random.shuffle(entries)

        selected_for_test = entries[:test_size]
        selected_for_train = entries[test_size:]

        test_set.extend(selected_for_test)
        train_set.extend(selected_for_train)

        print(f"  - {label}: {len(selected_for_train)} train, {len(selected_for_test)} test")

    # Write to JSONL files
    print(f"[INFO] Writing training data to {train_path}")
    with open(train_path, "w", encoding="utf-8") as f:
        for item in train_set:
            f.write(json.dumps(item) + "\n")

    print(f"[INFO] Writing test data to {test_path}")
    with open(test_path, "w", encoding="utf-8") as f:
        for item in test_set:
            f.write(json.dumps(item) + "\n")

    print(f"[✅ DONE] Total: {len(train_set)} train, {len(test_set)} test")

# Example usage:
split_dataset("../dataset/train/train_positive.jsonl", "../dataset/train/train_c.jsonl", "../dataset/test/test_c.jsonl")
