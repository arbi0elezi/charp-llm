import json
import os
import random
from collections import defaultdict

INPUT_FILE = os.path.abspath("dataset/train/train_positive.jsonl")
OUTPUT_FILE = os.path.abspath("dataset/train/train_random_balanced.jsonl")
MAX_SAMPLES_PER_LABEL = 2000

def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

def balance_dataset(data, label_key="label", max_samples=2000):
    label_to_entries = defaultdict(list)

    # Group entries by label
    for entry in data:
        label = entry[label_key]
        label_to_entries[label].append(entry)

    balanced_data = []

    for label, entries in label_to_entries.items():
        if len(entries) >= max_samples:
            sampled = random.sample(entries, max_samples)
        else:
            # Use all entries and fill the rest with random duplicates
            sampled = entries.copy()
            fill = random.choices(entries, k=max_samples - len(entries))
            sampled.extend(fill)
        balanced_data.extend(sampled)

    random.shuffle(balanced_data)
    return balanced_data

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    data = load_jsonl(INPUT_FILE)
    balanced_data = balance_dataset(data, label_key="label", max_samples=MAX_SAMPLES_PER_LABEL)
    save_jsonl(balanced_data, OUTPUT_FILE)
    print(f"Balanced dataset with {MAX_SAMPLES_PER_LABEL} per label saved to '{OUTPUT_FILE}'")
