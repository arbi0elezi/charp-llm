import json
import random
import sys
from collections import defaultdict

def trim_dataset(input_file, output_file, max_per_label=1000):
    """
    Reads a JSONL dataset with fields {text, label}, picks up to max_per_label
    random samples for each label, and writes a train_micro.jsonl output.
    """
    label_buckets = defaultdict(list)

    # 1) Read input file line by line
    with open(input_file, "r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            label = data.get("label", "UnknownLabel")
            label_buckets[label].append(data)

    # 2) Shuffle and trim each label
    for label, entries in label_buckets.items():
        random.shuffle(entries)
        if len(entries) > max_per_label:
            label_buckets[label] = entries[:max_per_label]
    
    # 3) Combine all trimmed lists into one
    combined = []
    for label_entries in label_buckets.values():
        combined.extend(label_entries)
    
    # Optional: shuffle entire dataset one more time
    random.shuffle(combined)

    # 4) Write to output file
    with open(output_file, "w", encoding="utf-8") as f_out:
        for item in combined:
            f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[INFO] Wrote {len(combined)} total samples to {output_file}")

if __name__ == "__main__":
    # Example usage:
    #   python trim_dataset.py train.jsonl train_micro.jsonl
    # You can pass a third argument to override max samples per label
    if len(sys.argv) < 3:
        print("Usage: python trim_dataset.py <input.jsonl> <output.jsonl> [max_per_label]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    max_label = 1000

    if len(sys.argv) > 3:
        max_label = int(sys.argv[3])
    
    trim_dataset(input_path, output_path, max_label)


if __name__ == "__main__":
    trim_dataset("../dataset/train/train.jsonl", "../dataset/train/train_micro.jsonl", 1000)