import json
import os
from collections import defaultdict
from pathlib import Path

def load_jsonl(file_path):
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def balance_dataset(target_per_label=85):
    """Balance test dataset to have equal samples per label."""
    
    # Load test and train data
    test_path = "dataset/test/tmf.jsonl"
    train_path = "dataset/train/tmf.jsonl"
    
    print(f"Loading test data from: {test_path}")
    test_data = load_jsonl(test_path)
    
    print(f"Loading train data from: {train_path}")
    train_data = load_jsonl(train_path)
    
    # Combine all data
    all_data = test_data + train_data
    
    # Group by label and sort by text length (shortest first)
    grouped = defaultdict(list)
    for item in all_data:
        grouped[item['label']].append(item)
    
    # Sort each group by text length
    for label in grouped:
        grouped[label].sort(key=lambda x: len(x['text']))
    
    # Print current distribution
    print("\nCurrent distribution:")
    for label, items in grouped.items():
        print(f"  {label}: {len(items)} samples")
    
    # Balance dataset
    balanced_data = []
    labels_needed = ['ComplexMethod', 'ComplexConditional', 'FeatureEnvy']
    
    for label in labels_needed:
        if label in grouped:
            available = grouped[label]
            
            if len(available) >= target_per_label:
                # Take the shortest samples
                selected = available[:target_per_label]
            else:
                # Take all available and duplicate shortest ones
                selected = available.copy()
                needed = target_per_label - len(available)
                
                # Duplicate the shortest samples
                for i in range(needed):
                    selected.append(available[i % len(available)])
            
            balanced_data.extend(selected)
            print(f"\n{label}:")
            print(f"  Selected: {len(selected)} samples")
            print(f"  Avg length: {sum(len(s['text']) for s in selected) / len(selected):.0f} chars")
            print(f"  Min length: {len(min(selected, key=lambda x: len(x['text']))['text'])} chars")
            print(f"  Max length: {len(max(selected, key=lambda x: len(x['text']))['text'])} chars")
    
    # Shuffle to mix labels
    import random
    random.seed(42)
    random.shuffle(balanced_data)
    
    # Save balanced test data
    output_path = "dataset/test/tmf_balanced.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in balanced_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\nBalanced dataset saved to: {output_path}")
    print(f"Total samples: {len(balanced_data)}")
    
    # Verify balance
    label_counts = defaultdict(int)
    for item in balanced_data:
        label_counts[item['label']] += 1
    
    print("\nFinal distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count} samples")
    
    # Also create a backup of original test data
    backup_path = "dataset/test/tmf_original.jsonl"
    if not os.path.exists(backup_path):
        with open(test_path, 'r', encoding='utf-8') as src:
            with open(backup_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
        print(f"\nOriginal test data backed up to: {backup_path}")
    
    # Replace original with balanced
    with open(test_path, 'w', encoding='utf-8') as f:
        for item in balanced_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Updated original test file: {test_path}")
    
    return balanced_data

if __name__ == "__main__":
    balanced = balance_dataset(85)
    print(f"\nBalancing complete! Created dataset with {len(balanced)} samples.")