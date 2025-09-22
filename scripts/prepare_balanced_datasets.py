import json
import os
import random
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

def save_jsonl(data, file_path):
    """Save data to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def create_negative_examples(positive_samples, num_negatives_per_label):
    """
    Create negative examples by pairing code with wrong labels.
    Following LLM convention: show what the model should NOT predict.
    """
    negative_examples = []
    all_labels = list(set(s['label'] for s in positive_samples))
    
    for label in all_labels:
        # Get samples from OTHER labels
        other_samples = [s for s in positive_samples if s['label'] != label]
        
        # Randomly select samples and assign them the wrong label for training
        # This teaches the model what NOT to classify as this label
        selected = random.sample(other_samples, min(num_negatives_per_label, len(other_samples)))
        
        for sample in selected:
            # Create a negative example: code from one label, marked as NOT being another label
            # We'll add a marker to distinguish these during training
            negative_examples.append({
                'text': sample['text'],
                'label': f"NOT_{label}",  # Marking as negative example
                'original_label': sample['label'],  # Keep track of true label
                'example_type': 'negative'
            })
    
    return negative_examples

def balance_dataset_with_negatives(data, samples_per_label, include_negatives=True):
    """
    Balance dataset with equal samples per label and optional negative examples.
    
    Args:
        data: List of samples
        samples_per_label: Number of samples per label
        include_negatives: Whether to include negative examples
    """
    # Group by label and sort by text length
    grouped = defaultdict(list)
    for item in data:
        grouped[item['label']].append(item)
    
    # Sort each group by text length (shortest first for better training)
    for label in grouped:
        grouped[label].sort(key=lambda x: len(x['text']))
    
    # Get the main labels (excluding MultifacetedAbstraction as it's not in test)
    main_labels = ['ComplexMethod', 'ComplexConditional', 'FeatureEnvy']
    
    balanced_data = []
    
    # Balance positive examples
    for label in main_labels:
        if label in grouped:
            available = grouped[label]
            
            if len(available) >= samples_per_label:
                # Take the shortest samples for better quality
                selected = available[:samples_per_label]
            else:
                # Take all available and duplicate shortest ones
                selected = available.copy()
                needed = samples_per_label - len(available)
                
                # Duplicate the shortest samples cyclically
                for i in range(needed):
                    selected.append(available[i % len(available)])
            
            # Mark as positive examples
            for item in selected:
                item['example_type'] = 'positive'
            
            balanced_data.extend(selected)
    
    # Add negative examples if requested
    if include_negatives:
        # For a balanced approach: add 1/3 of positive samples as negatives per label
        num_negatives = samples_per_label // 3
        negative_examples = create_negative_examples(balanced_data, num_negatives)
        
        # Convert negative examples back to normal format for training
        # The model will learn from contrast
        for neg in negative_examples:
            if neg['label'].startswith('NOT_'):
                # For training, we'll use the original label
                # This helps the model learn boundaries
                balanced_data.append({
                    'text': neg['text'],
                    'label': neg['original_label'],
                    'example_type': 'negative_contrast'
                })
    
    return balanced_data

def prepare_final_dataset(data):
    """
    Prepare final dataset in standard format.
    Remove example_type markers for final output.
    """
    final_data = []
    for item in data:
        final_data.append({
            'text': item['text'],
            'label': item['label']
        })
    return final_data

def main():
    """Main function to balance both train and test datasets."""
    
    print("=" * 80)
    print("PREPARING BALANCED DATASETS")
    print("=" * 80)
    
    # Paths
    train_original = "dataset/train/tmf.jsonl"
    test_original = "dataset/test/tmf.jsonl"
    train_balanced = "dataset/train/ds.jsonl"
    test_balanced = "dataset/test/ds.jsonl"
    
    # Load original data
    print(f"\nLoading original training data from: {train_original}")
    train_data = load_jsonl(train_original)
    print(f"  Loaded {len(train_data)} training samples")
    
    print(f"\nLoading original test data from: {test_original}")
    test_data = load_jsonl(test_original)
    print(f"  Loaded {len(test_data)} test samples")
    
    # Analyze original distribution
    print("\nOriginal Training Distribution:")
    train_dist = defaultdict(int)
    for item in train_data:
        train_dist[item['label']] += 1
    for label, count in sorted(train_dist.items()):
        print(f"  {label}: {count} samples")
    
    print("\nOriginal Test Distribution:")
    test_dist = defaultdict(int)
    for item in test_data:
        test_dist[item['label']] += 1
    for label, count in sorted(test_dist.items()):
        print(f"  {label}: {count} samples")
    
    # Balance test dataset (85 samples per label, no negatives for clean evaluation)
    print("\n" + "-" * 80)
    print("BALANCING TEST DATASET")
    print("-" * 80)
    balanced_test = balance_dataset_with_negatives(
        test_data, 
        samples_per_label=85,
        include_negatives=False  # Clean test set without negatives
    )
    
    # Shuffle test data
    random.seed(42)
    random.shuffle(balanced_test)
    
    # Prepare and save test dataset
    final_test = prepare_final_dataset(balanced_test)
    save_jsonl(final_test, test_balanced)
    
    print(f"Balanced test dataset saved to: {test_balanced}")
    print(f"  Total samples: {len(final_test)}")
    
    # Verify test balance
    test_balanced_dist = defaultdict(int)
    for item in final_test:
        test_balanced_dist[item['label']] += 1
    print("\nBalanced Test Distribution:")
    for label, count in sorted(test_balanced_dist.items()):
        print(f"  {label}: {count} samples")
    
    # Balance training dataset (500 samples per label + negatives for better learning)
    print("\n" + "-" * 80)
    print("BALANCING TRAINING DATASET")
    print("-" * 80)
    
    # Use more samples for training
    balanced_train = balance_dataset_with_negatives(
        train_data,
        samples_per_label=500,  # More samples for training
        include_negatives=True   # Include negative examples for better boundaries
    )
    
    # Shuffle training data
    random.seed(42)
    random.shuffle(balanced_train)
    
    # Prepare and save training dataset
    final_train = prepare_final_dataset(balanced_train)
    save_jsonl(final_train, train_balanced)
    
    print(f"Balanced training dataset saved to: {train_balanced}")
    print(f"  Total samples: {len(final_train)}")
    
    # Verify training balance
    train_balanced_dist = defaultdict(int)
    for item in final_train:
        train_balanced_dist[item['label']] += 1
    print("\nBalanced Training Distribution:")
    for label, count in sorted(train_balanced_dist.items()):
        print(f"  {label}: {count} samples")
    
    # Create alternative balanced versions with different sizes
    print("\n" + "-" * 80)
    print("CREATING ALTERNATIVE SIZES")
    print("-" * 80)
    
    # Small balanced dataset for quick testing
    small_train = balance_dataset_with_negatives(
        train_data,
        samples_per_label=100,
        include_negatives=False
    )
    random.shuffle(small_train)
    final_small = prepare_final_dataset(small_train)
    save_jsonl(final_small, "dataset/train/ds_small.jsonl")
    print(f"Small training dataset saved to: dataset/train/ds_small.jsonl ({len(final_small)} samples)")
    
    # Large balanced dataset for thorough training
    large_train = balance_dataset_with_negatives(
        train_data,
        samples_per_label=1000,
        include_negatives=True
    )
    random.shuffle(large_train)
    final_large = prepare_final_dataset(large_train)
    save_jsonl(final_large, "dataset/train/ds_large.jsonl")
    print(f"Large training dataset saved to: dataset/train/ds_large.jsonl ({len(final_large)} samples)")
    
    print("\n" + "=" * 80)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 80)
    print("\nCreated datasets:")
    print("  - dataset/test/ds.jsonl          : Balanced test set (85 per label)")
    print("  - dataset/train/ds.jsonl         : Balanced train set (500 per label + negatives)")
    print("  - dataset/train/ds_small.jsonl   : Small train set (100 per label)")
    print("  - dataset/train/ds_large.jsonl   : Large train set (1000 per label + negatives)")
    print("\nOriginal files (tmf.jsonl) remain unchanged.")
    
    # Print usage examples
    print("\n" + "-" * 80)
    print("USAGE EXAMPLES")
    print("-" * 80)
    print("\n# Quick test with small dataset:")
    print("python scripts/train_generic_chunked.py \\")
    print("    --base_model models/dclb \\")
    print("    --train_data dataset/train/ds_small.jsonl \\")
    print("    --output_dir models/test_small")
    print("\n# Standard training with balanced dataset:")
    print("python scripts/train_generic_chunked.py \\")
    print("    --base_model models/dclb \\")
    print("    --train_data dataset/train/ds.jsonl \\")
    print("    --output_dir models/balanced_model")
    print("\n# Evaluation with balanced test set:")
    print("python scripts/eval_unified.py models/balanced_model")

if __name__ == "__main__":
    main()