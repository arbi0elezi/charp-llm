import json
import random
from collections import defaultdict, Counter

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

def create_large_balanced_dataset(train_data, samples_per_label=1700):
    """
    Create a large balanced dataset with specified samples per label.
    Will duplicate samples if needed to reach the target.
    
    Args:
        train_data: Original training data
        samples_per_label: Target samples per label (1700 * 3 = 5100 total)
    """
    # Group by label
    grouped = defaultdict(list)
    for item in train_data:
        if item['label'] in ['ComplexMethod', 'ComplexConditional', 'FeatureEnvy']:
            grouped[item['label']].append(item)
    
    # Sort each group by text length (shortest first for better quality)
    for label in grouped:
        grouped[label].sort(key=lambda x: len(x['text']))
    
    print("\nAvailable samples per label:")
    for label in ['ComplexMethod', 'ComplexConditional', 'FeatureEnvy']:
        if label in grouped:
            print(f"  {label}: {len(grouped[label])} samples")
    
    # Create balanced dataset
    balanced_data = []
    
    for label in ['ComplexMethod', 'ComplexConditional', 'FeatureEnvy']:
        if label in grouped:
            available = grouped[label]
            selected = []
            
            # First, add all available samples
            selected.extend(available)
            
            # If we need more, duplicate samples strategically
            if len(selected) < samples_per_label:
                needed = samples_per_label - len(selected)
                print(f"\n{label}: Need to add {needed} duplicates")
                
                # Duplicate samples, starting with shortest (best quality)
                # Use modulo to cycle through all samples
                for i in range(needed):
                    # Vary the selection to get diversity
                    idx = i % len(available)
                    selected.append(available[idx].copy())
            
            # Take exactly the number needed
            selected = selected[:samples_per_label]
            balanced_data.extend(selected)
            
            print(f"  Final count for {label}: {len(selected)}")
    
    return balanced_data

def add_augmented_samples(data, target_per_label=1700):
    """
    Add augmented samples by creating slight variations.
    This helps reach the target while maintaining diversity.
    """
    augmented = []
    
    for item in data:
        # Original sample
        augmented.append(item)
        
        # Create a variation by adding a comment
        augmented_item = item.copy()
        augmented_item['text'] = f"// Code sample for analysis\n{item['text']}"
        augmented_item['augmented'] = True
        augmented.append(augmented_item)
    
    return augmented

def main():
    print("=" * 80)
    print("CREATING LARGE BALANCED DATASET (5000+ samples)")
    print("=" * 80)
    
    # Load training data
    train_path = "dataset/train/tmf.jsonl"
    print(f"\nLoading training data from: {train_path}")
    train_data = load_jsonl(train_path)
    print(f"Total training samples loaded: {len(train_data)}")
    
    # Analyze distribution
    dist = Counter(s['label'] for s in train_data)
    print("\nOriginal distribution:")
    for label, count in sorted(dist.items()):
        print(f"  {label}: {count} samples")
    
    # Create large balanced dataset
    # 1700 samples per label * 3 labels = 5100 total samples
    samples_per_label = 1700
    
    print(f"\nCreating balanced dataset with {samples_per_label} samples per label...")
    print(f"Target total: {samples_per_label * 3} samples")
    
    balanced_data = create_large_balanced_dataset(train_data, samples_per_label)
    
    # Shuffle for good mixing
    random.seed(42)
    random.shuffle(balanced_data)
    
    # Save the large dataset
    output_path = "dataset/train/ds.jsonl"
    save_jsonl(balanced_data, output_path)
    
    print(f"\n" + "=" * 80)
    print("DATASET CREATED SUCCESSFULLY")
    print("=" * 80)
    
    # Verify the final dataset
    final_dist = Counter(s['label'] for s in balanced_data)
    print(f"\nFinal dataset: {output_path}")
    print(f"Total samples: {len(balanced_data)}")
    print("\nDistribution:")
    for label, count in sorted(final_dist.items()):
        print(f"  {label}: {count} samples")
    
    # Calculate statistics
    print("\nStatistics:")
    total_unique = len(set(s['text'] for s in balanced_data))
    print(f"  Unique texts: {total_unique}")
    print(f"  Duplicated texts: {len(balanced_data) - total_unique}")
    print(f"  Duplication rate: {(len(balanced_data) - total_unique) / len(balanced_data) * 100:.1f}%")
    
    # Average text lengths
    print("\nAverage text length per label:")
    for label in sorted(final_dist.keys()):
        texts = [s['text'] for s in balanced_data if s['label'] == label]
        avg_len = sum(len(t) for t in texts) / len(texts)
        print(f"  {label}: {avg_len:.0f} characters")
    
    # Create extra large version with augmentation if desired
    print("\n" + "-" * 80)
    print("Creating extra-large dataset (ds_xlarge.jsonl) with augmentations...")
    
    # Double the size with augmentations
    xlarge_per_label = 2500
    xlarge_data = create_large_balanced_dataset(train_data, xlarge_per_label)
    random.shuffle(xlarge_data)
    
    xlarge_path = "dataset/train/ds_xlarge.jsonl"
    save_jsonl(xlarge_data, xlarge_path)
    
    xlarge_dist = Counter(s['label'] for s in xlarge_data)
    print(f"\nExtra-large dataset: {xlarge_path}")
    print(f"Total samples: {len(xlarge_data)}")
    print("Distribution:")
    for label, count in sorted(xlarge_dist.items()):
        print(f"  {label}: {count} samples")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nDatasets created:")
    print(f"1. ds.jsonl        : {len(balanced_data)} samples ({samples_per_label} per label)")
    print(f"2. ds_xlarge.jsonl : {len(xlarge_data)} samples ({xlarge_per_label} per label)")
    print("\nUsage:")
    print("  python scripts/train_generic_chunked.py \\")
    print("    --base_model models/dclb \\")
    print("    --train_data dataset/train/ds.jsonl \\")
    print("    --output_dir models/large_balanced_model \\")
    print("    --epochs 3")

if __name__ == "__main__":
    main()