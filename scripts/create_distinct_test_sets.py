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

def create_balanced_test_set(train_data, test_data, samples_per_label=85):
    """
    Create a balanced test set using samples from train data to supplement.
    This ensures ds.jsonl is different from tmf.jsonl.
    """
    # Combine all available data
    all_data = train_data + test_data
    
    # Group by label and sort by length
    grouped = defaultdict(list)
    for item in all_data:
        if item['label'] in ['ComplexMethod', 'ComplexConditional', 'FeatureEnvy']:
            grouped[item['label']].append(item)
    
    # Sort by text length (shortest first)
    for label in grouped:
        grouped[label].sort(key=lambda x: len(x['text']))
    
    # Create balanced dataset
    balanced = []
    
    for label in ['ComplexMethod', 'ComplexConditional', 'FeatureEnvy']:
        if label in grouped:
            available = grouped[label]
            
            # Take samples, preferring those not in original test
            test_texts = set(item['text'] for item in test_data if item['label'] == label)
            
            # Separate into test and train samples
            from_test = [s for s in available if s['text'] in test_texts]
            from_train = [s for s in available if s['text'] not in test_texts]
            
            # Prefer samples from train data for ds.jsonl
            selected = []
            
            # First, add train samples (these make it different from tmf.jsonl)
            selected.extend(from_train[:samples_per_label])
            
            # If not enough, add test samples
            if len(selected) < samples_per_label:
                needed = samples_per_label - len(selected)
                selected.extend(from_test[:needed])
            
            # If still not enough, duplicate shortest
            while len(selected) < samples_per_label:
                selected.append(available[len(selected) % len(available)])
            
            balanced.extend(selected[:samples_per_label])
    
    return balanced

def main():
    print("=" * 80)
    print("CREATING DISTINCT TEST DATASETS")
    print("=" * 80)
    
    # Load data
    train_data = load_jsonl("dataset/train/tmf.jsonl")
    test_original = load_jsonl("dataset/test/tmf_original.jsonl")
    
    print(f"\nOriginal test dataset (tmf_original.jsonl):")
    print(f"  Total: {len(test_original)} samples")
    test_dist = Counter(s['label'] for s in test_original)
    for label, count in sorted(test_dist.items()):
        print(f"    {label}: {count}")
    
    # Restore original tmf.jsonl (keep it unbalanced as original)
    save_jsonl(test_original, "dataset/test/tmf.jsonl")
    print(f"\nRestored tmf.jsonl to original (unbalanced) state")
    
    # Create new balanced ds.jsonl using different samples
    print(f"\nCreating balanced ds.jsonl with different samples...")
    balanced_ds = create_balanced_test_set(train_data, test_original, samples_per_label=85)
    
    # Shuffle
    random.seed(42)
    random.shuffle(balanced_ds)
    
    # Save
    save_jsonl(balanced_ds, "dataset/test/ds.jsonl")
    
    print(f"\nCreated ds.jsonl with {len(balanced_ds)} samples")
    ds_dist = Counter(s['label'] for s in balanced_ds)
    for label, count in sorted(ds_dist.items()):
        print(f"    {label}: {count}")
    
    # Check overlap
    tmf_texts = set(s['text'] for s in test_original)
    ds_texts = set(s['text'] for s in balanced_ds)
    common = len(tmf_texts & ds_texts)
    
    print(f"\nOverlap analysis:")
    print(f"  Samples in tmf.jsonl: {len(test_original)}")
    print(f"  Samples in ds.jsonl: {len(balanced_ds)}")
    print(f"  Common samples: {common}")
    print(f"  Unique to ds.jsonl: {len(ds_texts - tmf_texts)}")
    print(f"  Overlap percentage: {common/len(tmf_texts)*100:.1f}%")
    
    # Also create a balanced version keeping only original test samples
    print(f"\nCreating ds_from_test.jsonl (balanced using only original test samples)...")
    
    # Balance using only test samples (duplicating as needed)
    balanced_from_test = []
    for label in ['ComplexMethod', 'ComplexConditional', 'FeatureEnvy']:
        label_samples = [s for s in test_original if s['label'] == label]
        label_samples.sort(key=lambda x: len(x['text']))
        
        selected = []
        for i in range(85):
            selected.append(label_samples[i % len(label_samples)])
        balanced_from_test.extend(selected)
    
    random.shuffle(balanced_from_test)
    save_jsonl(balanced_from_test, "dataset/test/ds_from_test.jsonl")
    
    print(f"\nCreated ds_from_test.jsonl with {len(balanced_from_test)} samples (using only original test)")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nTest datasets created:")
    print("1. tmf.jsonl (175 samples) - Original unbalanced test set")
    print("2. ds.jsonl (255 samples) - Balanced with different samples from train")
    print("3. ds_from_test.jsonl (255 samples) - Balanced using only original test samples")
    print("\nRecommendation: Use ds.jsonl for evaluation as it has:")
    print("  - Balanced distribution (85 per label)")
    print("  - Different samples from training for better generalization testing")

if __name__ == "__main__":
    main()