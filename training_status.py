import os
import json
from pathlib import Path
from datetime import datetime

output_dir = "models/balanced_model"

print("=" * 80)
print("TRAINING STATUS CHECK")
print("=" * 80)

# Check if directory exists
if not Path(output_dir).exists():
    print(f"Output directory not created yet: {output_dir}")
else:
    # Check training info
    info_file = Path(output_dir) / "training_info.json"
    if info_file.exists():
        with open(info_file, 'r') as f:
            info = json.load(f)
        
        print(f"Training Configuration:")
        print(f"  Base model: {info['base_model']}")
        print(f"  Dataset: {info['training_data']}")
        print(f"  Total samples: {info['total_samples']:,}")
        print(f"  Epochs: {info['num_epochs']}")
        print(f"  Started: {info['timestamp']}")
        
        # Calculate expected steps
        batch_size = info['batch_size']
        gradient_acc = 8
        steps_per_epoch = info['total_samples'] // (batch_size * gradient_acc)
        total_steps = steps_per_epoch * info['num_epochs']
        
        print(f"\nExpected Progress:")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Total steps: {total_steps}")
    
    # Check for checkpoints
    checkpoints = []
    for item in Path(output_dir).iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            try:
                step = int(item.name.split("-")[1])
                checkpoints.append(step)
            except:
                pass
    
    if checkpoints:
        checkpoints.sort()
        print(f"\nCheckpoints saved: {checkpoints}")
        print(f"Latest checkpoint: step {checkpoints[-1]}")
    else:
        print("\nNo checkpoints saved yet")
    
    # Check for logs
    log_dir = Path(output_dir) / "logs"
    if log_dir.exists():
        print("\nTraining logs directory exists")

print("\n" + "=" * 80)
print("Training is running in background (bash_15)")
print("The model will be saved to: models/balanced_model")
print("This will take several hours to complete.")
print("=" * 80)