"""
Monitor Llama 2 v2 training progress
"""

import json
import time
from pathlib import Path
from datetime import datetime

def check_progress():
    # Check for checkpoint directories
    model_dir = Path("models/llama2_finetuned_v2")
    
    if not model_dir.exists():
        print("Model directory not found. Training may not have started.")
        return False
    
    # Look for checkpoints
    checkpoints = list(model_dir.glob("checkpoint-*"))
    if checkpoints:
        latest = max(checkpoints, key=lambda p: int(p.name.split("-")[1]))
        print(f"Latest checkpoint: {latest.name}")
        
        # Check trainer state
        state_file = latest / "trainer_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            print(f"Step: {state.get('global_step', 'N/A')}/1595")
            print(f"Epoch: {state.get('epoch', 'N/A'):.2f}/5")
            
            if 'log_history' in state and state['log_history']:
                latest_log = state['log_history'][-1]
                if 'loss' in latest_log:
                    print(f"Loss: {latest_log['loss']:.4f}")
                if 'eval_loss' in latest_log:
                    print(f"Eval Loss: {latest_log['eval_loss']:.4f}")
                if 'learning_rate' in latest_log:
                    print(f"Learning Rate: {latest_log['learning_rate']:.2e}")
            
            return True
    
    # Check for training info
    info_file = model_dir / "training_info.json"
    if info_file.exists():
        print("\nTraining completed!")
        with open(info_file, 'r') as f:
            info = json.load(f)
        print(f"Duration: {info.get('training_duration', 'N/A')}")
        print(f"Final loss: {info.get('final_loss', 'N/A')}")
        return True
    
    print("No checkpoints found yet. Training in progress...")
    return False

if __name__ == "__main__":
    print("=" * 60)
    print("Llama 2 v2 Training Monitor")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    
    while True:
        check_progress()
        print("-" * 60)
        time.sleep(30)  # Check every 30 seconds