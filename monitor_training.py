import time
import os
import json
from datetime import datetime

def check_training_progress():
    """Check the latest training checkpoint and progress"""
    output_dir = "D:\\csharp-llm\\models\\tff_rag"
    
    # Find all checkpoint directories
    checkpoints = []
    if os.path.exists(output_dir):
        for item in os.listdir(output_dir):
            if item.startswith("checkpoint-"):
                try:
                    step = int(item.split("-")[1])
                    checkpoints.append(step)
                except:
                    pass
    
    if checkpoints:
        latest_checkpoint = max(checkpoints)
        trainer_state_path = os.path.join(output_dir, f"checkpoint-{latest_checkpoint}", "trainer_state.json")
        
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path, 'r') as f:
                state = json.load(f)
                
            current_step = state.get("global_step", 0)
            best_metric = state.get("best_metric", "N/A")
            current_epoch = state.get("epoch", 0)
            
            # Get latest loss from log history
            log_history = state.get("log_history", [])
            latest_loss = "N/A"
            if log_history:
                for entry in reversed(log_history):
                    if "loss" in entry:
                        latest_loss = entry["loss"]
                        break
            
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training Progress:")
            print(f"  Latest checkpoint: step {latest_checkpoint}")
            print(f"  Current epoch: {current_epoch:.2f}/3.0")
            print(f"  Latest loss: {latest_loss}")
            print(f"  Best metric: {best_metric}")
            
            # Estimate time remaining (assuming ~50s per step from test run)
            total_steps = 2435  # Approximate for 6494 samples with batch_size=1, grad_accum=8
            if current_step > 0:
                progress_pct = (current_step / total_steps) * 100
                print(f"  Progress: {current_step}/{total_steps} steps ({progress_pct:.1f}%)")
                
                # Rough estimate
                steps_remaining = total_steps - current_step
                time_remaining_minutes = (steps_remaining * 50) / 60
                hours = int(time_remaining_minutes // 60)
                minutes = int(time_remaining_minutes % 60)
                print(f"  Estimated time remaining: {hours}h {minutes}m")
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] No checkpoints found yet. Training may still be initializing...")

if __name__ == "__main__":
    print("Monitoring training progress...")
    print("Will check every 5 minutes. Press Ctrl+C to stop.")
    
    while True:
        check_training_progress()
        time.sleep(300)  # Check every 5 minutes