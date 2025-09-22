#!/usr/bin/env python
"""
Quick training status checker
Shows current progress, GPU usage, and estimates
"""

import os
import json
import subprocess
from datetime import datetime
from pathlib import Path

def check_training_status():
    output_dir = Path("D:/csharp-llm/models/tff_rag")
    
    # Find all checkpoints
    checkpoints = []
    if output_dir.exists():
        for item in output_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                try:
                    step = int(item.name.split("-")[1])
                    checkpoints.append(step)
                except:
                    pass
    
    print("=" * 60)
    print(f"Training Status - {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 60)
    
    if checkpoints:
        latest = max(checkpoints)
        checkpoint_dir = output_dir / f"checkpoint-{latest}"
        trainer_state_path = checkpoint_dir / "trainer_state.json"
        
        if trainer_state_path.exists():
            with open(trainer_state_path, 'r') as f:
                state = json.load(f)
            
            current_step = state.get("global_step", 0)
            current_epoch = state.get("epoch", 0)
            
            # Get latest loss
            log_history = state.get("log_history", [])
            latest_loss = None
            for entry in reversed(log_history):
                if "loss" in entry:
                    latest_loss = entry["loss"]
                    break
            
            total_steps = 2436  # For full training
            progress_pct = (current_step / total_steps) * 100
            
            print(f"Latest checkpoint: Step {current_step}/{total_steps} ({progress_pct:.1f}%)")
            print(f"Epoch: {current_epoch:.2f}/3.0")
            if latest_loss:
                print(f"Loss: {latest_loss:.4f}")
            
            # Time estimates
            if current_step > 0:
                steps_remaining = total_steps - current_step
                time_remaining_hours = (steps_remaining * 50) / 3600  # ~50s per step
                print(f"Estimated time remaining: {time_remaining_hours:.1f} hours")
            
            # Next checkpoint
            next_checkpoint = ((current_step // 500) + 1) * 500
            if next_checkpoint <= total_steps:
                steps_to_next = next_checkpoint - current_step
                time_to_next = (steps_to_next * 50) / 60
                print(f"\nNext checkpoint: Step {next_checkpoint} (in ~{time_to_next:.0f} minutes)")
    else:
        print("No checkpoints found yet")
        print("First checkpoint will save at step 500")
        print("Currently training is in progress...")
        
        # Estimate based on start time (roughly 6.5 hours ago)
        estimated_current = 464  # Update this based on when you run
        progress_pct = (estimated_current / 2436) * 100
        steps_to_500 = 500 - estimated_current
        time_to_checkpoint = (steps_to_500 * 50) / 60
        
        print(f"\nEstimated current: ~Step {estimated_current}/2436 ({progress_pct:.1f}%)")
        print(f"First checkpoint in: ~{time_to_checkpoint:.0f} minutes")
    
    # Check GPU
    print("\nGPU Status:")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            values = result.stdout.strip().split(', ')
            mem_used = int(values[0])
            mem_total = int(values[1])
            gpu_util = int(values[2])
            temp = int(values[3])
            power = float(values[4])
            
            print(f"  Memory: {mem_used}/{mem_total} MB ({(mem_used/mem_total)*100:.1f}%)")
            print(f"  Utilization: {gpu_util}%")
            print(f"  Temperature: {temp}°C")
            print(f"  Power: {power:.1f}W")
    except Exception as e:
        print(f"  Could not read GPU status: {e}")
    
    print("=" * 60)

if __name__ == "__main__":
    check_training_status()