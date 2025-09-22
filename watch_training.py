#!/usr/bin/env python
"""
Live training monitor - updates every 30 seconds
Press Ctrl+C to stop
"""

import os
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_training_status():
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
    
    status = {}
    
    if checkpoints:
        latest = max(checkpoints)
        checkpoint_dir = output_dir / f"checkpoint-{latest}"
        trainer_state_path = checkpoint_dir / "trainer_state.json"
        
        if trainer_state_path.exists():
            with open(trainer_state_path, 'r') as f:
                state = json.load(f)
            
            status['checkpoint'] = latest
            status['step'] = state.get("global_step", latest)
            status['epoch'] = state.get("epoch", 0)
            
            # Get latest loss
            log_history = state.get("log_history", [])
            for entry in reversed(log_history):
                if "loss" in entry:
                    status['loss'] = entry["loss"]
                    break
    
    # Get GPU status
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            values = result.stdout.strip().split(', ')
            status['gpu_mem_used'] = int(values[0])
            status['gpu_mem_total'] = int(values[1])
            status['gpu_util'] = int(values[2])
            status['gpu_temp'] = int(values[3])
    except:
        pass
    
    return status

def display_status():
    clear_screen()
    status = get_training_status()
    
    print("╔" + "═" * 58 + "╗")
    print(f"║{'DeepSeek C# Training Monitor':^58}║")
    print(f"║{datetime.now().strftime('%Y-%m-%d %H:%M:%S'):^58}║")
    print("╠" + "═" * 58 + "╣")
    
    total_steps = 2436
    
    if 'step' in status:
        step = status['step']
        progress = (step / total_steps) * 100
        
        # Progress bar
        bar_width = 40
        filled = int(bar_width * progress / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        print(f"║ Progress: [{bar}] {progress:5.1f}% ║")
        print(f"║ Step: {step:4d}/{total_steps} | Epoch: {status.get('epoch', 0):4.2f}/3.00        ║")
        
        if 'loss' in status:
            print(f"║ Loss: {status['loss']:.4f}                                       ║")
        
        # Time estimates
        steps_remaining = total_steps - step
        hours_remaining = (steps_remaining * 50) / 3600
        
        print(f"║ Est. time remaining: {hours_remaining:4.1f} hours                    ║")
        
        # Next checkpoint
        next_cp = ((step // 500) + 1) * 500
        if next_cp <= total_steps:
            steps_to_cp = next_cp - step
            mins_to_cp = (steps_to_cp * 50) / 60
            print(f"║ Next checkpoint: {next_cp:4d} (in {mins_to_cp:3.0f} min)              ║")
    else:
        # No checkpoint yet, estimate
        # Training started ~6.5 hours ago, ~50s per step
        elapsed_hours = 6.5
        estimated_step = int((elapsed_hours * 3600) / 50)
        estimated_step = min(estimated_step, 470)  # Cap at reasonable estimate
        
        progress = (estimated_step / total_steps) * 100
        bar_width = 40
        filled = int(bar_width * progress / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        print(f"║ Progress: [{bar}] ~{progress:4.1f}% ║")
        print(f"║ Estimated step: ~{estimated_step}/{total_steps}                     ║")
        print(f"║ First checkpoint saves at step 500                  ║")
        
        steps_to_500 = max(500 - estimated_step, 0)
        mins_to_cp = (steps_to_500 * 50) / 60
        print(f"║ First checkpoint in: ~{mins_to_cp:3.0f} minutes                 ║")
    
    print("╠" + "═" * 58 + "╣")
    
    # GPU Status
    if 'gpu_mem_used' in status:
        mem_pct = (status['gpu_mem_used'] / status['gpu_mem_total']) * 100
        print(f"║ GPU Memory: {status['gpu_mem_used']:5d}/{status['gpu_mem_total']:5d} MB ({mem_pct:4.1f}%)         ║")
        print(f"║ GPU Usage: {status['gpu_util']:3d}% | Temperature: {status['gpu_temp']:2d}°C            ║")
    
    print("╚" + "═" * 58 + "╝")
    print("\nPress Ctrl+C to stop monitoring...")

def main():
    print("Starting training monitor...")
    print("Will update every 30 seconds")
    
    try:
        while True:
            display_status()
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    main()