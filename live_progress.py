#!/usr/bin/env python
"""
Live training progress tracker
Parses the actual training output to show current step in real-time
"""

import re
import subprocess
import time
import os
from datetime import datetime

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_live_progress():
    """
    Parse the training output to find current step.
    Training output format: "19%|#8        | 464/2436 [6:30:16<27:41:45, 50.56s/it]"
    """
    
    # Since training is in a background process, we need to find and read its output
    # In your case, it's running as PID 20096
    
    # Try to find the latest step from any available source
    status = {
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'found': False
    }
    
    # Method 1: Check if there's a training log file
    log_paths = [
        "D:/csharp-llm/logs",
        "D:/csharp-llm/training.log",
        "D:/csharp-llm/models/tff_rag/training.log"
    ]
    
    for log_path in log_paths:
        if os.path.exists(log_path):
            try:
                # Read last lines of log
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                    for line in reversed(lines[-100:]):  # Check last 100 lines
                        # Look for pattern like "464/2436"
                        match = re.search(r'(\d+)/(\d+)', line)
                        if match:
                            status['current_step'] = int(match.group(1))
                            status['total_steps'] = int(match.group(2))
                            status['found'] = True
                            
                            # Try to find loss
                            loss_match = re.search(r'loss[:\s]+(\d+\.\d+)', line, re.IGNORECASE)
                            if loss_match:
                                status['loss'] = float(loss_match.group(1))
                            
                            # Try to find time per step
                            time_match = re.search(r'(\d+\.\d+)s/it', line)
                            if time_match:
                                status['time_per_step'] = float(time_match.group(1))
                            
                            break
            except:
                pass
    
    # Method 2: Try to capture from process (Windows specific)
    if not status['found']:
        try:
            # Get memory info about the python process
            result = subprocess.run(
                ['wmic', 'process', 'where', 'processid=20096', 'get', 'WorkingSetSize,PageFileUsage'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                # Process is still running
                status['process_running'] = True
        except:
            pass
    
    # Method 3: Estimate based on time elapsed
    if not status['found']:
        # Training started approximately at 9:00 AM, it's now around 3:30 PM
        # That's about 6.5 hours = 390 minutes = 23400 seconds
        # At ~50 seconds per step
        elapsed_seconds = (datetime.now().hour - 9) * 3600 + datetime.now().minute * 60
        estimated_step = min(int(elapsed_seconds / 50), 480)  # Cap at reasonable number
        
        status['current_step'] = estimated_step
        status['total_steps'] = 2436
        status['estimated'] = True
        status['time_per_step'] = 50.0
    
    # Calculate derived values
    if 'current_step' in status:
        status['progress_pct'] = (status['current_step'] / status['total_steps']) * 100
        status['steps_remaining'] = status['total_steps'] - status['current_step']
        
        if 'time_per_step' in status:
            status['hours_remaining'] = (status['steps_remaining'] * status['time_per_step']) / 3600
            
            # Time to next checkpoint
            next_checkpoint = ((status['current_step'] // 500) + 1) * 500
            if next_checkpoint <= status['total_steps']:
                status['next_checkpoint'] = next_checkpoint
                status['steps_to_checkpoint'] = next_checkpoint - status['current_step']
                status['mins_to_checkpoint'] = (status['steps_to_checkpoint'] * status['time_per_step']) / 60
    
    # Get GPU status
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            values = result.stdout.strip().split(', ')
            status['gpu_mem_used'] = int(values[0])
            status['gpu_mem_total'] = int(values[1])
            status['gpu_util'] = int(values[2])
            status['gpu_temp'] = int(values[3])
            status['gpu_power'] = float(values[4])
    except:
        pass
    
    return status

def display_live_status():
    clear_screen()
    status = get_live_progress()
    
    print("╔" + "═" * 58 + "╗")
    print(f"║{'🔥 LIVE Training Progress 🔥':^58}║")
    print(f"║{status['timestamp']:^58}║")
    print("╠" + "═" * 58 + "╣")
    
    if 'current_step' in status:
        # Progress bar
        progress = status.get('progress_pct', 0)
        bar_width = 40
        filled = int(bar_width * progress / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        estimate_marker = " (estimated)" if status.get('estimated') else ""
        
        print(f"║ Progress: [{bar}] {progress:5.1f}% ║")
        print(f"║ Current Step: {status['current_step']:4d}/{status['total_steps']}{estimate_marker:14} ║")
        
        if 'loss' in status:
            print(f"║ Loss: {status['loss']:.4f}                                       ║")
        
        if 'time_per_step' in status:
            print(f"║ Speed: {status['time_per_step']:.1f} seconds/step                         ║")
        
        if 'hours_remaining' in status:
            hours = int(status['hours_remaining'])
            mins = int((status['hours_remaining'] - hours) * 60)
            print(f"║ Time remaining: {hours}h {mins}m                              ║")
        
        print("╠" + "═" * 58 + "╣")
        
        # Checkpoint info
        if 'next_checkpoint' in status:
            print(f"║ 💾 Next checkpoint: Step {status['next_checkpoint']:4d}                    ║")
            print(f"║    → {status['steps_to_checkpoint']:3d} steps away ({status['mins_to_checkpoint']:.0f} minutes)            ║")
        
        # Special alert if close to checkpoint
        if status.get('steps_to_checkpoint', 999) < 20:
            print("║ ⚠️  CHECKPOINT APPROACHING SOON! ⚠️                      ║")
    else:
        print("║ Unable to read live progress...                     ║")
        print("║ Training is running in background                   ║")
    
    print("╠" + "═" * 58 + "╣")
    
    # GPU Status
    if 'gpu_mem_used' in status:
        mem_pct = (status['gpu_mem_used'] / status['gpu_mem_total']) * 100
        print(f"║ GPU Memory: {status['gpu_mem_used']:5d}/{status['gpu_mem_total']:5d} MB ({mem_pct:4.1f}%)         ║")
        print(f"║ GPU Usage: {status['gpu_util']:3d}% | Temp: {status['gpu_temp']:2d}°C | Power: {status['gpu_power']:.0f}W     ║")
    
    print("╚" + "═" * 58 + "╝")

def main():
    print("Starting LIVE progress monitor...")
    print("Note: Since training is in background, showing estimated progress")
    print("based on elapsed time (~50 seconds per step)")
    print("\nUpdates every 10 seconds. Press Ctrl+C to stop.\n")
    
    time.sleep(2)
    
    try:
        while True:
            display_live_status()
            time.sleep(10)  # Update every 10 seconds
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    main()