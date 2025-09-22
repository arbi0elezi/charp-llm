#!/usr/bin/env python
"""
Claude Training Monitor - Autonomous monitoring that triggers Claude when needed
This script monitors training and creates a context file for Claude to resume
"""

import os
import sys
import json
import time
import subprocess
import psutil
from datetime import datetime
from pathlib import Path

class ClaudeTrainingMonitor:
    def __init__(self):
        self.state_file = Path("D:/csharp-llm/training_state.json")
        self.claude_context_file = Path("D:/csharp-llm/CLAUDE_CONTEXT.md")
        self.log_file = Path("D:/csharp-llm/training_monitor.log")
        self.output_dir = Path("D:/csharp-llm/models/tff_rag")
        self.script_path = Path("D:/csharp-llm/scripts/rag.py")
        self.state = self.load_state()
        
    def load_state(self):
        """Load previous state or initialize new one"""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            "training_started": True,
            "last_checkpoint": 0,
            "last_check_time": datetime.now().isoformat(),
            "total_steps": 2436,
            "target_epochs": 3,
            "bash_id": "bash_10",  # Current training process ID in Claude
            "last_loss": None,
            "last_epoch": 0,
            "issues_detected": [],
            "needs_claude_intervention": False
        }
    
    def save_state(self):
        """Save current state to file"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def create_claude_context(self, issue_type, details):
        """Create a context file for Claude to understand the situation"""
        
        context = f"""# CLAUDE TRAINING MONITOR ALERT

## Immediate Context
You are monitoring a deepseek-coder-v2-lite-base fine-tuning job for C# code smell detection.
The training was started in background process `{self.state['bash_id']}`.

## Current Situation
**Issue Type:** {issue_type}
**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Last Checkpoint:** {self.state['last_checkpoint']}
**Progress:** Step {self.state['last_checkpoint']}/{self.state['total_steps']} ({(self.state['last_checkpoint']/self.state['total_steps']*100):.1f}%)
**Last Epoch:** {self.state['last_epoch']:.2f}/3.0
**Last Loss:** {self.state['last_loss'] if self.state['last_loss'] else 'Unknown'}

## Issue Details
{details}

## Training Configuration
- Model: deepseek-coder-v2-lite-base (15.7B params)
- LoRA: r=16, targeting q_proj, k_proj, v_proj, o_proj
- Dataset: 6494 samples from train/tmf.jsonl
- Batch size: 1, Gradient accumulation: 8
- GPU: RTX A5000 (24GB VRAM)
- Output directory: D:\\csharp-llm\\models\\tff_rag

## Available Commands
1. Check training status: `BashOutput bash_id="{self.state['bash_id']}"`
2. Check GPU: `Bash command="nvidia-smi"`
3. Check latest checkpoint: `LS path="D:\\csharp-llm\\models\\tff_rag"`
4. Resume training if needed: Edit rag.py and restart

## Recommended Actions
"""
        
        if issue_type == "TRAINING_STOPPED":
            context += """
1. First check if training completed successfully (reached 2436 steps)
2. If not complete, check the last output with BashOutput
3. Look for OOM errors or other issues
4. Consider resuming from last checkpoint
5. May need to reduce batch size or sequence length if OOM
"""
        elif issue_type == "TRAINING_STALLED":
            context += """
1. Check if the process is still running with BashOutput
2. Look at GPU utilization with nvidia-smi
3. Check if it's stuck on a particular batch
4. May need to kill and restart if truly frozen
"""
        elif issue_type == "HIGH_LOSS":
            context += """
1. Check the loss trend over recent checkpoints
2. Verify the learning rate schedule
3. Consider if model is overfitting
4. May need to adjust hyperparameters
"""
        elif issue_type == "GPU_MEMORY_CRITICAL":
            context += """
1. Check current GPU memory usage
2. Consider reducing batch size (currently 1)
3. Reduce max sequence length (currently 512)
4. Clear GPU cache if needed
"""
        
        context += f"""

## Previous Issues Detected
{json.dumps(self.state['issues_detected'][-5:], indent=2) if self.state['issues_detected'] else 'None'}

## Next Steps
Please investigate the issue and take appropriate action to ensure training continues successfully.
The training should complete all 2436 steps (3 epochs) and save the final model.

Remember: This is running on Windows with paths like D:\\csharp-llm\\
"""
        
        # Save context file
        with open(self.claude_context_file, 'w') as f:
            f.write(context)
        
        print("\n" + "="*60)
        print("CLAUDE INTERVENTION NEEDED!")
        print("="*60)
        print(f"Issue: {issue_type}")
        print(f"Context saved to: {self.claude_context_file}")
        print("\nTo resume: Open Claude and provide this context")
        print("="*60 + "\n")
        
        # Also save a flag file that could trigger other automations
        trigger_file = Path("D:/csharp-llm/CLAUDE_NEEDED.flag")
        with open(trigger_file, 'w') as f:
            f.write(json.dumps({
                "issue": issue_type,
                "time": datetime.now().isoformat(),
                "bash_id": self.state['bash_id'],
                "checkpoint": self.state['last_checkpoint']
            }))
    
    def check_training_process(self):
        """Check if training is still running via process check"""
        # Check for python processes running rag.py
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline')
                if cmdline and any('rag.py' in arg for arg in cmdline):
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False
    
    def get_latest_checkpoint_info(self):
        """Get information about the latest checkpoint"""
        checkpoints = []
        if self.output_dir.exists():
            for item in self.output_dir.iterdir():
                if item.is_dir() and item.name.startswith("checkpoint-"):
                    try:
                        step = int(item.name.split("-")[1])
                        checkpoints.append(step)
                    except:
                        pass
        
        if not checkpoints:
            return None
            
        latest = max(checkpoints)
        checkpoint_dir = self.output_dir / f"checkpoint-{latest}"
        trainer_state_path = checkpoint_dir / "trainer_state.json"
        
        if trainer_state_path.exists():
            with open(trainer_state_path, 'r') as f:
                state = json.load(f)
            
            # Get latest loss
            log_history = state.get("log_history", [])
            latest_loss = None
            for entry in reversed(log_history):
                if "loss" in entry:
                    latest_loss = entry["loss"]
                    break
            
            return {
                "step": state.get("global_step", latest),
                "epoch": state.get("epoch", 0),
                "loss": latest_loss
            }
        
        return {"step": latest, "epoch": 0, "loss": None}
    
    def check_gpu_memory(self):
        """Check GPU memory usage"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", 
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                mem_used = int(values[0])
                mem_total = int(values[1])
                mem_percent = (mem_used / mem_total) * 100
                return mem_percent
        except:
            pass
        return None
    
    def monitor_training(self):
        """Main monitoring function"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Checking training status...")
        
        # Check if process is running
        is_running = self.check_training_process()
        
        # Get latest checkpoint info
        checkpoint_info = self.get_latest_checkpoint_info()
        
        # Check GPU memory
        gpu_mem_percent = self.check_gpu_memory()
        
        # Update state
        if checkpoint_info:
            old_checkpoint = self.state['last_checkpoint']
            self.state['last_checkpoint'] = checkpoint_info['step']
            self.state['last_epoch'] = checkpoint_info['epoch']
            self.state['last_loss'] = checkpoint_info['loss']
            
            print(f"  Latest checkpoint: {checkpoint_info['step']}/{self.state['total_steps']}")
            print(f"  Epoch: {checkpoint_info['epoch']:.2f}/3.0")
            print(f"  Loss: {checkpoint_info['loss']:.4f}" if checkpoint_info['loss'] else "  Loss: N/A")
            
            # Check if training is complete
            if checkpoint_info['step'] >= self.state['total_steps'] - 5:
                print("  Training appears to be COMPLETE!")
                self.create_claude_context(
                    "TRAINING_COMPLETED",
                    f"Training has reached step {checkpoint_info['step']} of {self.state['total_steps']}. Final model should be saved."
                )
                return False  # Stop monitoring
            
            # Check if training is stalled (no progress in last check)
            if old_checkpoint == checkpoint_info['step'] and old_checkpoint > 0:
                minutes_since_last = 5  # We check every 5 minutes
                if not is_running:
                    print("  WARNING: Training process not found!")
                    self.state['issues_detected'].append({
                        "time": datetime.now().isoformat(),
                        "issue": "process_stopped",
                        "checkpoint": checkpoint_info['step']
                    })
                    self.create_claude_context(
                        "TRAINING_STOPPED",
                        f"Training process is not running. Last checkpoint: {checkpoint_info['step']}"
                    )
                    return False
                else:
                    print(f"  WARNING: No progress in {minutes_since_last} minutes")
                    # Could be processing a slow batch, wait one more cycle
        
        # Check GPU memory
        if gpu_mem_percent:
            print(f"  GPU Memory: {gpu_mem_percent:.1f}%")
            if gpu_mem_percent > 95:
                print("  WARNING: GPU memory critical!")
                self.state['issues_detected'].append({
                    "time": datetime.now().isoformat(),
                    "issue": "gpu_memory_critical",
                    "memory_percent": gpu_mem_percent
                })
                # Don't immediately trigger Claude, but note the issue
        
        if not is_running and (not checkpoint_info or checkpoint_info['step'] < self.state['total_steps'] - 5):
            print("  ERROR: Training not running and not complete!")
            self.create_claude_context(
                "TRAINING_STOPPED",
                f"Training stopped unexpectedly at step {checkpoint_info['step'] if checkpoint_info else 0}"
            )
            return False
        
        self.state['last_check_time'] = datetime.now().isoformat()
        self.save_state()
        
        print(f"  Status: {'Running' if is_running else 'Not Running'}")
        print(f"  Next check in 5 minutes...")
        return True

def main():
    monitor = ClaudeTrainingMonitor()
    
    print("="*60)
    print("Claude Training Monitor Started")
    print("="*60)
    print(f"Monitoring training in background")
    print(f"Will check every 5 minutes")
    print(f"Will create context file if intervention needed")
    print("Press Ctrl+C to stop monitoring")
    print("="*60 + "\n")
    
    while True:
        try:
            continue_monitoring = monitor.monitor_training()
            
            if not continue_monitoring:
                print("\nMonitoring stopped. Check CLAUDE_CONTEXT.md for details.")
                break
            
            # Wait 5 minutes before next check
            time.sleep(300)
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"\nError in monitoring: {e}")
            monitor.create_claude_context(
                "MONITORING_ERROR",
                f"Monitor script crashed with error: {str(e)}"
            )
            break

if __name__ == "__main__":
    main()