#!/usr/bin/env python
"""
Autonomous Training Monitor with Context Preservation
This script monitors the training process and maintains state for recovery
"""

import os
import sys
import json
import time
import subprocess
import psutil
from datetime import datetime
from pathlib import Path

class TrainingMonitor:
    def __init__(self):
        self.state_file = Path("D:/csharp-llm/training_state.json")
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
            "training_started": False,
            "last_checkpoint": 0,
            "last_check_time": None,
            "crashes": 0,
            "total_steps": 2436,
            "target_epochs": 3,
            "using_test_data": False,
            "process_id": None,
            "last_loss": None,
            "last_epoch": 0,
            "errors": [],
            "auto_restart_count": 0
        }
    
    def save_state(self):
        """Save current state to file"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def log(self, message, level="INFO"):
        """Log messages with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry + "\n")
    
    def check_gpu_status(self):
        """Check GPU memory and utilization"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", 
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                values = result.stdout.strip().split(', ')
                mem_used = int(values[0])
                mem_total = int(values[1])
                gpu_util = int(values[2])
                
                mem_percent = (mem_used / mem_total) * 100
                
                self.log(f"GPU Status - Memory: {mem_used}/{mem_total}MB ({mem_percent:.1f}%), Utilization: {gpu_util}%")
                
                # Check for potential issues
                if mem_percent > 95:
                    self.log("WARNING: GPU memory usage very high!", "WARNING")
                    return False
                    
                return True
        except Exception as e:
            self.log(f"Failed to check GPU status: {e}", "ERROR")
            return None
    
    def is_training_running(self):
        """Check if training process is still running"""
        if self.state["process_id"]:
            try:
                process = psutil.Process(self.state["process_id"])
                if process.is_running() and "python" in process.name().lower():
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Alternative: Check for python processes running rag.py
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline')
                if cmdline and any('rag.py' in arg for arg in cmdline):
                    self.state["process_id"] = proc.info['pid']
                    self.save_state()
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return False
    
    def get_latest_checkpoint(self):
        """Find the latest checkpoint directory"""
        checkpoints = []
        if self.output_dir.exists():
            for item in self.output_dir.iterdir():
                if item.is_dir() and item.name.startswith("checkpoint-"):
                    try:
                        step = int(item.name.split("-")[1])
                        checkpoints.append(step)
                    except:
                        pass
        
        return max(checkpoints) if checkpoints else 0
    
    def analyze_checkpoint(self, checkpoint_num):
        """Analyze a specific checkpoint for training metrics"""
        checkpoint_dir = self.output_dir / f"checkpoint-{checkpoint_num}"
        trainer_state_path = checkpoint_dir / "trainer_state.json"
        
        if trainer_state_path.exists():
            with open(trainer_state_path, 'r') as f:
                state = json.load(f)
            
            current_step = state.get("global_step", 0)
            current_epoch = state.get("epoch", 0)
            best_metric = state.get("best_metric", None)
            
            # Get latest loss
            log_history = state.get("log_history", [])
            latest_loss = None
            for entry in reversed(log_history):
                if "loss" in entry:
                    latest_loss = entry["loss"]
                    break
            
            return {
                "step": current_step,
                "epoch": current_epoch,
                "loss": latest_loss,
                "best_metric": best_metric
            }
        return None
    
    def restart_training(self, resume_from_checkpoint=True):
        """Restart the training process"""
        self.log("Attempting to restart training...", "WARNING")
        
        # Check if we should modify the script to resume
        if resume_from_checkpoint and self.state["last_checkpoint"] > 0:
            # We would need to modify the training script to resume from checkpoint
            # For now, log the instruction
            self.log(f"Should resume from checkpoint-{self.state['last_checkpoint']}", "INFO")
            
            # Create a resume script
            resume_script = f"""
import sys
sys.path.insert(0, 'D:/csharp-llm')
from scripts.rag import train

# Modify training args to resume
import os
os.environ["RESUME_FROM_CHECKPOINT"] = "checkpoint-{self.state['last_checkpoint']}"

if __name__ == "__main__":
    train()
"""
            resume_path = Path("D:/csharp-llm/scripts/resume_training.py")
            with open(resume_path, 'w') as f:
                f.write(resume_script)
            
            cmd = [sys.executable, str(resume_path)]
        else:
            cmd = [sys.executable, str(self.script_path)]
        
        try:
            # Start training process
            process = subprocess.Popen(
                cmd,
                cwd="D:/csharp-llm",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.state["process_id"] = process.pid
            self.state["auto_restart_count"] += 1
            self.save_state()
            
            self.log(f"Training restarted with PID: {process.pid}", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"Failed to restart training: {e}", "ERROR")
            self.state["errors"].append({
                "time": datetime.now().isoformat(),
                "error": str(e),
                "action": "restart_failed"
            })
            self.save_state()
            return False
    
    def check_training_health(self):
        """Main health check routine"""
        self.log("=" * 60)
        self.log("Running training health check...")
        
        # Check if training is running
        is_running = self.is_training_running()
        self.log(f"Training process running: {is_running}")
        
        # Check GPU status
        gpu_ok = self.check_gpu_status()
        
        # Get latest checkpoint
        latest_checkpoint = self.get_latest_checkpoint()
        
        if latest_checkpoint != self.state["last_checkpoint"]:
            self.log(f"New checkpoint found: {latest_checkpoint} (was {self.state['last_checkpoint']})")
            self.state["last_checkpoint"] = latest_checkpoint
            
            # Analyze the new checkpoint
            metrics = self.analyze_checkpoint(latest_checkpoint)
            if metrics:
                self.state["last_loss"] = metrics["loss"]
                self.state["last_epoch"] = metrics["epoch"]
                
                progress_pct = (metrics["step"] / self.state["total_steps"]) * 100
                
                self.log(f"Progress: Step {metrics['step']}/{self.state['total_steps']} ({progress_pct:.1f}%)")
                self.log(f"Epoch: {metrics['epoch']:.2f}/{self.state['target_epochs']}")
                self.log(f"Loss: {metrics['loss']:.4f}" if metrics['loss'] else "Loss: N/A")
                
                # Estimate time remaining
                if metrics["step"] > 0:
                    steps_remaining = self.state["total_steps"] - metrics["step"]
                    # Assuming ~45s per step average
                    time_remaining_hours = (steps_remaining * 45) / 3600
                    self.log(f"Estimated time remaining: {time_remaining_hours:.1f} hours")
        
        # Decision logic
        should_restart = False
        
        if not is_running:
            self.log("Training process not found!", "ERROR")
            
            # Check if training completed
            if latest_checkpoint >= self.state["total_steps"] - 10:  # Near completion
                self.log("Training appears to be complete!", "SUCCESS")
                self.create_completion_report()
                return False  # Stop monitoring
            else:
                should_restart = True
                
        elif gpu_ok == False:  # Explicitly False, not None
            self.log("GPU memory critical, may need restart", "WARNING")
            # Could implement logic to pause/restart if needed
        
        if should_restart and self.state["auto_restart_count"] < 5:
            self.log(f"Attempting auto-restart (attempt {self.state['auto_restart_count'] + 1}/5)")
            if self.restart_training():
                time.sleep(60)  # Give it time to start
            else:
                self.create_error_report()
                return False  # Stop monitoring if restart failed
        elif self.state["auto_restart_count"] >= 5:
            self.log("Maximum restart attempts reached. Manual intervention needed.", "ERROR")
            self.create_error_report()
            return False
        
        self.state["last_check_time"] = datetime.now().isoformat()
        self.save_state()
        return True  # Continue monitoring
    
    def create_completion_report(self):
        """Create a report when training completes"""
        report = {
            "status": "COMPLETED",
            "timestamp": datetime.now().isoformat(),
            "final_checkpoint": self.state["last_checkpoint"],
            "final_epoch": self.state["last_epoch"],
            "final_loss": self.state["last_loss"],
            "total_restarts": self.state["auto_restart_count"],
            "model_location": str(self.output_dir),
            "next_steps": [
                "Model saved at: " + str(self.output_dir),
                "To use: Load with transformers and PEFT",
                "To evaluate: Run evaluation script with test data",
                "To continue training: Modify script to resume from checkpoint"
            ]
        }
        
        report_path = Path("D:/csharp-llm/training_completion_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.log("Training completed! Report saved to training_completion_report.json", "SUCCESS")
    
    def create_error_report(self):
        """Create an error report for manual intervention"""
        report = {
            "status": "ERROR - MANUAL INTERVENTION NEEDED",
            "timestamp": datetime.now().isoformat(),
            "last_checkpoint": self.state["last_checkpoint"],
            "last_epoch": self.state["last_epoch"],
            "last_loss": self.state["last_loss"],
            "restart_attempts": self.state["auto_restart_count"],
            "errors": self.state["errors"][-5:],  # Last 5 errors
            "recommended_actions": [
                "1. Check GPU memory: nvidia-smi",
                "2. Check disk space for model outputs",
                "3. Review training_monitor.log for details",
                "4. Manually restart with: python scripts/rag.py",
                f"5. Or resume from checkpoint-{self.state['last_checkpoint']}"
            ]
        }
        
        report_path = Path("D:/csharp-llm/training_error_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.log("Error report saved to training_error_report.json", "ERROR")
    
    def run_monitoring_loop(self, check_interval=300):
        """Main monitoring loop - checks every N seconds"""
        self.log("Starting training monitor loop...")
        self.log(f"Will check every {check_interval} seconds ({check_interval/60:.1f} minutes)")
        
        while True:
            try:
                continue_monitoring = self.check_training_health()
                
                if not continue_monitoring:
                    self.log("Monitoring stopped. Check reports for details.")
                    break
                
                # Wait before next check
                self.log(f"Next check in {check_interval} seconds...")
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                self.log("Monitoring interrupted by user", "INFO")
                break
            except Exception as e:
                self.log(f"Unexpected error in monitoring loop: {e}", "ERROR")
                self.state["errors"].append({
                    "time": datetime.now().isoformat(),
                    "error": str(e),
                    "action": "monitoring_error"
                })
                self.save_state()
                time.sleep(60)  # Brief pause before retry

if __name__ == "__main__":
    monitor = TrainingMonitor()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--check-once":
            monitor.check_training_health()
        elif sys.argv[1] == "--status":
            print(json.dumps(monitor.state, indent=2))
        else:
            interval = int(sys.argv[1]) if sys.argv[1].isdigit() else 300
            monitor.run_monitoring_loop(interval)
    else:
        # Default: run continuous monitoring every 5 minutes
        monitor.run_monitoring_loop(300)