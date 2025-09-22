import time
import json
import os
from datetime import datetime

def monitor_evaluation():
    """Monitor the evaluation progress by checking output files."""
    
    detailed_file = "evaluation_results/detailed_predictions.json"
    summary_file = "evaluation_results/evaluation_summary.json"
    
    print("=" * 80)
    print("EVALUATION MONITOR")
    print("=" * 80)
    print(f"Started monitoring at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Checking for: {detailed_file}")
    print("-" * 80)
    
    last_count = 0
    
    while True:
        try:
            # Check if detailed predictions file exists
            if os.path.exists(detailed_file):
                with open(detailed_file, 'r') as f:
                    data = json.load(f)
                    current_count = len(data)
                    
                    if current_count > last_count:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Processed: {current_count} samples")
                        
                        # Show recent predictions
                        if current_count > 0:
                            latest = data[-1]
                            accuracy = sum(1 for d in data if d['correct']) / len(data)
                            print(f"  Current accuracy: {accuracy:.3f}")
                            print(f"  Last prediction: {latest['true_label']} -> {latest['predicted_label']} "
                                  f"({'✓' if latest['correct'] else '✗'})")
                        
                        last_count = current_count
                        
                        # Check if completed (175 total samples)
                        if current_count >= 175:
                            print("\n" + "=" * 80)
                            print("EVALUATION COMPLETE!")
                            print("=" * 80)
                            
                            # Load and show summary
                            if os.path.exists(summary_file):
                                with open(summary_file, 'r') as f:
                                    summary = json.load(f)
                                    print(f"Final Accuracy: {summary['accuracy']:.4f}")
                                    print(f"Correct: {summary['correct_predictions']}/{summary['total_samples']}")
                                    print(f"Time: {summary['evaluation_time_seconds']:.1f} seconds")
                            break
            
            time.sleep(10)  # Check every 10 seconds
            
        except Exception as e:
            # File might be being written to
            time.sleep(5)
            continue
    
    print(f"\nMonitoring ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    monitor_evaluation()