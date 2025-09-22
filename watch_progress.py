import json
import time
import os
from datetime import datetime

def watch_progress():
    """Monitor evaluation progress in real-time."""
    print("=" * 80)
    print("EVALUATION PROGRESS MONITOR")
    print("=" * 80)
    print(f"Started monitoring at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    
    progress_file = "evaluation_results/progress.json"
    detailed_file = "evaluation_results/detailed_predictions.json"
    
    last_completed = -1
    
    while True:
        try:
            # Read progress file
            if os.path.exists(progress_file):
                with open(progress_file, 'r') as f:
                    progress = json.load(f)
                
                completed = progress.get('completed', 0)
                total = progress.get('total_samples', 175)
                status = progress.get('status', 'unknown')
                
                # Only print if there's new progress
                if completed != last_completed:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    
                    if status == 'loading_model':
                        print(f"[{timestamp}] Status: Loading model...")
                    elif status == 'evaluating':
                        pct = (completed / total) * 100
                        acc = progress.get('current_accuracy', 0)
                        last_pred = progress.get('last_prediction', 'N/A')
                        last_true = progress.get('last_true_label', 'N/A')
                        
                        # Check if last prediction was correct
                        correct_symbol = "✓" if last_pred == last_true else "✗"
                        
                        print(f"[{timestamp}] Progress: {completed:3}/{total} ({pct:5.1f}%) | "
                              f"Accuracy: {acc:.3f} | "
                              f"Last: {last_true} -> {last_pred} {correct_symbol}")
                    elif status == 'completed':
                        final_acc = progress.get('final_accuracy', 0)
                        print(f"[{timestamp}] COMPLETED! Final Accuracy: {final_acc:.4f}")
                        print("=" * 80)
                        
                        # Show final summary
                        if os.path.exists(detailed_file):
                            with open(detailed_file, 'r') as f:
                                results = json.load(f)
                            
                            correct = sum(1 for r in results if r['correct'])
                            print(f"Total: {len(results)} samples")
                            print(f"Correct: {correct}")
                            print(f"Incorrect: {len(results) - correct}")
                            
                            # Count predictions by type
                            from collections import Counter
                            pred_counts = Counter(r['predicted_label'] for r in results)
                            true_counts = Counter(r['true_label'] for r in results)
                            
                            print("\nTrue Label Distribution:")
                            for label, count in true_counts.most_common():
                                print(f"  {label}: {count}")
                            
                            print("\nPredicted Label Distribution:")
                            for label, count in pred_counts.most_common():
                                print(f"  {label}: {count}")
                        
                        break
                    
                    last_completed = completed
            
            time.sleep(2)  # Check every 2 seconds
            
        except Exception as e:
            # File might be being written to
            time.sleep(1)
            continue
    
    print(f"\nMonitoring ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    watch_progress()