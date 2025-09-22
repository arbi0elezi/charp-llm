"""
Alternative: Use Qwen model in place of Llama for testing.
This creates a symbolic link or copies Qwen to use as "Llama" for testing purposes.
"""

import os
import shutil
from pathlib import Path

print("=" * 80)
print("USING QWEN AS LLAMA ALTERNATIVE")
print("=" * 80)
print("\nSince Llama requires authentication, we can use Qwen 7B as an alternative.")
print("Qwen 7B is a powerful model comparable to Llama 8B for code smell detection.")

qwen_path = Path("models/qwen_7b_base")
llama_path = Path("models/llama_8b_baseline")

if not qwen_path.exists():
    print(f"\n[ERROR] Qwen model not found at {qwen_path}")
    print("Please run: python scripts/download_qwen.py")
    exit(1)

if llama_path.exists():
    print(f"\n[WARNING] {llama_path} already exists.")
    response = input("Overwrite with Qwen model? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        exit(0)
    shutil.rmtree(llama_path)

print(f"\n[INFO] Creating symlink from Qwen to Llama path...")
print(f"  Source: {qwen_path.absolute()}")
print(f"  Target: {llama_path.absolute()}")

# On Windows, we need to copy instead of symlink (unless running as admin)
try:
    # Try to create a symlink first
    llama_path.symlink_to(qwen_path.absolute())
    print("\n[SUCCESS] Created symbolic link successfully")
except:
    # Fall back to copying
    print("\n[INFO] Cannot create symlink, copying files instead...")
    shutil.copytree(qwen_path, llama_path)
    print("\n[SUCCESS] Copied Qwen model to Llama path")

print("\n[INFO] You can now use the 'Llama' scripts with the Qwen model:")
print("  - Training: python scripts/train_llama_balanced.py")
print("  - Evaluation: python scripts/eval_llama.py baseline")
print("\n[NOTE] This is using Qwen 7B, not actual Llama 8B")
print("Results will be saved as 'llama_8b' but are actually from Qwen 7B")