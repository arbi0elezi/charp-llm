"""
Setup script for Llama model authentication.
Llama models require HuggingFace authentication.
"""

import os
import sys

print("=" * 80)
print("LLAMA MODEL SETUP")
print("=" * 80)
print("\nLlama models require authentication from Meta.")
print("\nTo use Llama models, you need to:")
print("1. Request access at: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct")
print("2. Get your HuggingFace token from: https://huggingface.co/settings/tokens")
print("3. Set your token using one of these methods:")
print("   a) Run: huggingface-cli login")
print("   b) Set environment variable: HF_TOKEN=your_token_here")
print("\nOnce you have access, the download and training scripts will work.")

# Check if token is already set
token = os.getenv("HF_TOKEN")
if token:
    print(f"\n✓ HF_TOKEN environment variable is set (length: {len(token)})")
else:
    print("\n✗ HF_TOKEN environment variable is not set")
    
print("\nWould you like to set your token now? (y/n): ", end="")
response = input().strip().lower()

if response == 'y':
    print("Enter your HuggingFace token: ", end="")
    token = input().strip()
    
    # Save to environment variable for this session
    os.environ["HF_TOKEN"] = token
    
    # Optionally save to a .env file
    print("\nSave token to .env file for future use? (y/n): ", end="")
    if input().strip().lower() == 'y':
        with open(".env", "a") as f:
            f.write(f"\nHF_TOKEN={token}\n")
        print("✓ Token saved to .env file")
    
    print("\n✓ Token set for this session")
    print("You can now run the Llama download and training scripts.")
else:
    print("\nPlease set up authentication before running Llama scripts.")
    sys.exit(1)