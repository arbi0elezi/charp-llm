"""
Setup HuggingFace authentication for Llama models
"""

import os
from huggingface_hub import login, whoami

print("=" * 60)
print("HUGGINGFACE AUTHENTICATION SETUP")
print("=" * 60)

print("\nThis script will help you authenticate with HuggingFace.")
print("\nSTEP 1: Get access to Llama")
print("-" * 30)
print("1. Go to: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct")
print("2. Click 'Request access' button")
print("3. Fill the form (usually instant approval)")

print("\nSTEP 2: Get your token")
print("-" * 30)
print("1. Go to: https://huggingface.co/settings/tokens")
print("2. Click 'New token'")
print("3. Name it (e.g., 'llama-access')")
print("4. Select 'read' permission")
print("5. Copy the token")

print("\nSTEP 3: Enter your token")
print("-" * 30)

# Check if already logged in
try:
    user_info = whoami()
    print(f"\nYou're already logged in as: {user_info['name']}")
    print("Do you want to re-authenticate? (y/n): ", end="")
    response = input().lower()
    if response != 'y':
        print("\nUsing existing authentication.")
        exit(0)
except:
    print("Not currently authenticated.")

# Get token from user
print("\nPaste your HuggingFace token here (it will be hidden): ")
print("Token format: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
token = input("Token: ").strip()

if not token.startswith("hf_"):
    print("\nError: Token should start with 'hf_'")
    print("Please get a valid token from https://huggingface.co/settings/tokens")
    exit(1)

# Login
print("\nAuthenticating...")
try:
    login(token=token)
    print("\n✓ SUCCESS! You're now authenticated.")
    
    # Verify
    user_info = whoami()
    print(f"Logged in as: {user_info['name']}")
    
    # Test Llama access
    print("\nTesting Llama model access...")
    from transformers import AutoConfig
    try:
        config = AutoConfig.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        print("✓ Llama 3 8B access confirmed!")
        print(f"  Model type: {config.model_type}")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Layers: {config.num_hidden_layers}")
    except Exception as e:
        if "403" in str(e) or "401" in str(e):
            print("✗ Llama access not yet approved.")
            print("  Please wait for Meta to approve your request.")
            print("  Check your email for approval notification.")
        else:
            print(f"✗ Error accessing Llama: {e}")
    
    print("\n" + "=" * 60)
    print("Setup complete! You can now use Llama models.")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ Authentication failed: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure your token is valid")
    print("2. Check you have the right permissions")
    print("3. Try creating a new token")