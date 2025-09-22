import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

print(f"Attempting to access {model_id}...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    print("\n1. Trying to load tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_auth_token=False,
        trust_remote_code=True
    )
    print("   ✓ Tokenizer loaded successfully!")
    
    print("\n2. Checking model config...")
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(
        model_id,
        use_auth_token=False,
        trust_remote_code=True
    )
    print(f"   ✓ Model config accessible!")
    print(f"   Model type: {config.model_type}")
    print(f"   Hidden size: {config.hidden_size}")
    print(f"   Num layers: {config.num_hidden_layers}")
    
    print("\n3. Attempting to download model (this will fail if auth is required)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        use_auth_token=False,
        trust_remote_code=True
    )
    print("   ✓ Model loaded successfully! No authentication required.")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    if "401" in str(e) or "unauthorized" in str(e).lower() or "restricted" in str(e).lower():
        print("\n⚠ This model requires authentication. You need to:")
        print("  1. Get access approval from Meta at: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct")
        print("  2. Login with: huggingface-cli login")
        print("\nAlternative open models you can use without authentication:")
        print("  - microsoft/phi-2")
        print("  - mistralai/Mistral-7B-Instruct-v0.2")
        print("  - teknium/OpenHermes-2.5-Mistral-7B")
        sys.exit(1)
    else:
        print(f"Unexpected error: {e}")
        sys.exit(1)

print("\n✓ Success! The model can be accessed without authentication.")