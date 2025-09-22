import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import sys

# Models that should work without authentication - focusing on truly open ones
open_models = [
    "microsoft/phi-2",  # Microsoft's small model
    "EleutherAI/gpt-neo-1.3B",  # EleutherAI models are open
    "EleutherAI/gpt-neo-2.7B",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b", 
    "bigcode/starcoder2-3b",  # BigCode models are typically open
    "stabilityai/stablelm-3b-4e1t",  # Stability AI
    "facebook/opt-1.3b",  # Facebook OPT models are open
    "facebook/opt-2.7b",
    "google/gemma-2b",  # Google's smaller model might be open
    "Qwen/Qwen1.5-1.8B",  # Alibaba's model
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Community model
]

print(f"Testing open models that don't require authentication...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

print("\n" + "="*60)

working_models = []

for model_id in open_models:
    print(f"\nTesting: {model_id}")
    print("-" * 40)
    
    try:
        # Try to get config first (lightweight check)
        print("  Checking config access...", end="")
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        print(" OK")
        
        # Get model details
        print(f"  Model type: {config.model_type}")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Layers: {config.num_hidden_layers}")
        print(f"  Max length: {getattr(config, 'max_position_embeddings', 'N/A')}")
        
        working_models.append({
            'id': model_id,
            'type': config.model_type,
            'layers': config.num_hidden_layers,
            'hidden_size': config.hidden_size
        })
        print(f"  [OK] ACCESSIBLE without authentication!")
        
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "unauthorized" in error_msg.lower() or "restricted" in error_msg.lower() or "gated" in error_msg.lower():
            print(f"  [X] Requires authentication")
        elif "404" in error_msg:
            print(f"  [X] Model not found")
        else:
            print(f"  [X] Error: {error_msg[:100]}...")

print("\n" + "="*60)
print("\nSUMMARY - Models accessible without authentication:")
print("-" * 40)

if working_models:
    for model in working_models:
        print(f"[OK] {model['id']}")
        print(f"     Type: {model['type']}, Layers: {model['layers']}, Hidden: {model['hidden_size']}")
    
    print("\nRECOMMENDATION for code smell detection:")
    # Sort by size (layers * hidden_size as proxy)
    models_by_size = sorted(working_models, key=lambda x: x['layers'] * x['hidden_size'])
    
    print("\nSmaller models (faster training, good for testing):")
    for model in models_by_size[:3]:
        print(f"  -> {model['id']} ({model['layers']} layers)")
    
    print("\nLarger models (better performance, slower training):")
    for model in models_by_size[-3:]:
        print(f"  -> {model['id']} ({model['layers']} layers)")
        
else:
    print("[X] No models accessible without authentication")
    print("\nYou may need to:")
    print("1. Run: huggingface-cli login")
    print("2. Get access to gated models on HuggingFace")