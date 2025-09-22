"""
Quick test to verify Llama 2 fix works
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Test with fine-tuned model
print("Testing Llama 2 fine-tuned model fix...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("models/llama2_finetuned")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load model with quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print("Loading model...")
base_model_id = "NousResearch/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# Load LoRA adapters
model = PeftModel.from_pretrained(model, "models/llama2_finetuned")
model = model.merge_and_unload()
model.eval()

print("Model loaded! Testing generation...")

# Test code
test_code = """
public void ProcessData(DataManager manager) {
    var data = manager.GetData();
    var processed = manager.ProcessData(data);
    var validated = manager.ValidateData(processed);
    manager.SaveData(validated);
    manager.LogActivity("Data processed");
}
"""

# Create prompt matching training format
prompt = f"""<s>[INST] <<SYS>>
You are a code smell classifier. Analyze C# code and identify the specific code smell present.
<</SYS>>

Classify this C# code snippet and identify the code smell:

```csharp
{test_code}
```

What code smell does this code exhibit? [/INST]

This code exhibits the"""

print("\nPrompt (last 200 chars):")
print(prompt[-200:])

# Generate
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        temperature=0.1,
        do_sample=True,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

# Decode only generated part
generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

print(f"\nGenerated response: '{response}'")

# Check label extraction
response_lower = response.lower().strip()
if "featureenvy" in response_lower or "feature envy" in response_lower:
    print("Extracted label: FeatureEnvy ✓")
elif "complexmethod" in response_lower or "complex method" in response_lower:
    print("Extracted label: ComplexMethod")
elif "complexconditional" in response_lower or "complex conditional" in response_lower:
    print("Extracted label: ComplexConditional")
else:
    print(f"Could not extract label from: '{response}'")

print("\nTest complete!")