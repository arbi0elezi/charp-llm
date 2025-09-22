"""
Test script for the improved Llama 2 fine-tuned model (v2)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json
from pathlib import Path

print("=" * 60)
print("Testing Llama 2 Fine-tuned Model v2")
print("=" * 60)

# Check if model exists
model_path = "models/llama2_finetuned_v2"
if not Path(model_path).exists():
    print(f"ERROR: Model not found at {model_path}")
    print("Training may still be in progress.")
    exit(1)

# Load tokenizer
print("\n1. Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load model with quantization
print("\n2. Loading model with 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

base_model_id = "NousResearch/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# Load LoRA adapters
print("\n3. Loading LoRA adapters...")
model = PeftModel.from_pretrained(model, model_path)
model = model.merge_and_unload()
model.eval()

print("\n4. Testing on sample code...")

# Test samples for each code smell type
test_samples = [
    {
        "code": """
public void ProcessData(DataManager manager) {
    var data = manager.GetData();
    var processed = manager.ProcessData(data);
    var validated = manager.ValidateData(processed);
    manager.SaveData(validated);
    manager.LogActivity("Data processed");
}""",
        "expected": "FeatureEnvy"
    },
    {
        "code": """
public void ProcessOrder(Order order) {
    if (order.Status == "NEW") {
        if (order.Priority == "HIGH") {
            if (order.Customer.Type == "PREMIUM") {
                if (order.Items.Count > 10) {
                    // Process premium high priority large order
                }
            }
        }
    }
}""",
        "expected": "ComplexConditional"
    },
    {
        "code": """
public List<Product> GetProductsWithDiscounts(List<Product> products, 
    List<Category> categories, List<Discount> discounts, 
    DateTime startDate, DateTime endDate, decimal minPrice, 
    decimal maxPrice, string region, CustomerType customerType) {
    var result = new List<Product>();
    foreach (var product in products) {
        var category = categories.FirstOrDefault(c => c.Id == product.CategoryId);
        if (category != null && category.IsActive) {
            var applicableDiscounts = discounts.Where(d => 
                d.CategoryId == category.Id && 
                d.StartDate <= startDate && 
                d.EndDate >= endDate);
            foreach (var discount in applicableDiscounts) {
                // Many more lines of complex logic...
            }
        }
    }
    return result;
}""",
        "expected": "ComplexMethod"
    }
]

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

correct = 0
for i, sample in enumerate(test_samples, 1):
    print(f"\nTest {i}: Expected = {sample['expected']}")
    print("-" * 40)
    
    # Create prompt using the training format
    prompt = f"""### Task: Identify the code smell in this C# code.

### Code:
```csharp
{sample['code']}
```

### Answer:"""
    
    # Generate prediction
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the generated part
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    print(f"Model output: '{response}'")
    
    # Check if correct
    if sample['expected'].lower() in response.lower():
        print("[CORRECT]")
        correct += 1
    else:
        print("[INCORRECT]")

print("\n" + "=" * 60)
print(f"SUMMARY: {correct}/{len(test_samples)} correct ({correct/len(test_samples)*100:.1f}%)")
print("=" * 60)

# Load and display training info if available
training_info_path = Path(model_path) / "training_info.json"
if training_info_path.exists():
    with open(training_info_path, 'r') as f:
        info = json.load(f)
    print("\nTraining Information:")
    print(f"- Base model: {info.get('base_model', 'N/A')}")
    print(f"- Training completed: {info.get('training_completed', 'N/A')}")
    print(f"- Duration: {info.get('training_duration', 'N/A')}")
    print(f"- Final loss: {info.get('final_loss', 'N/A')}")