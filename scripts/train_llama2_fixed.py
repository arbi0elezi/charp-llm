"""
Fixed training script for Llama 2 7B on code smell detection dataset
Improved prompt format and training parameters
"""

import os
import sys
import torch
import json
from datetime import datetime
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import warnings
warnings.filterwarnings("ignore")

# Model configuration
MODEL_ID = "NousResearch/Llama-2-7b-hf"  # Community Llama 2, no auth needed
OUTPUT_DIR = "models/llama2_finetuned_v2"

# Better training configuration
TRAIN_CONFIG = {
    "learning_rate": 3e-4,  # Increased from 2e-4
    "num_train_epochs": 5,  # Increased from 3
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "warmup_ratio": 0.05,  # Reduced warmup
    "logging_steps": 10,
    "save_steps": 200,
    "eval_steps": 200,
    "max_grad_norm": 0.3,
    "fp16": True,
    "optim": "paged_adamw_8bit",
    "gradient_checkpointing": True,
}

# Enhanced LoRA configuration
LORA_CONFIG = {
    "r": 32,  # Increased from 16
    "lora_alpha": 64,  # Increased from 32
    "lora_dropout": 0.05,  # Reduced from 0.1
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "task_type": TaskType.CAUSAL_LM,
}

def load_dataset_file(file_path):
    """Load dataset from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            data.append(item)
    return data

def prepare_dataset(tokenizer):
    """Prepare training and validation datasets"""
    print("Loading datasets...")
    
    # Load balanced training data
    train_data = load_dataset_file("dataset/train/ds.jsonl")
    val_data = load_dataset_file("dataset/test/ds.jsonl")[:50]  # Use subset for validation
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Simple, clear format for training
    def format_example(example):
        code = example.get('code', example.get('text', ''))
        label = example['label']
        
        # Simplified format that's easier to learn
        prompt = f"""### Task: Identify the code smell in this C# code.

### Code:
```csharp
{code[:800]}  # Truncate long code
```

### Answer: {label}"""
        
        return {"text": prompt}
    
    # Convert to datasets
    train_texts = [format_example(ex) for ex in train_data]
    val_texts = [format_example(ex) for ex in val_data]
    
    # Print a few examples to verify format
    print("\nExample training prompts:")
    for i in range(min(2, len(train_texts))):
        print(f"\n--- Example {i+1} ---")
        print(train_texts[i]['text'][:500])
        print("...")
    
    train_dataset = Dataset.from_list(train_texts)
    val_dataset = Dataset.from_list(val_texts)
    
    # Tokenize datasets
    def tokenize_function(examples):
        # Add EOS token to help model learn when to stop
        texts = [text + tokenizer.eos_token for text in examples["text"]]
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=512,
        )
    
    print("\nTokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    return train_dataset, val_dataset

def main():
    print("=" * 60)
    print(f"Training Llama 2 7B for Code Smell Detection (Fixed)")
    print(f"Model: {MODEL_ID}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # Use right padding for training
    
    # Prepare datasets
    print("\n2. Preparing datasets...")
    train_dataset, val_dataset = prepare_dataset(tokenizer)
    
    # Configure 4-bit quantization
    print("\n3. Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        use_cache=False,
    )
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    print("\n4. Applying LoRA configuration...")
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"Total parameters: {total_params:,}")
    
    # Setup training arguments
    print("\n5. Setting up training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        **TRAIN_CONFIG,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        push_to_hub=False,
        report_to="none",
        logging_dir=f"{OUTPUT_DIR}/logs",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        save_total_limit=2,  # Keep only 2 best checkpoints
    )
    
    # Setup data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Calculate training steps
    total_steps = len(train_dataset) * TRAIN_CONFIG['num_train_epochs'] // (TRAIN_CONFIG['per_device_train_batch_size'] * TRAIN_CONFIG['gradient_accumulation_steps'])
    print(f"\n6. Starting training...")
    print(f"Total training steps: {total_steps}")
    print(f"Estimated time: {total_steps * 10 / 60:.1f} minutes (rough estimate)")
    
    start_time = datetime.now()
    
    # Train
    trainer.train()
    
    # Save final model
    print("\n7. Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save training info
    training_info = {
        "base_model": MODEL_ID,
        "training_completed": datetime.now().isoformat(),
        "training_duration": str(datetime.now() - start_time),
        "final_loss": trainer.state.best_metric,
        "total_steps": total_steps,
        "config": {
            "lora": LORA_CONFIG,
            "training": TRAIN_CONFIG
        }
    }
    
    with open(f"{OUTPUT_DIR}/training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    # Test the model quickly
    print("\n8. Quick test of trained model...")
    test_code = """
public void ProcessData(DataManager manager) {
    var data = manager.GetData();
    var processed = manager.ProcessData(data);
    manager.SaveData(processed);
}
"""
    
    test_prompt = f"""### Task: Identify the code smell in this C# code.

### Code:
```csharp
{test_code}
```

### Answer:"""
    
    inputs = tokenizer(test_prompt, return_tensors="pt", truncation=True, max_length=512)
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
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"Test response: '{response}'")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Training duration: {datetime.now() - start_time}")
    print("=" * 60)

if __name__ == "__main__":
    main()