"""
Training script for Llama 2 7B on code smell detection dataset
Using NousResearch/Llama-2-7b-hf (community version, no Meta approval needed)
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
OUTPUT_DIR = "models/llama2_finetuned"

# Training configuration optimized for RTX A5000 (24GB)
TRAIN_CONFIG = {
    "learning_rate": 2e-4,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,  # Small batch for 7B model
    "gradient_accumulation_steps": 16,  # Effective batch size = 16
    "warmup_ratio": 0.1,
    "logging_steps": 10,
    "save_steps": 100,
    "eval_steps": 100,
    "max_grad_norm": 0.3,
    "fp16": True,
    "optim": "paged_adamw_8bit",
    "gradient_checkpointing": True,  # Save memory
}

# LoRA configuration for Llama
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
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
    
    # Format data for Llama 2
    def format_example(example):
        code = example.get('code', example.get('text', ''))
        label = example['label']
        
        # Llama 2 instruction format
        prompt = f"""<s>[INST] <<SYS>>
You are a code smell classifier. Analyze C# code and identify the specific code smell present.
<</SYS>>

Classify this C# code snippet and identify the code smell:

```csharp
{code[:1000]}  # Truncate very long code
```

What code smell does this code exhibit? [/INST]

This code exhibits the {label} code smell.</s>"""
        
        return {"text": prompt}
    
    # Convert to datasets
    train_texts = [format_example(ex) for ex in train_data]
    val_texts = [format_example(ex) for ex in val_data]
    
    train_dataset = Dataset.from_list(train_texts)
    val_dataset = Dataset.from_list(val_texts)
    
    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )
    
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    return train_dataset, val_dataset

def main():
    print("=" * 60)
    print(f"Training Llama 2 7B for Code Smell Detection")
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
        tokenizer.padding_side = "left"  # Llama uses left padding
    
    # Prepare datasets
    print("\n2. Preparing datasets...")
    train_dataset, val_dataset = prepare_dataset(tokenizer)
    
    # Configure 4-bit quantization for memory efficiency
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
        use_cache=False,  # Disable for gradient checkpointing
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
        eval_strategy="steps",  # Changed from evaluation_strategy
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        push_to_hub=False,
        report_to="none",  # Disable tensorboard for now
        logging_dir=f"{OUTPUT_DIR}/logs",
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Windows compatibility
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
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Training duration: {datetime.now() - start_time}")
    print("=" * 60)

if __name__ == "__main__":
    main()