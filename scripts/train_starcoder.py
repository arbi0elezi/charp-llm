"""
Training script for StarCoder2-3B model on code smell detection dataset
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
MODEL_ID = "bigcode/starcoder2-3b"
OUTPUT_DIR = "models/starcoder2_finetuned"

# Training configuration
TRAIN_CONFIG = {
    "learning_rate": 2e-4,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "warmup_ratio": 0.1,
    "logging_steps": 10,
    "save_steps": 100,
    "eval_steps": 100,
    "max_grad_norm": 0.3,
    "fp16": True,
    "optim": "paged_adamw_8bit",
}

# LoRA configuration
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
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
    
    # Load balanced training data - using ds.jsonl which is the balanced dataset
    train_data = load_dataset_file("dataset/train/ds.jsonl")
    # Use test data for validation since no validation set exists
    val_data = load_dataset_file("dataset/test/ds.jsonl")
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Format data for training
    def format_example(example):
        code = example.get('code', example.get('text', ''))
        label = example['label']
        
        # Create instruction-based prompt
        prompt = f"""<|system|>
You are a code smell classifier. Analyze the given C# code and identify the code smell.
<|user|>
Classify this C# code snippet:

```csharp
{code[:1500]}  # Truncate very long code
```

What code smell does this exhibit?
<|assistant|>
This code exhibits the {label} code smell.<|endoftext|>"""
        
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
    print(f"Training StarCoder2-3B for Code Smell Detection")
    print(f"Model: {MODEL_ID}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, training will be slow!")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
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
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    
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
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        push_to_hub=False,
        report_to=["tensorboard"],
        logging_dir=f"{OUTPUT_DIR}/logs",
        remove_unused_columns=False,
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
    
    # Save initial checkpoint
    print("\n6. Starting training...")
    print(f"Total training steps: {len(train_dataset) * TRAIN_CONFIG['num_train_epochs'] // (TRAIN_CONFIG['per_device_train_batch_size'] * TRAIN_CONFIG['gradient_accumulation_steps'])}")
    
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