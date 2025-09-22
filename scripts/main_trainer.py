import os
import json
import torch
import argparse
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
from typing import List, Dict

def chunk_text(text: str, max_length: int = 512, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to split
        max_length: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
    """
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + max_length, len(text))
        
        # Try to break at a natural boundary (newline or space)
        if end < len(text):
            # Look for newline first
            newline_pos = text.rfind('\n', start + overlap, end)
            if newline_pos > start:
                end = newline_pos + 1
            else:
                # Look for space
                space_pos = text.rfind(' ', start + overlap, end)
                if space_pos > start:
                    end = space_pos + 1
        
        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else end
    
    return chunks

def prepare_dataset(data_path: str, tokenizer, max_token_length: int = 512):
    """
    Load and prepare dataset with chunking for long texts.
    """
    print(f"Loading data from: {data_path}")
    
    # Load JSONL data
    raw_data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                raw_data.append(json.loads(line))
    
    print(f"Loaded {len(raw_data)} samples")
    
    # Template for training
    template = """You are a code smell classifier. Given the following C# snippet, classify it.

Code:
{text}

In your answer strictly follow this format
FINAL ANSWER: followed by the label, example: FINAL ANSWER: ComplexMethod

FINAL ANSWER: {label}"""
    
    # Process and chunk data
    processed_data = []
    total_chunks = 0
    
    for item in raw_data:
        text = item['text']
        label = item['label']
        
        # Estimate token length (rough approximation)
        estimated_tokens = len(text) // 4  # Rough estimate: 1 token ≈ 4 chars
        
        if estimated_tokens > max_token_length:
            # Chunk the text
            char_limit = max_token_length * 3  # Conservative estimate
            chunks = chunk_text(text, max_length=char_limit, overlap=50)
            
            for chunk in chunks:
                formatted = template.format(text=chunk, label=label)
                processed_data.append({"text": formatted})
                total_chunks += 1
        else:
            # Use full text
            formatted = template.format(text=text, label=label)
            processed_data.append({"text": formatted})
            total_chunks += 1
    
    print(f"Created {total_chunks} training samples (with chunking)")
    print(f"Average chunks per original: {total_chunks / len(raw_data):.2f}")
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_token_length,
            padding="max_length"
        )
    
    # Create dataset
    dataset = Dataset.from_list(processed_data)
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset = tokenized_dataset.map(lambda x: {"labels": x["input_ids"]}, batched=True)
    
    return tokenized_dataset

def train_model(
    base_model_path: str,
    train_data_path: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 2e-4,
    save_steps: int = 500,
    use_lora: bool = True,
    model_type: str = "auto"
):
    """
    Generic training function with chunking support.
    
    Args:
        base_model_path: Path to base model
        train_data_path: Path to training data (JSONL)
        output_dir: Where to save the fine-tuned model
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        save_steps: Save checkpoint every N steps
        use_lora: Whether to use LoRA for efficient fine-tuning
    """
    
    # Detect model type
    if model_type == "auto":
        if "qwen" in base_model_path.lower():
            model_type = "Qwen"
        elif "deepseek" in base_model_path.lower():
            model_type = "DeepSeek"
        elif "llama" in base_model_path.lower():
            model_type = "Llama"
        else:
            model_type = "Unknown"
    
    print("=" * 80)
    print(f"MAIN TRAINER - {model_type.upper()} MODEL WITH CHUNKING")
    print("=" * 80)
    print(f"Base model: {base_model_path}")
    print(f"Training data: {train_data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Using LoRA: {use_lora}")
    print(f"Chunking: Enabled (better for long code samples)")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("Loading model...")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    if use_lora:
        # Load with quantization for LoRA
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=quant_config,
            trust_remote_code=True,
            device_map="auto"
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        # Full model fine-tuning
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=compute_dtype,
            trust_remote_code=True,
            device_map="auto"
        )
    
    model.config.use_cache = False
    
    # Prepare dataset with chunking
    print("\nPreparing dataset with chunking...")
    train_dataset = prepare_dataset(train_data_path, tokenizer, max_token_length=512)
    
    # Save training configuration
    training_info = {
        "base_model": base_model_path,
        "training_data": train_data_path,
        "output_dir": output_dir,
        "model_type": model_type,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "save_steps": save_steps,
        "use_lora": use_lora,
        "use_chunking": True,
        "chunk_max_length": 512,
        "chunk_overlap": 50,
        "total_samples": len(train_dataset),
        "timestamp": datetime.now().isoformat()
    }
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=20,
        optim="paged_adamw_8bit" if use_lora else "adamw_torch",
        gradient_checkpointing=False if use_lora else True,
        fp16=False,
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
        load_best_model_at_end=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # Save training info
    training_info = {
        "base_model": base_model_path,
        "training_data": train_data_path,
        "output_dir": output_dir,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "use_lora": use_lora,
        "total_samples": len(train_dataset),
        "timestamp": datetime.now().isoformat()
    }
    
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    # Train
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print(f"Total training samples (with chunks): {len(train_dataset)}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"Effective batch size: {batch_size * training_args.gradient_accumulation_steps}")
    
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\nTraining complete! Model saved to: {output_dir}")
    
    return trainer

def main():
    parser = argparse.ArgumentParser(description="Generic model training with text chunking")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model")
    parser.add_argument("--train_data", type=str, required=True, help="Path to training data (JSONL)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA (use full fine-tuning)")
    
    args = parser.parse_args()
    
    train_model(
        base_model_path=args.base_model,
        train_data_path=args.train_data,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        use_lora=not args.no_lora
    )

if __name__ == "__main__":
    main()