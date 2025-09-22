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

def prepare_dataset(data_path: str, tokenizer, max_token_length: int = 1024):
    """
    Load and prepare dataset for Llama training.
    Using the same balanced dataset as DeepSeek.
    """
    print(f"Loading data from: {data_path}")
    
    # Load JSONL data
    raw_data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                raw_data.append(json.loads(line))
    
    print(f"Loaded {len(raw_data)} samples")
    
    # Llama 3.1 chat template
    template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a code smell classifier. Analyze C# code and classify it as one of: ComplexMethod, ComplexConditional, or FeatureEnvy.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Classify this C# code:

{text}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
{label}<|eot_id|>"""
    
    # Process data
    processed_data = []
    for item in raw_data:
        text = item['text']
        label = item['label']
        
        # Truncate if too long (same as DeepSeek)
        if len(text) > 2000:
            text = text[:2000] + "..."
        
        formatted = template.format(text=text, label=label)
        processed_data.append({"text": formatted})
    
    print(f"Processed {len(processed_data)} training samples")
    
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

def train_llama_model(
    base_model_path: str = "models/llama_8b_baseline",
    train_data_path: str = "dataset/train/ds.jsonl",
    output_dir: str = "models/llama_8b_finetuned",
    num_epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 2e-4
):
    """
    Train Llama 8B using the exact same strategy as DeepSeek balanced model.
    
    Args:
        base_model_path: HuggingFace model ID or local path
        train_data_path: Path to balanced training data (ds.jsonl)
        output_dir: Where to save the fine-tuned model
        num_epochs: Number of training epochs (same as DeepSeek: 3)
        batch_size: Training batch size (same as DeepSeek: 1)
        learning_rate: Learning rate (same as DeepSeek: 2e-4)
    """
    
    print("=" * 80)
    print("LLAMA 8B BALANCED TRAINING")
    print("=" * 80)
    print(f"Base model: {base_model_path}")
    print(f"Training data: {train_data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    
    # Check if using local model or HuggingFace model
    token = None
    if base_model_path.startswith("meta-llama/"):
        # Only need token for downloading from HuggingFace
        token = os.getenv("HF_TOKEN")
        if not token:
            print("\n[WARNING] Using HuggingFace model but HF_TOKEN not set.")
            print("If you haven't downloaded the model yet, you'll need authentication.")
            print("To use a local model, provide a local path instead.")
    else:
        print(f"\n[INFO] Using local model from: {base_model_path}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Set memory fraction (same as DeepSeek)
        torch.cuda.set_per_process_memory_fraction(0.9, 0)
        torch.cuda.empty_cache()
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, 
        trust_remote_code=True,
        token=token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load model with 4-bit quantization (same as DeepSeek)
    print("Loading model with 4-bit quantization...")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
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
        device_map="auto",
        token=token,
        torch_dtype=compute_dtype,
        attn_implementation="eager"  # Avoid Flash Attention issues on Windows
    )
    
    # Configure LoRA (same settings as DeepSeek)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Llama attention modules
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    print("\nApplying LoRA...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Disable cache for training
    model.config.use_cache = False
    
    # Prepare dataset
    print("\nPreparing dataset...")
    train_dataset = prepare_dataset(train_data_path, tokenizer)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments (same as DeepSeek balanced model)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,  # Effective batch size: 4
        warmup_steps=100,
        learning_rate=learning_rate,
        fp16=False,
        bf16=compute_dtype == torch.bfloat16,
        logging_steps=50,
        save_steps=500,
        eval_strategy="no",  # Same as DeepSeek - no evaluation during training
        save_total_limit=3,
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
        report_to="none",
        optim="adamw_torch",
        weight_decay=0.01,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Save training info
    training_info = {
        "base_model": base_model_path,
        "training_data": train_data_path,
        "output_dir": output_dir,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "use_lora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "total_samples": len(train_dataset),
        "timestamp": datetime.now().isoformat()
    }
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "training_info.json"), "w") as f:
        json.dump(training_info, f, indent=2)
    
    # Train
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    
    trainer.train()
    
    # Save model
    print("\nSaving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n✓ Training complete! Model saved to: {output_dir}")
    print("\nTo evaluate the model, run:")
    print(f"python scripts/eval_llama.py finetuned {output_dir}")
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description="Train Llama 8B on balanced dataset")
    parser.add_argument("--base_model", type=str, default="models/llama_8b_baseline",
                        help="Base model path (local)")
    parser.add_argument("--train_data", type=str, default="dataset/train/ds.jsonl",
                        help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="models/llama_8b_finetuned",
                        help="Output directory for fine-tuned model")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    
    args = parser.parse_args()
    
    train_llama_model(
        base_model_path=args.base_model,
        train_data_path=args.train_data,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )

if __name__ == "__main__":
    main()