import os

# Prevent memory fragmentation and over-allocation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch

# Cap PyTorch to 70% of GPU memory
torch.cuda.set_per_process_memory_fraction(0.7, 0)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.set_float32_matmul_precision("high")

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType

# === Paths ===
MODEL_NAME = os.path.abspath("models/dcb")
OUTPUT_DIR = os.path.abspath("models/tff_gpu")
DATA_PATH = os.path.abspath("dataset/train/train_random_balanced.jsonl")
INSTRUCTION_TEMPLATE = """You are a code smell classifier. Given the following C# snippet, classify it.

Code:
{text}
In your answer strictly follow this format
FINAL ANSWER: followed by the label, example: FINAL ANSWER: ComplexMethod"""

# === Preprocessing ===
def preprocess_prompt_response(example):
    prompt = INSTRUCTION_TEMPLATE.replace("{text}", example["text"])
    response = f" {example['label'].strip()}"
    return {"prompt": prompt, "response": response}

def tokenize(example, tokenizer):
    full = example["prompt"] + example["response"]
    tokenized = tokenizer(
        full,
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    tokenized["labels"] = tokenized["input_ids"]
    return tokenized




# === Train Function ===
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model and LoRA
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False  # ✅ Required for gradient checkpointing

    # LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model = model.to(device)

    # Load dataset
    raw_dataset = load_dataset("json", data_files={"train": DATA_PATH})["train"]
    processed_dataset = raw_dataset.map(preprocess_prompt_response)
    tokenized_dataset = processed_dataset.map(
        lambda e: tokenize(e, tokenizer),
        remove_columns=processed_dataset.column_names
    )



    # Collator
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training config
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        learning_rate=5e-5,
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=20,
        optim="adamw_torch",
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=collator
    )

    def check_loss_graph():
        inputs = {
            k: torch.tensor(v).unsqueeze(0).to(device)
            for k, v in tokenized_dataset[0].items()
        }
        outputs = model(**inputs)
        print("[DEBUG] Loss requires grad:", outputs.loss.requires_grad)

    check_loss_graph()

    model.train()
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("[INFO] Training complete. Model saved.")

# Run it
train()
