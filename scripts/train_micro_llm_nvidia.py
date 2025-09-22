# 🚨 Set these BEFORE any torch or transformers imports
import os
# Configure CUDA memory management
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_score

MODEL_NAME = "../models/deepseek-coder-6.7b-base"
OUTPUT_DIR = "./finetuned-lora-model"

# Set up device - prioritize CUDA (NVIDIA GPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU (no GPU available)")

# 📊 Metric computation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="macro", zero_division=0)
    }

# 📈 Track epoch metrics
class EpochMetricsCallback(TrainerCallback):
    def __init__(self, output_file):
        self.epoch_data = []
        self.output_file = output_file

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self.epoch_data.append({
                "epoch": state.epoch,
                "accuracy": metrics.get("eval_accuracy", 0),
                "precision": metrics.get("eval_precision", 0)
            })
            with open(self.output_file, "w") as f:
                json.dump(self.epoch_data, f, indent=2)

# 📦 Tokenize and encode
def load_data(train_path, test_path, tokenizer):
    dataset = load_dataset("json", data_files={"train": train_path, "test": test_path})
    label_list = sorted(set(example["label"] for example in dataset["train"]))
    label2id = {label: i for i, label in enumerate(label_list)}

    def tokenize(batch):
        combined = [text + " " + label for text, label in zip(batch["text"], batch["label"])]
        tokenized = tokenizer(
            combined,
            truncation=True,
            padding="max_length",
            max_length=128,  # Can use longer sequences with GPU
        )
        tokenized["label"] = [label2id[label] for label in batch["label"]]
        return tokenized

    return dataset.map(tokenize, batched=True), label2id

# 🧠 TRAIN
def train():
    # Clean CUDA memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenized, label2id = load_data(
        "../dataset/train/train.jsonl", 
        "../dataset/test/test.jsonl",
        tokenizer
    )

    # Check for bf16 support (A100, A6000, RTX 30xx, etc.)
    bf16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    
    # Check for mixed precision support
    fp16_supported = torch.cuda.is_available()

    # Load model with appropriate precision settings
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16_supported else (torch.float16 if fp16_supported else torch.float32)
    )
    
    # Move model to GPU
    model = model.to(device)

    # 🧠 LoRA adapters
    lora_config = LoraConfig(
        r=8,  # Can use larger rank with GPU
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )
    model = get_peft_model(model, lora_config)

    # 💾 Enable gradient checkpointing to save GPU memory
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    # Configure appropriate batch sizes based on device
    train_batch_size = 4 if device.type == "cuda" else 1
    eval_batch_size = 4 if device.type == "cuda" else 1
    grad_accum_steps = 16 if device.type == "cuda" else 128

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-4,
        num_train_epochs=3,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=10,
        report_to="tensorboard",
        # Precision settings
        bf16=bf16_supported,  # Use bfloat16 if available
        fp16=fp16_supported and not bf16_supported,  # Otherwise use fp16 if available
        # Use better optimizer for GPU
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        label_names=["label"],
        # CUDA settings
        no_cuda=not torch.cuda.is_available(),
        dataloader_num_workers=4 if device.type == "cuda" else 0,  # Parallel loading with GPU
    )

    from transformers import set_seed
    set_seed(42)  # Set seed for reproducibility
    
    import gc
    gc.collect()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EpochMetricsCallback(output_file=f"{OUTPUT_DIR}/epoch_metrics.json")]
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    train()