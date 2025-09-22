import json
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    TrainerCallback,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

MODEL_NAME = os.path.abspath("models/dcb")
OUTPUT_DIR = os.path.abspath("models/flmco")

# Set device to CUDA for NVIDIA GPU (Windows)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

class EpochMetricsCallback(TrainerCallback):
    def __init__(self, output_file):
        self.epoch_data = []
        self.output_file = output_file

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            self.epoch_data.append({
                "epoch": state.epoch,
                "accuracy": metrics.get("eval_accuracy", 0),
                "precision": metrics.get("eval_precision", 0),
                "recall": metrics.get("eval_recall", 0),
                "f1": metrics.get("eval_f1", 0)
            })

    def on_train_end(self, args, state, control, **kwargs):
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(self.epoch_data, f, indent=2)

def tokenize_function(batch, tokenizer, label2id):
    tokenized = tokenizer(batch["text"], truncation=True, max_length=256)
    tokenized["label"] = [label2id[label.lower()] for label in batch["label"]]
    return tokenized

def train():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, local_files_only=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files={
        "train": os.path.abspath("dataset/train/train_c.jsonl"),
        "test": os.path.abspath("dataset/test/test_c.jsonl")
    })

    labels = sorted(set(label.lower() for label in dataset["train"]["label"]))
    label2id = {label: idx for idx, label in enumerate(labels)}

    tokenized_dataset = dataset.map(
        lambda batch: tokenize_function(batch, tokenizer, label2id), batched=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label2id),
        trust_remote_code=True,
        local_files_only=True
    ).to(device)

    model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,

        learning_rate=1e-4,
        num_train_epochs=10,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        weight_decay=0.01,

        fp16=True,
        optim="adamw_torch_fused",
        logging_dir="./logs",
        logging_strategy="epoch",
        report_to="none",
        no_cuda=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[
            EpochMetricsCallback(output_file=f"{OUTPUT_DIR}/epoch_metrics.json"),
            EarlyStoppingCallback(early_stopping_patience=2)
        ]
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)

    if device.type == "cuda":
        torch.cuda.empty_cache()

if __name__ == "__main__":
    train()
