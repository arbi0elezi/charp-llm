import os
import json
import torch
import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType

# === Memory + Device Config ===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.set_per_process_memory_fraction(0.7, 0)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.set_float32_matmul_precision("high")

if not torch.cuda.is_available():
    raise RuntimeError("[ERROR] CUDA is not available. This script requires CUDA.")

device = torch.device("cuda")
print(f"[INFO] Using device: {device}")

# === Paths ===
MODEL_NAME = os.path.abspath("models/dclb")
OUTPUT_DIR = os.path.abspath("models/tff_rag")
DATA_PATH = os.path.abspath("dataset/train/tmf.jsonl")
CORPUS_PATH = os.path.abspath("dataset/train/tmf.jsonl")
FAISS_INDEX_PATH = os.path.abspath("rag_index.faiss")  # Absolute path in current directory
FAISS_CORPUS_JSON = os.path.abspath("rag_corpus.json")  # Absolute path in current directory

# === Load/Build Retriever ===
def build_or_load_retriever():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"[INFO] FAISS index path: {FAISS_INDEX_PATH}")
    print(f"[INFO] FAISS corpus path: {FAISS_CORPUS_JSON}")

    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_CORPUS_JSON):
        print("[INFO] Loading existing FAISS index and corpus")
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_CORPUS_JSON, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        print(f"[INFO] Loaded index with {index.ntotal} vectors")
    else:
        print("[INFO] Building new FAISS index from corpus")
        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            corpus_data = [json.loads(line.strip()) for line in f]
            corpus = [item["text"] for item in corpus_data]
        embeddings = embedder.encode(corpus, convert_to_numpy=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        print(f"[INFO] Built index with {index.ntotal} vectors")
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(FAISS_CORPUS_JSON, "w", encoding="utf-8") as f:
            json.dump(corpus, f)

    return embedder, index, corpus

embedder, retriever_index, retriever_corpus = build_or_load_retriever()

# === Retrieval-Aware Preprocessing ===
def retrieve_context(text, k=2):
    query_emb = embedder.encode([text])[0]
    _, I = retriever_index.search(np.array([query_emb]), k)
    return "\n".join([retriever_corpus[i] for i in I[0]])

INSTRUCTION_TEMPLATE = """You are a code smell classifier. Given the following C# snippet, classify it.

Relevant Info:
{retrieved}

Code:
{text}

In your answer strictly follow this format
FINAL ANSWER: followed by the label, example: FINAL ANSWER: ComplexMethod"""

def preprocess_prompt_response(example):
    retrieved = retrieve_context(example["text"])
    prompt = INSTRUCTION_TEMPLATE.replace("{retrieved}", retrieved).replace("{text}", example["text"])
    response = f" {example['label'].strip()}"
    return {"prompt": prompt, "response": response}

def tokenize(example, tokenizer):
    full = example["prompt"] + example["response"]
    tokenized = tokenizer(full, truncation=True, max_length=512, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"]
    return tokenized

# === Training Function ===
def train():
    # Check if model directory exists; throw error if not
    if not os.path.exists(MODEL_NAME):
        raise FileNotFoundError(f"[ERROR] Model directory does not exist: {MODEL_NAME}. Please ensure the model is available locally.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        trust_remote_code=True,
        device_map="auto"
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

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

    raw_dataset = load_dataset("json", data_files={"train": DATA_PATH})["train"]
    processed_dataset = raw_dataset.map(preprocess_prompt_response)
    tokenized_dataset = processed_dataset.map(lambda e: tokenize(e, tokenizer), remove_columns=processed_dataset.column_names)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=collator
    )

    model.train()
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("[INFO] Training complete. Model saved.")

# === Run ===
if __name__ == "__main__":
    train()