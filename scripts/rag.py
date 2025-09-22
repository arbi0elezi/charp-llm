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

# === CONFIGURATION - CHANGE THIS TO SWITCH BETWEEN TEST AND TRAIN ===
USE_TEST_DATA = False  # Set to False to use full training data
SAMPLE_SIZE = 50  # Number of samples to use for initial testing

# === Memory + Device Config (Optimized for RTX A5000) ===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.set_per_process_memory_fraction(0.7, 0)  # Balanced for stability
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.set_float32_matmul_precision("high")

if not torch.cuda.is_available():
    raise RuntimeError("[ERROR] CUDA is not available. This script requires CUDA.")

device = torch.device("cuda")
print(f"[INFO] Using device: {device}")
print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# === Paths ===
MODEL_NAME = os.path.abspath("models/dclb")
OUTPUT_DIR = os.path.abspath("models/tff_rag")

# Select dataset based on configuration
if USE_TEST_DATA:
    DATA_PATH = os.path.abspath("dataset/test/tmf.jsonl")
    CORPUS_PATH = os.path.abspath("dataset/test/tmf.jsonl")
    print(f"[INFO] Using TEST data for verification: {DATA_PATH}")
else:
    DATA_PATH = os.path.abspath("dataset/train/tmf.jsonl")
    CORPUS_PATH = os.path.abspath("dataset/train/tmf.jsonl")
    print(f"[INFO] Using TRAIN data for full training: {DATA_PATH}")

FAISS_INDEX_PATH = os.path.abspath("rag_index_test.faiss" if USE_TEST_DATA else "rag_index.faiss")
FAISS_CORPUS_JSON = os.path.abspath("rag_corpus_test.json" if USE_TEST_DATA else "rag_corpus.json")

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
            
            # Limit samples for testing
            if USE_TEST_DATA and SAMPLE_SIZE:
                corpus_data = corpus_data[:SAMPLE_SIZE]
                print(f"[INFO] Limited to {len(corpus_data)} samples for testing")
            
            corpus = [item["text"] for item in corpus_data]
        embeddings = embedder.encode(corpus, convert_to_numpy=True, show_progress_bar=True)
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
    distances, indices = retriever_index.search(np.array([query_emb]), k)
    
    # Don't retrieve the exact same snippet
    contexts = []
    for i in indices[0]:
        if i < len(retriever_corpus) and retriever_corpus[i] != text:
            contexts.append(retriever_corpus[i])
    
    return "\n".join(contexts[:k])

INSTRUCTION_TEMPLATE = """You are a code smell classifier. Given the following C# snippet, classify it.

Relevant Info:
{retrieved}

Code:
{text}

In your answer strictly follow this format
FINAL ANSWER: followed by the label, example: FINAL ANSWER: ComplexMethod"""

def preprocess_prompt_response(example):
    retrieved = retrieve_context(example["text"])
    # Limit context to prevent OOM
    prompt = INSTRUCTION_TEMPLATE.replace("{retrieved}", retrieved[:500]).replace("{text}", example["text"][:800])
    response = f" {example['label'].strip()}"
    return {"prompt": prompt, "response": response}

def tokenize(example, tokenizer):
    full = example["prompt"] + example["response"]
    tokenized = tokenizer(full, truncation=True, max_length=512, padding="max_length")  # Reduced for memory
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
    # Disable gradient checkpointing for 4-bit models
    # model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=16,  # Increased rank for better learning
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target more modules
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    raw_dataset = load_dataset("json", data_files={"train": DATA_PATH})["train"]
    
    # Limit samples for testing
    if USE_TEST_DATA and SAMPLE_SIZE:
        raw_dataset = raw_dataset.select(range(min(SAMPLE_SIZE, len(raw_dataset))))
        print(f"[INFO] Using {len(raw_dataset)} samples for testing")
    
    processed_dataset = raw_dataset.map(preprocess_prompt_response, num_proc=4)
    tokenized_dataset = processed_dataset.map(lambda e: tokenize(e, tokenizer), remove_columns=processed_dataset.column_names, num_proc=4)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5 if USE_TEST_DATA else 3,  # More epochs for small test data
        per_device_train_batch_size=1,  # Reduced for memory
        gradient_accumulation_steps=8,  # Increased to compensate
        eval_strategy="no",
        save_strategy="steps",
        save_steps=20 if USE_TEST_DATA else 500,
        save_total_limit=3,
        learning_rate=2e-4,  # Higher learning rate for LoRA
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=5 if USE_TEST_DATA else 20,
        optim="paged_adamw_8bit",
        gradient_checkpointing=False,  # Disabled for 4-bit models
        fp16=False,  # Use bf16 instead
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=collator
    )

    print(f"[INFO] Starting training...")
    print(f"[INFO] Total training samples: {len(tokenized_dataset)}")
    print(f"[INFO] Configuration: {'TEST' if USE_TEST_DATA else 'TRAIN'} data")
    
    model.train()
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save config info
    config = {
        "base_model": MODEL_NAME,
        "training_data": DATA_PATH,
        "use_test_data": USE_TEST_DATA,
        "sample_size": SAMPLE_SIZE if USE_TEST_DATA else "full",
        "training_samples": len(tokenized_dataset)
    }
    with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"[INFO] Training complete. Model saved to {OUTPUT_DIR}")
    print("[INFO] To switch to full training, set USE_TEST_DATA = False at the top of the script")
    
    # Clear GPU memory
    torch.cuda.empty_cache()

# === Run ===
if __name__ == "__main__":
    train()