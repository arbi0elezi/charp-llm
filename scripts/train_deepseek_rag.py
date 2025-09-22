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
import warnings
warnings.filterwarnings("ignore")

# === RTX A5000 Optimized Config ===
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.set_per_process_memory_fraction(0.8, 0)  # Increased for A5000 with 24GB VRAM
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
torch.set_float32_matmul_precision("high")

if not torch.cuda.is_available():
    raise RuntimeError("[ERROR] CUDA is not available. This script requires CUDA.")

device = torch.device("cuda")
print(f"[INFO] Using device: {device}")
print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# === Configuration ===
USE_TEST_DATA = True  # Set to False to use full training data
SAMPLE_SIZE = 50  # Number of samples to use for initial testing

# === Paths ===
MODEL_NAME = os.path.abspath("models/dclb")  # deepseek-coder-v2-lite-base
OUTPUT_DIR = os.path.abspath("models/deepseek_rag_finetuned")

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

print("[INFO] Initializing retriever...")
embedder, retriever_index, retriever_corpus = build_or_load_retriever()

# === Retrieval-Aware Preprocessing ===
def retrieve_context(text, k=2):
    """Retrieve k most similar code snippets"""
    query_emb = embedder.encode([text])[0]
    distances, indices = retriever_index.search(np.array([query_emb]), k)
    
    # Don't retrieve the exact same snippet
    contexts = []
    for i in indices[0]:
        if i < len(retriever_corpus) and retriever_corpus[i] != text:
            contexts.append(retriever_corpus[i])
    
    return "\n".join(contexts[:k])

INSTRUCTION_TEMPLATE = """You are a C# code smell detector. Analyze the following code and similar examples to identify the code smell.

Similar code examples:
{retrieved}

Code to analyze:
{text}

Classification: {label}"""

def preprocess_prompt_response(example):
    """Prepare prompt-response pairs for training"""
    retrieved = retrieve_context(example["text"])
    
    # For training, we include the label in the prompt
    prompt = INSTRUCTION_TEMPLATE.format(
        retrieved=retrieved[:500],  # Limit context length
        text=example["text"][:800],  # Limit code length
        label=""  # Will be filled by response
    )
    
    response = example['label'].strip()
    
    return {
        "prompt": prompt,
        "response": response,
        "full_text": prompt + response
    }

def tokenize(example, tokenizer):
    """Tokenize the full text for training"""
    tokenized = tokenizer(
        example["full_text"],
        truncation=True,
        max_length=1024,  # Increased for better context
        padding="max_length"
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# === Training Function ===
def train():
    # Check if model directory exists
    if not os.path.exists(MODEL_NAME):
        raise FileNotFoundError(f"[ERROR] Model directory does not exist: {MODEL_NAME}")

    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_fast=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # RTX A5000 can handle bf16
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[INFO] Using compute dtype: {compute_dtype}")

    # Optimized quantization for A5000
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    print("[INFO] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=compute_dtype
    )
    
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # LoRA configuration optimized for code understanding
    lora_config = LoraConfig(
        r=16,  # Increased rank for better learning
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target more modules
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    print("[INFO] Applying LoRA...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print("[INFO] Loading dataset...")
    raw_dataset = load_dataset("json", data_files={"train": DATA_PATH})["train"]
    
    # Limit samples for testing
    if USE_TEST_DATA and SAMPLE_SIZE:
        raw_dataset = raw_dataset.select(range(min(SAMPLE_SIZE, len(raw_dataset))))
        print(f"[INFO] Using {len(raw_dataset)} samples for testing")
    
    # Split for validation if using test data
    if USE_TEST_DATA:
        split = raw_dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"[INFO] Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    else:
        train_dataset = raw_dataset
        eval_dataset = None
    
    print("[INFO] Preprocessing dataset...")
    processed_train = train_dataset.map(
        preprocess_prompt_response,
        num_proc=4,
        desc="Preprocessing train"
    )
    
    print("[INFO] Tokenizing dataset...")
    tokenized_train = processed_train.map(
        lambda e: tokenize(e, tokenizer),
        remove_columns=processed_train.column_names,
        num_proc=4,
        desc="Tokenizing train"
    )
    
    if eval_dataset:
        processed_eval = eval_dataset.map(
            preprocess_prompt_response,
            num_proc=4,
            desc="Preprocessing eval"
        )
        tokenized_eval = processed_eval.map(
            lambda e: tokenize(e, tokenizer),
            remove_columns=processed_eval.column_names,
            num_proc=4,
            desc="Tokenizing eval"
        )
    else:
        tokenized_eval = None

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Training arguments optimized for RTX A5000
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3 if not USE_TEST_DATA else 5,  # More epochs for small test data
        per_device_train_batch_size=2,  # A5000 can handle larger batches
        per_device_eval_batch_size=4 if USE_TEST_DATA else 2,
        gradient_accumulation_steps=4,
        eval_strategy="steps" if USE_TEST_DATA else "no",
        eval_steps=10 if USE_TEST_DATA else None,
        save_strategy="steps",
        save_steps=50 if not USE_TEST_DATA else 20,
        save_total_limit=3,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=5 if USE_TEST_DATA else 20,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        fp16=False,  # Use bf16 instead
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
        load_best_model_at_end=USE_TEST_DATA,
        metric_for_best_model="loss" if USE_TEST_DATA else None,
        greater_is_better=False if USE_TEST_DATA else None,
    )

    print("[INFO] Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=collator
    )

    print("[INFO] Starting training...")
    print(f"[INFO] Total training samples: {len(tokenized_train)}")
    if tokenized_eval:
        print(f"[INFO] Total evaluation samples: {len(tokenized_eval)}")
    
    # Train the model
    trainer.train()
    
    # Save the model
    print("[INFO] Saving model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save training config
    config = {
        "base_model": MODEL_NAME,
        "training_data": DATA_PATH,
        "use_test_data": USE_TEST_DATA,
        "sample_size": SAMPLE_SIZE if USE_TEST_DATA else "full",
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "training_samples": len(tokenized_train)
    }
    
    with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"[INFO] Training complete! Model saved to {OUTPUT_DIR}")
    print("[INFO] To switch to full training data, set USE_TEST_DATA = False and run again")
    
    # Clear GPU memory
    torch.cuda.empty_cache()

# === Run ===
if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        torch.cuda.empty_cache()
        raise