#!/bin/bash

# Example 1: Train with LoRA (efficient, recommended)
python scripts/train_generic_chunked.py \
    --base_model models/dclb \
    --train_data dataset/train/tmf.jsonl \
    --output_dir models/tff_chunked \
    --epochs 3 \
    --batch_size 1 \
    --learning_rate 2e-4 \
    --save_steps 500

# Example 2: Train without LoRA (full fine-tuning, requires more memory)
# python scripts/train_generic_chunked.py \
#     --base_model models/dclb \
#     --train_data dataset/train/tmf.jsonl \
#     --output_dir models/tff_full \
#     --epochs 3 \
#     --batch_size 1 \
#     --learning_rate 5e-5 \
#     --save_steps 500 \
#     --no_lora

# Example 3: Quick test with balanced test data
# python scripts/train_generic_chunked.py \
#     --base_model models/dclb \
#     --train_data dataset/test/tmf.jsonl \
#     --output_dir models/test_balanced \
#     --epochs 1 \
#     --batch_size 2 \
#     --learning_rate 2e-4 \
#     --save_steps 50