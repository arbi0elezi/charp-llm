#!/usr/bin/env python3
import os
from pathlib import Path

def reassemble():
    """Reassemble the split file."""
    parts_dir = Path(__file__).parent
    output_file = parts_dir.parent / "train_positive.jsonl"
    
    print(f"Reassembling {output_file}...")
    
    # Find all part files
    part_files = sorted(parts_dir.glob("train_positive.jsonl.part*"))
    
    with open(output_file, 'wb') as output:
        for part_file in part_files:
            print(f"  Processing {part_file}...")
            with open(part_file, 'rb') as part:
                output.write(part.read())
    
    print(f"File reassembled: {output_file}")
    print(f"Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    reassemble()
