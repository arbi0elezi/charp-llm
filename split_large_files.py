#!/usr/bin/env python3
"""
Split large files into smaller chunks for GitHub upload.
GitHub has a 100MB file size limit.
"""

import os
import sys
import argparse
from pathlib import Path

def split_file(filepath, chunk_size_mb=95):
    """Split a file into smaller chunks."""
    chunk_size = chunk_size_mb * 1024 * 1024  # Convert to bytes
    filepath = Path(filepath)
    
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return False
    
    file_size = filepath.stat().st_size
    if file_size <= chunk_size:
        print(f"File {filepath} is already small enough ({file_size / 1024 / 1024:.2f} MB)")
        return True
    
    print(f"Splitting {filepath} ({file_size / 1024 / 1024:.2f} MB) into chunks...")
    
    # Create output directory
    output_dir = filepath.parent / f"{filepath.name}.parts"
    output_dir.mkdir(exist_ok=True)
    
    # Split the file
    with open(filepath, 'rb') as input_file:
        chunk_num = 0
        while True:
            chunk_data = input_file.read(chunk_size)
            if not chunk_data:
                break
            
            chunk_filename = output_dir / f"{filepath.name}.part{chunk_num:03d}"
            with open(chunk_filename, 'wb') as chunk_file:
                chunk_file.write(chunk_data)
            
            print(f"  Created: {chunk_filename} ({len(chunk_data) / 1024 / 1024:.2f} MB)")
            chunk_num += 1
    
    # Create a script to reassemble the file
    reassemble_script = output_dir / f"reassemble_{filepath.name}.py"
    with open(reassemble_script, 'w') as f:
        f.write(f'''#!/usr/bin/env python3
import os
from pathlib import Path

def reassemble():
    """Reassemble the split file."""
    parts_dir = Path(__file__).parent
    output_file = parts_dir.parent / "{filepath.name}"
    
    print(f"Reassembling {{output_file}}...")
    
    # Find all part files
    part_files = sorted(parts_dir.glob("{filepath.name}.part*"))
    
    with open(output_file, 'wb') as output:
        for part_file in part_files:
            print(f"  Processing {{part_file}}...")
            with open(part_file, 'rb') as part:
                output.write(part.read())
    
    print(f"File reassembled: {{output_file}}")
    print(f"Size: {{output_file.stat().st_size / 1024 / 1024:.2f}} MB")

if __name__ == "__main__":
    reassemble()
''')
    
    print(f"  Created reassembly script: {reassemble_script}")
    print(f"Successfully split {filepath} into {chunk_num} chunks")
    return True

def merge_file(parts_dir):
    """Merge split file parts back together."""
    parts_dir = Path(parts_dir)
    if not parts_dir.exists() or not parts_dir.is_dir():
        print(f"Parts directory not found: {parts_dir}")
        return False
    
    # Find part files
    part_files = sorted(parts_dir.glob("*.part*"))
    if not part_files:
        print(f"No part files found in {parts_dir}")
        return False
    
    # Determine output filename
    base_name = part_files[0].name.rsplit('.part', 1)[0]
    output_file = parts_dir.parent / base_name
    
    print(f"Merging {len(part_files)} parts into {output_file}...")
    
    with open(output_file, 'wb') as output:
        for part_file in part_files:
            print(f"  Processing {part_file}...")
            with open(part_file, 'rb') as part:
                output.write(part.read())
    
    print(f"File reassembled: {output_file}")
    print(f"Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    return True

def find_large_files(directory, min_size_mb=100):
    """Find all files larger than the specified size."""
    directory = Path(directory)
    large_files = []
    
    for root, dirs, files in os.walk(directory):
        # Skip .git and venv directories
        dirs[:] = [d for d in dirs if d not in ['.git', 'venv', '__pycache__']]
        
        for file in files:
            filepath = Path(root) / file
            if filepath.stat().st_size > min_size_mb * 1024 * 1024:
                large_files.append(filepath)
    
    return large_files

def main():
    parser = argparse.ArgumentParser(description='Split large files for GitHub upload')
    parser.add_argument('action', choices=['split', 'merge', 'find', 'split-all'],
                       help='Action to perform')
    parser.add_argument('path', nargs='?', default='.',
                       help='File to split/merge or directory to search')
    parser.add_argument('--chunk-size', type=int, default=95,
                       help='Chunk size in MB (default: 95)')
    parser.add_argument('--min-size', type=int, default=100,
                       help='Minimum file size in MB to consider large (default: 100)')
    
    args = parser.parse_args()
    
    if args.action == 'split':
        if args.path == '.':
            print("Please specify a file to split")
            sys.exit(1)
        success = split_file(args.path, args.chunk_size)
        sys.exit(0 if success else 1)
    
    elif args.action == 'merge':
        if args.path == '.':
            print("Please specify a parts directory to merge")
            sys.exit(1)
        success = merge_file(args.path)
        sys.exit(0 if success else 1)
    
    elif args.action == 'find':
        large_files = find_large_files(args.path, args.min_size)
        if large_files:
            print(f"Found {len(large_files)} large files (>{args.min_size}MB):")
            for file in large_files:
                size_mb = file.stat().st_size / 1024 / 1024
                print(f"  {file}: {size_mb:.2f} MB")
        else:
            print(f"No files larger than {args.min_size}MB found")
    
    elif args.action == 'split-all':
        large_files = find_large_files(args.path, args.min_size)
        if not large_files:
            print(f"No files larger than {args.min_size}MB found")
            sys.exit(0)
        
        print(f"Found {len(large_files)} large files to split")
        for file in large_files:
            split_file(file, args.chunk_size)

if __name__ == "__main__":
    main()