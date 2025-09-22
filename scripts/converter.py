import os
import json

def convert_code_smells_to_jsonl(base_path, output_path):
    entries = []
    total_files = 0

    print(f"[INFO] Starting conversion from: {base_path}")
    if not os.path.exists(base_path):
        print(f"[ERROR] Base path does not exist: {base_path}")
        return

    for smell_type in os.listdir(base_path):
        smell_path = os.path.join(base_path, smell_type)
        if not os.path.isdir(smell_path):
            print(f"[WARN] Skipping non-folder entry: {smell_path}")
            continue
        print(f"[INFO] Processing smell type: {smell_type}")

        for label in ["positive", "negative"]:
            label_path = os.path.join(smell_path, label)
            if not os.path.isdir(label_path):
                print(f"[WARN] Missing expected folder: {label_path}")
                continue

            print(f"  - Reading {label} examples from: {label_path}")
            files = os.listdir(label_path)
            print(f"    > Found {len(files)} files")

            for i, filename in enumerate(files, 1):
                file_path = os.path.join(label_path, filename)
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        code = f.read()
                        entries.append({
                            "text": code,
                            "label": f"{smell_type}_{label}"
                        })
                        total_files += 1
                    if i % 50 == 0:
                        print(f"    ...processed {i} files so far in {label_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to read file: {file_path}\n  > {e}")

    print(f"[INFO] Finished processing. Total files collected: {total_files}")
    print(f"[INFO] Writing to JSONL file: {output_path}")

    try:
        with open(output_path, "w", encoding="utf-8") as f_out:
            for entry in entries:
                f_out.write(json.dumps(entry) + "\n")
        print(f"[SUCCESS] Written {len(entries)} entries to {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to write output file:\n  > {e}")

# Example usage:
convert_code_smells_to_jsonl("training_data_cs", "processed_data.jsonl")
