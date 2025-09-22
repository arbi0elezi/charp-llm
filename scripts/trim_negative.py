import json
import sys

def remove_negative_entries(input_file, output_file):
    """
    Reads a JSONL dataset with fields {text, label} and writes only those lines
    which DO NOT end with '_negative'.
    """
    lines_in = 0
    lines_out = 0

    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            label = data.get("label", "")
            lines_in += 1

            # Keep only if label does not end in '_negative'
            if not label.endswith("_negative"):
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                lines_out += 1

    print(f"[INFO] Finished. Read {lines_in} lines, wrote {lines_out} (no '_negative').")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python remove_negatives.py <input.jsonl> <output.jsonl>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    remove_negative_entries(input_path, output_path)
