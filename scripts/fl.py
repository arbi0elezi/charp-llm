import json
import os
import statistics

INPUT_FILE = os.path.abspath("dataset/test/test_min.jsonl")
OUTPUT_FILE = os.path.abspath("dataset/test/test_min_filtered.jsonl")
TEXT_KEY = "text"
MAX_ALLOWED_LENGTH = 10_000

def find_text_stats_without_outliers(file_path, text_key="text", max_length_threshold=MAX_ALLOWED_LENGTH):
    max_length = 0
    max_sample = None
    lengths = []
    filtered_entries = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            try:
                entry = json.loads(line)
                text = entry.get(text_key, "")
                length = len(text)
                if length <= max_length_threshold:
                    filtered_entries.append(entry)
                    lengths.append(length)
                    if length > max_length:
                        max_length = length
                        max_sample = text
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {line_num}: {e}")

    # Save the filtered dataset
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in filtered_entries:
            f.write(json.dumps(entry) + "\n")

    median_length = statistics.median(lengths)
    average_length = statistics.mean(lengths)

    print(f"[INFO] Filtered entries: {len(filtered_entries)}")
    print(f"Max `{text_key}` length (<= {max_length_threshold}): {max_length} characters")
    print(f"Median `{text_key}` length: {median_length:.2f} characters")
    print(f"Average `{text_key}` length: {average_length:.2f} characters")

    return max_length, median_length, average_length, max_sample

if __name__ == "__main__":
    max_len, median_len, avg_len, sample = find_text_stats_without_outliers(INPUT_FILE, TEXT_KEY)
    print("\nExample of the longest filtered text entry:\n")
    print(sample[:500] + ("..." if len(sample) > 500 else ""))
