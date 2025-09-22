import json
import os

def minify_text(text: str) -> str:
    # Remove newlines, tabs, and collapse multiple spaces
    return ' '.join(text.replace('\n', ' ').replace('\t', ' ').split())

def minify_jsonl(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            data = json.loads(line)
            data['text'] = minify_text(data['text'])
            fout.write(json.dumps(data) + '\n')

if __name__ == "__main__":
    input_file = os.path.abspath("dataset/test/test_c.jsonl")  
    output_file = os.path.abspath("dataset/test/test_min.jsonl")
    minify_jsonl(input_file, output_file)
    print(f"[DONE] Minified JSONL saved to: {output_file}")
