import json

input_path = "/home/laura/vqa-ir-qa/data/output_internvl3_1b_answers_annotated.jsonl"
output_path = "/home/laura/vqa-ir-qa/data/output_internvl3_1b_answers_annotated.json"

data = []
with open(input_path, "r", encoding="utf-8") as infile:
    for i, line in enumerate(infile, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Skipping invalid JSON on line {i}: {e}")

with open(output_path, "w", encoding="utf-8") as outfile:
    json.dump(data, outfile, indent=4, ensure_ascii=False)

print(f"Converted {len(data)} records to {output_path}")
