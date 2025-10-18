import json
from pathlib import Path

# === Paths ===
input_dir = Path("data/annotations/final_annotations")
output_dir = input_dir / "merged_annotations"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "merged_annotations.json"

# === Merge ===
json_files = list(input_dir.glob("*.json"))
merged_data = []
total_count = 0

print("Merging JSON files...\n")

for json_file in json_files:
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                count = len(data)
                merged_data.extend(data)
                total_count += count
                print(f"{json_file.name}: {count} elements")
            else:
                print(f"⚠️ Skipping {json_file.name}: not a list")
    except Exception as e:
        print(f"⚠️ Error reading {json_file.name}: {e}")

print(f"\nTotal merged elements: {total_count}")
print(f"Total files processed: {len(json_files)}")

# === Save merged file ===
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)

print(f"\n✅ Merged annotations saved to: {output_path}")