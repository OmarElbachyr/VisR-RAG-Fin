import json

# --- script to delete the questions that are found in failed results.txts ---


def load_image_filenames(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    image_filenames = {line.split('\t')[0].strip() for line in lines if line.strip()}
    return image_filenames

def filter_json(json_path, output_path, images_to_remove):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    def is_image_in_list(image_path):
        return any(image_path.endswith(img) for img in images_to_remove)

    filtered_data = [
        entry for entry in data
        if not is_image_in_list(entry.get('image_filename', ''))
    ]

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)

    print(f"Filtered JSON saved to: {output_path}")
    print(f"Removed {len(data) - len(filtered_data)} entries.")


if __name__ == "__main__":
    txt_file = "/home/laura/vqa-ir-qa/data/qwen_failed_cases.txt"          # Replace with your actual TXT file path
    input_json = "/home/laura/vqa-ir-qa/data/ls_single_page_visual_onpoint_gpt-4o-2024-08-06.json"      # Replace with your full input JSON path
    output_json = "filtered_data.json"  # Output JSON path

    images_to_remove = load_image_filenames(txt_file)
    filter_json(input_json, output_json, images_to_remove)
