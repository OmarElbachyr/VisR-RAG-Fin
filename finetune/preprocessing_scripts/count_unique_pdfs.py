import json

# Path to the JSON file
json_path = 'data/annotations/final_annotations/merged_annotations/merged_annotations.json'

def count_unique_pdfs(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    unique_filenames = set()
    for entry in data:
        if 'hashed_filename' in entry:
            unique_filenames.add(entry['hashed_filename'])
    print(f"Unique PDFs: {len(unique_filenames)}")
    return unique_filenames

import os

def check_pdfs_exist(json_path, pdfs_folder):
    unique_filenames = count_unique_pdfs(json_path)
    missing = []
    for pdf in unique_filenames:
        if not os.path.exists(os.path.join(pdfs_folder, pdf)):
            missing.append(pdf)
    if missing:
        print(f"Missing PDFs ({len(missing)}):")
        for m in missing:
            print(m)
    else:
        print("All referenced PDFs exist in the folder.")

if __name__ == "__main__":
    # Count unique PDFs
    count_unique_pdfs(json_path)
    # Check if all PDFs exist in the folder
    check_pdfs_exist(json_path, 'data/indexed_pdfs')
