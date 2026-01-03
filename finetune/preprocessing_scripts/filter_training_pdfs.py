import os
import json
import shutil

SRC_ROOT = 'finetune/original_documents'
DST_ROOT = 'finetune/training_documents'


def filter_pdfs(json_path, src_root=SRC_ROOT, dst_root=DST_ROOT):
    """Filter PDFs by excluding those listed in the JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load JSON: {e}")
        return

    # Create exclude set with (company, filename) tuples
    exclude_files = {(item.get('company'), item.get('original_filename')) for item in data if item.get('company') and item.get('original_filename')}

    if not os.path.exists(dst_root):
        os.makedirs(dst_root)

    # Get company folders
    company_folders = [d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))]

    pdf_count = 0
    for company in company_folders:
        src_folder = os.path.join(src_root, company)

        for filename in os.listdir(src_folder):
            if filename.lower().endswith('.pdf') and (company, filename) not in exclude_files:
                src_file = os.path.join(src_folder, filename)
                dst_file = os.path.join(dst_root, filename)
                shutil.copy2(src_file, dst_file)
                pdf_count += 1

    print(f"Total PDFs copied: {pdf_count}")
    print(f"Filtered PDFs copied to {dst_root}")


if __name__ == "__main__":
    json_path = "data/annotations/final_annotations/merged_annotations/merged_annotations_filtered.json"  # Replace with the actual path to your JSON file
    filter_pdfs(json_path)