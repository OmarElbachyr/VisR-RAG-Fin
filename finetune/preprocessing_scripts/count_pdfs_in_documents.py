import os
import json

ORIGINAL_ROOT = 'finetune/original_documents'

def count_pdfs():
    original_pdfs = [f for f in os.listdir(ORIGINAL_ROOT) if os.path.isdir(os.path.join(ORIGINAL_ROOT, f))]
    total = 0
    for company_dir in original_pdfs:
        path = os.path.join(ORIGINAL_ROOT, company_dir)
        count = len([f for f in os.listdir(path) if f.lower().endswith('.pdf')])
        total += count
    print(f"Total PDFs in original_documents: {total}")

def check_visual_pdfs(json_path, original_root=ORIGINAL_ROOT):
    """Check that each unique `original_filename` in the JSON exists in original_documents folder.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load JSON: {e}")
        return None

    # Get unique filenames with their companies
    unique_files = {}
    for item in data:
        company_name = item.get('company')
        filename = item.get('original_filename')
        if company_name and filename:
            key = (company_name, filename)
            if key not in unique_files:
                unique_files[key] = True
    
    existing = 0
    missing = 0
    missing_files = []
    
    for (company_name, filename) in unique_files:
        file_path = os.path.join(original_root, company_name, filename)
        if os.path.exists(file_path):
            existing += 1
        else:
            missing += 1
            missing_files.append((company_name, filename))

    print(f"Unique existing files: {existing}")
    print(f"Unique missing files: {missing}")
    
    if missing_files:
        print(f"\nSample missing files:")
        for company, filename in missing_files[:5]:
            print(f"  {company}/{filename}")

if __name__ == "__main__":
    count_pdfs()
    # Example usage of check_visual_pdfs
    json_path = "data/annotations/final_annotations/merged_annotations/merged_annotations_filtered.json"  # Replace with the actual path to your JSON file
    check_visual_pdfs(json_path)
