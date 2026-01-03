import os
import shutil

SRC_ROOT = 'finetune/filtered/scraped_pdfs'
DST_ROOT = 'finetune/documents'
STRUCTURED = 'structured_documents'
VISUAL = 'visual_documents'

# Ensure destination folders exist
def ensure_dirs():
    os.makedirs(os.path.join(DST_ROOT, STRUCTURED), exist_ok=True)
    os.makedirs(os.path.join(DST_ROOT, VISUAL), exist_ok=True)

def copy_pdfs():
    ensure_dirs()
    for company in os.listdir(SRC_ROOT):
        company_path = os.path.join(SRC_ROOT, company)
        if not os.path.isdir(company_path):
            continue
        for doc_type in [STRUCTURED, VISUAL]:
            src_folder = os.path.join(company_path, doc_type)
            if not os.path.isdir(src_folder):
                continue
            for fname in os.listdir(src_folder):
                src_file = os.path.join(src_folder, fname)
                if not os.path.isfile(src_file):
                    continue
                dst_file = os.path.join(DST_ROOT, doc_type, fname)
                shutil.copy2(src_file, dst_file)
                print(f"Copied {src_file} -> {dst_file}")

if __name__ == "__main__":
    copy_pdfs()
