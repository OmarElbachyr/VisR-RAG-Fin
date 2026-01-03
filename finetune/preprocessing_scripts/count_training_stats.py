import os
import fitz  # PyMuPDF
import numpy as np
import json

TRAINING_ROOT = 'finetune/training_documents'
VISUAL_METADATA = 'finetune/visual_metadata.json'

def count_pages_in_pdf(pdf_path):
    """Count the number of pages in a PDF file."""
    try:
        with fitz.open(pdf_path) as doc:
            return len(doc)
    except Exception as e:
        print(f"[ERROR] Failed to read {pdf_path}: {e}")
        return 0

def count_training_stats(training_root=TRAINING_ROOT):
    """Count the number of PDFs and pages in training documents."""
    pdf_count = 0
    page_count = 0
    page_counts = []

    print("\nTraining Statistics:\n--------------------")

    for filename in os.listdir(training_root):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(training_root, filename)
            pages = count_pages_in_pdf(pdf_path)
            pdf_count += 1
            page_count += pages
            page_counts.append(pages)

    quartiles = np.percentile(page_counts, [25, 50, 75]) if page_counts else [0, 0, 0]

    print(f"Total PDFs: {pdf_count}")
    print(f"Total Pages: {page_count}")
    print(f"Page Count Distribution (Q1, Q2, Q3): {quartiles}\n")

def count_visual_pages(metadata_path=VISUAL_METADATA):
    """Count visual and non-visual pages from metadata."""
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load metadata: {e}")
        return

    total_visual_count = 0
    total_non_visual_count = 0

    for item in data:
        total_visual_count += item.get('visual_pages', 0)
        total_non_visual_count += item.get('non_visual_pages', 0)

    print("Visual Page Statistics:")
    print(f"  Visual Pages: {total_visual_count}")
    print(f"  Non-Visual Pages: {total_non_visual_count}\n")

if __name__ == "__main__":
    count_training_stats()
    count_visual_pages()