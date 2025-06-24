import os
import fitz
from PIL import Image

PDF_DIR = "/home/laura/vqa-ir-qa/data/indexed_pdfs"
OUTPUT_IMAGE_DIR = "/home/laura/vqa-ir-qa/data/indexed_pages"

def page_pixmap_to_image(page):
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    mode = "RGBA" if pix.alpha else "RGB"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    if mode == "RGBA":
        img = img.convert("RGB")
    return img

for root, _, files in os.walk(PDF_DIR):
    for fname in files:
        if not fname.lower().endswith(".pdf"):
            continue
        pdf_path = os.path.join(root, fname)
        base_name, _ = os.path.splitext(fname)

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"❌ Error opening {pdf_path}: {e}")
            continue

        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            img = page_pixmap_to_image(page)
            page_no = page_index + 1
            out_fname = f"{base_name}_{page_no}.png"

            rel_path = os.path.relpath(root, PDF_DIR)
            out_dir = os.path.join(OUTPUT_IMAGE_DIR, rel_path)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, out_fname)

            img.save(out_path, format="PNG")
            print(f"✓ Saved: {out_path}")
