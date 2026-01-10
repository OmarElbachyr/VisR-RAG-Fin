import os
from pathlib import Path
import fitz
from PIL import Image

PDF_DIR = "data/indexed_pdfs"
OUTPUT_IMAGE_DIR = Path("data/noise_pages")

# Configurable settings
DPI = 100
IMAGE_FORMAT = "jpg"  # "jpg" or "png"
JPEG_QUALITY = 90

OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

def page_pixmap_to_image(page, dpi=150):
    """Render page to image at specified DPI"""
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    
    # Always convert to RGB
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
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
            img = page_pixmap_to_image(page, dpi=DPI)
            page_no = page_index + 1
            out_fname = f"{base_name}_{page_no}.{IMAGE_FORMAT}"

            rel_path = os.path.relpath(root, PDF_DIR)
            out_dir = os.path.join(OUTPUT_IMAGE_DIR, rel_path)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, out_fname)

            # Save with format
            if IMAGE_FORMAT.lower() == "jpg":
                img.save(out_path, "JPEG", quality=JPEG_QUALITY)
            else:
                img.save(out_path, format=IMAGE_FORMAT.upper())
            print(f"✓ Saved: {out_path}")
