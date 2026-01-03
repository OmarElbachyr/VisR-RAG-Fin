"""
Script to prepare and save dataset locally from visual_pages_responses.json.
Expands each page record into 3 rows (one per query type: broad_topical, specific_detail, visual_element).
Uses PyMuPDF for fast rendering, and stores images on disk instead of embedding PIL images.
"""

import json
import random
import shutil
import hashlib
from pathlib import Path
from typing import Optional

from PIL import Image
import fitz  # PyMuPDF


def generate_doc_id(filename: str) -> str:
    """Generate a consistent hash-based doc_id from filename."""
    return hashlib.md5(filename.encode()).hexdigest()


def load_dataset_from_json(
    json_path: str = "finetune/scripts/visual_pages_responses.json",
    pdf_base_path: str = "finetune/training_documents",
    train_test_split: float = 0.98,
    output_dir: Optional[str] = None,
    dpi: int = 150,
    image_format: str = "jpg",   # "png" also supported
    jpeg_quality: int = 90,       # recommended
) -> None:

    json_path = Path(json_path)
    pdf_base_path = Path(pdf_base_path)

    if output_dir:
        output_dir = Path(output_dir)
        
        # Clear existing output directory for fresh start
        if output_dir.exists():
            shutil.rmtree(output_dir)
            print(f"Cleared existing output directory: {output_dir}")
        
        # Create train and test directories for ImageFolder format
        train_dir = output_dir / "train"
        test_dir = output_dir / "test"
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
    else:
        train_dir = None
        test_dir = None

    print(f"Starting dataset loading from {json_path}...")

    # -----------------
    # Load JSON file
    # -----------------
    with open(json_path, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} entries from JSON file")

    # -----------------------------------------
    # Process records and expand to 3 rows per page
    # -----------------------------------------
    samples = []
    skipped_count = 0
    processed_records = 0
    cached_pdfs = {}  # Cache opened PDFs to avoid reopening

    # Map filename to set of pages being processed
    filename_pages = {}

    print(f"Expanding records: 1 page → 3 query rows...")

    for record_idx, record in enumerate(data, 1):
        filename = record.get("filename")
        page_number = record.get("page_number")
        response = record.get("response", {})
        total_pages = record.get("total_pages")
        visual_pages = record.get("visual_pages")

        if not all([filename, page_number is not None, response, total_pages is not None]):
            skipped_count += 1
            continue

        # Track which pages we need for each file
        if filename not in filename_pages:
            filename_pages[filename] = set()
        filename_pages[filename].add(page_number)

        doc_id = generate_doc_id(filename)

        # Create 3 samples: one per query type
        query_types = [
            ("broad_topical", response.get("broad_topical_query"), response.get("broad_topical_explanation")),
            ("specific_detail", response.get("specific_detail_query"), response.get("specific_detail_explanation")),
            ("visual_element", response.get("visual_element_query"), response.get("visual_element_explanation")),
        ]

        for query_type, query_text, explanation in query_types:
            if not query_text:
                skipped_count += 1
                continue

            samples.append({
                "filename": filename,
                "page_number": page_number,
                "doc_id": doc_id,
                "query": query_text,
                "query_type": query_type,
                "query_explanation": explanation or "",
                "total_pages": total_pages,
                "visual_pages": visual_pages,
                "image": None,  # Will be filled later
                "image_filename": None,  # Will be filled later
            })

        if record_idx % 100 == 0 or record_idx == 1:
            print(
                f"Processed {record_idx} records → {len(samples)} query samples so far"
            )

    print(f"\n✓ Expanded to {len(samples)} query samples")
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} invalid records/queries")

    # -----------------------------------------
    # Render PDF pages and attach images
    # -----------------------------------------
    print(f"\nRendering PDF pages...")

    total_filenames = len(filename_pages)
    for file_idx, (filename, page_set) in enumerate(filename_pages.items(), 1):
        pdf_path = pdf_base_path / filename

        if not pdf_path.exists():
            print(f"⚠️  PDF not found: {pdf_path}")
            samples = [s for s in samples if s["filename"] != filename]
            continue

        if file_idx % 20 == 0 or file_idx == 1:
            print(f"  Rendering {file_idx}/{total_filenames}: {filename}")

        try:
            doc = fitz.open(str(pdf_path))
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)

            page_images = {}
            for page_num in sorted(page_set):
                try:
                    page = doc.load_page(page_num - 1)
                    pix = page.get_pixmap(matrix=mat)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    page_images[page_num] = img
                except Exception as e:
                    print(f"    ⚠️  Error rendering page {page_num}: {e}")
                    continue

            doc.close()

            doc_id = generate_doc_id(filename)
            for sample in samples:
                if sample["doc_id"] == doc_id:
                    page_num = sample["page_number"]
                    if page_num in page_images:
                        sample["image"] = page_images[page_num]
                        sample["image_filename"] = f"{doc_id}_page_{page_num}.{image_format}"
                    else:
                        sample["image"] = None

        except Exception as e:
            print(f"⚠️  Error processing {pdf_path}: {e}")
            samples = [s for s in samples if s["filename"] != filename]
            continue

    samples_before = len(samples)
    samples = [s for s in samples if s["image"] is not None]
    print(f"\n✓ Rendered images: {len(samples)} samples (removed {samples_before - len(samples)})")

    # -----------------------------------------
    # Shuffle + train/test split
    # -----------------------------------------
    random.seed(42)
    random.shuffle(samples)

    split_idx = int(len(samples) * train_test_split)
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]

    print(f"\n✓ Dataset split:")
    print(f"  Train size: {len(train_samples)}")
    print(f"  Test size: {len(test_samples)}")

    # -----------------------------
    # Save images and metadata in ImageFolder format
    # -----------------------------
    if output_dir:
        print(f"\nSaving dataset in ImageFolder format to {output_dir}...")
        
        def save_split(samples, split_dir, split_name):
            """Save images and metadata.jsonl for a split"""
            metadata_lines = []
            
            for idx, sample in enumerate(samples, 1):
                img = sample["image"]
                img_filename = sample["image_filename"]
                img_path = split_dir / img_filename
                
                # Save image
                if image_format.lower() == "jpg":
                    img.save(img_path, "JPEG", quality=jpeg_quality)
                else:
                    img.save(img_path, format=image_format.upper())

                if idx % 500 == 0:
                    print(f"    Saved {idx} samples for {split_name} split...")
                
                # Prepare metadata entry
                metadata_lines.append(json.dumps({
                    "file_name": img_filename,
                    "query": sample["query"],
                    "query_type": sample["query_type"],
                    "query_explanation": sample["query_explanation"],
                    "filename": sample["filename"],
                    "page_number": sample["page_number"],
                    "doc_id": sample["doc_id"],
                    "total_pages": sample["total_pages"],
                    "visual_pages": sample["visual_pages"],
                }))
            
            # Save metadata.jsonl
            metadata_path = split_dir / "metadata.jsonl"
            with open(metadata_path, "w") as f:
                f.write("\n".join(metadata_lines))
            
            print(f"  ✓ Saved {len(samples)} images and metadata for {split_name} split")
        
        save_split(train_samples, train_dir, "train")
        save_split(test_samples, test_dir, "test")
        
        print(f"\n✓ Dataset saved in ImageFolder format to {output_dir}")
        print(f"  Structure:")
        print(f"    train/")
        print(f"      ├── *.jpg (images)")
        print(f"      └── metadata.jsonl")
        print(f"    test/")
        print(f"      ├── *.jpg (images)")
        print(f"      └── metadata.jsonl")

if __name__ == "__main__":
    load_dataset_from_json(
        json_path="finetune/scripts/visual_pages_responses.json",
        pdf_base_path="finetune/training_documents",
        train_test_split=0.98,
        output_dir="finetune/datasets/visual_queries_dataset",
        dpi=100,
        image_format="jpg",
        jpeg_quality=90,
    )

    print("\n✓ Dataset preparation complete!")
