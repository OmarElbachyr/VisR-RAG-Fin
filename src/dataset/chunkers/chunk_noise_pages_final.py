from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple
import random

import pandas as pd
from PIL import Image
from tqdm import tqdm
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.image import partition_image

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
NOISE_IMG_DIR = Path("data/hard_negative_pages_text")
QA_CHUNKS_CSV = Path("src/dataset/chunks/second_pass/chunked_pages_second_pass.csv")
CSV_PATH = Path("src/dataset/chunks/noise_pages_chunks/hard_negative_chunks.csv")
STRATEGY = "hi_res"
NUM_NOISE_PAGES_TO_SAMPLE = -1  # -1 means process all pages, positive number samples that many
random.seed(32)

CHUNK_IMG_DIR = Path("chunks_images")
CHUNK_IMG_DIR.mkdir(parents=True, exist_ok=True)
PADDING = 0.05

# ensure output folder exists
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

COLUMNS = [
    "query",
    "answer",
    "hint",
    "question_id",
    "question_category",
    "image",
    "image_filename",
    "page_number",
    "chunk_type",
    "text_description",
    "chunk_id",
    "hashed_filename",
    "evidence",
    "question_type",
    "company",
    "language",
]

# ---------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------------------

def get_qa_pages() -> set[str]:
    """Get set of unique image filenames from QA chunks CSV file."""
    if not QA_CHUNKS_CSV.exists():
        raise ValueError(f"QA chunks CSV not found: {QA_CHUNKS_CSV}")
    
    df = pd.read_csv(QA_CHUNKS_CSV)
    qa_pages = set(df["image_filename"].unique())
    tqdm.write(f"✅ Loaded {len(qa_pages)} unique QA page filenames from {QA_CHUNKS_CSV.name}")
    return qa_pages

def get_noise_pages_to_process(num_samples: int) -> List[Path]:
    """
    Get noise pages from data/noise_pages, excluding those already in QA chunks CSV.
    If num_samples is -1, process all pages. Otherwise, sample num_samples pages randomly.
    """
    if not NOISE_IMG_DIR.exists():
        raise ValueError(f"Noise pages directory not found: {NOISE_IMG_DIR}")
    
    qa_pages = get_qa_pages()
    all_noise_pages = [f for f in NOISE_IMG_DIR.glob("*") if f.is_file() and f.name not in qa_pages]
    
    # Process all pages if num_samples is -1
    if num_samples == -1:
        tqdm.write(f"✅ Processing all {len(all_noise_pages)} noise pages "
                   f"(excluding {len(qa_pages)} QA pages)")
        return all_noise_pages
    
    # Sample num_samples pages
    if len(all_noise_pages) < num_samples:
        tqdm.write(f"⚠️  Warning: Only {len(all_noise_pages)} noise pages available (excluding QA pages), "
                   f"but {num_samples} requested. Using all available.")
        return all_noise_pages
    
    sampled = random.sample(all_noise_pages, num_samples)
    tqdm.write(f"✅ Sampled {len(sampled)} noise pages from {len(all_noise_pages)} available "
               f"(excluding {len(qa_pages)} QA pages)")
    return sampled

def _extract_page_chunks(img_path: Path) -> List[Tuple[str, str, str]]:
    """Partition a page into (text, type, id) tuples."""
    try:
        elements = partition_image(filename=str(img_path), strategy=STRATEGY)
    except Exception as exc:
        tqdm.write(f"OCR failed for {img_path.name}: {exc}")
        return []

    text_elements   = [e for e in elements if getattr(e, "category", "") not in {"Image", "Table"}]
    figure_elements = [e for e in elements if getattr(e, "category", "") in {"Image", "Table"}]

    chunks: List[Tuple[str, str, str]] = []
    idx = 0
    page_img = Image.open(img_path)

    # text chunks
    for chunk in chunk_by_title(text_elements):
        txt = getattr(chunk, "text", "").strip()
        if not txt:
            continue
        chunks.append((txt, "text", f"{img_path.stem}_{idx}"))
        idx += 1

    # figure chunks
    for el in figure_elements:
        text_attr = getattr(el, "text", "")
        desc = text_attr.strip() if isinstance(text_attr, str) else str(text_attr).strip() if text_attr else ""
        coords = getattr(el.metadata if hasattr(el, "metadata") else {}, "coordinates", None)
        if not coords:
            continue
        points = coords.get("points") if isinstance(coords, dict) else getattr(coords, "points", None)
        if not points:
            continue

        xs = [float(x) for x, _ in points]
        ys = [float(y) for _, y in points]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        w, h = x2 - x1, y2 - y1
        pad_x, pad_y = PADDING * w, PADDING * h

        left  = max(int(x1 - pad_x), 0)
        upper = max(int(y1 - pad_y), 0)
        right = min(int(x2 + pad_x), page_img.width)
        lower = min(int(y2 + pad_y), page_img.height)

        chunk_id = f"{img_path.stem}_{idx}"
        crop_path = CHUNK_IMG_DIR / f"{chunk_id}.png"
        page_img.crop((left, upper, right, lower)).save(crop_path)

        chunks.append((desc, "figure", chunk_id))
        idx += 1

    return chunks

def extract_pdf_hash_from_filename(filename: str) -> str:
    """
    Extract PDF hash from filename.
    Assumes format: <hash>_<page_number>.png
    Returns: <hash>
    """
    stem = Path(filename).stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0]
    return stem

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    # Get noise pages to process (excluding QA pages)
    noise_pages = get_noise_pages_to_process(NUM_NOISE_PAGES_TO_SAMPLE)
    
    if not noise_pages:
        tqdm.write("❌ No hard negative pages to process. Exiting.")
        return

    all_rows: list[dict] = []
    
    for img_path in tqdm(noise_pages, desc="Processing hard negative pages"):
        # Extract page chunks
        chunks = _extract_page_chunks(img_path)
        if not chunks:
            continue

        # Extract metadata from filename
        hashed_filename = extract_pdf_hash_from_filename(img_path.name)
        
        # For noise pages, we don't have explicit page numbers from metadata,
        # so we try to extract from filename or use 0 as placeholder
        page_number_str = img_path.stem.rsplit("_", 1)[-1] if "_" in img_path.stem else "0"
        try:
            page_number = int(page_number_str)
        except ValueError:
            page_number = 0

        # Create chunks with empty query/answer (to be filtered out by QueryQrelsBuilder)
        for text, ctype, cid in chunks:
            all_rows.append({
                "query":            "",  # Empty for noise pages
                "answer":           "",  # Empty for noise pages
                "hint":             "",
                "question_id":      "",
                "question_category": "",
                "image":            str(img_path),
                "image_filename":   img_path.name,
                "page_number":      page_number,
                "chunk_type":       ctype,
                "text_description": text,
                "chunk_id":         cid,
                "hashed_filename":  hashed_filename,
                "evidence":         "",
                "question_type":    "",
                "company":          "unknown",  # Placeholder - will be merged with QA data
                "language":         "EN",       # Default to English
            })

    # Write out CSV with noise page chunks
    df = pd.DataFrame(all_rows, columns=COLUMNS)
    df.to_csv(CSV_PATH, index=False)
    tqdm.write(f"✅ Wrote {len(df)} noise page chunks → {CSV_PATH}")

if __name__ == "__main__":
    main()
