from __future__ import annotations

import json
import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from PIL import Image
from tqdm import tqdm
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.image import partition_image  # OCR / layout extraction

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SAMPLED_DIR    = Path("data/pages")
OUTPUT_CSV   = Path("src/dataset/chunks/chunked_sampled_pages.csv")
STRATEGY = "hi_res"
JSON_DATA_PATH = Path("data/label-studio/label-studio-data.json")
LIMIT = -1

CHUNK_IMG_DIR = Path("chunks_images/noise")
CHUNK_IMG_DIR.mkdir(parents=True, exist_ok=True)
PADDING = 0.05


COLUMNS = [
    "query",
    "original_answer",
    "image",
    "image_filename",
    "page_number",
    "chunk_type",
    "text_description",
    "chunk_id",
    "hashed_filename",
    "company",
    "language",
]

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
        desc = getattr(el, "text", "").strip()
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

def main() -> None:
    # Load JSON data
    with open(JSON_DATA_PATH, 'r') as f:
        json_data = json.load(f)
    
    # Apply LIMIT to JSON data (-1 means process all)
    if LIMIT != -1:
        json_data = json_data[:LIMIT]
    
    # Create a lookup dictionary for faster access
    json_lookup = {}
    for record in json_data:
        hashed_filename_base = record['hashed_filename'].split('.')[0]
        page_number = record['page_number']
        key = f"{hashed_filename_base}_{page_number}"
        json_lookup[key] = record
    
    processed_filenames = set()  # or load from existing CSV if desired
    all_rows = []

    # Only process images that have corresponding JSON records
    available_images = []
    for img_path in SAMPLED_DIR.glob("*.png"):
        image_filename = img_path.name
        base_name = image_filename.split('.')[0]
        if base_name in json_lookup:
            available_images.append(img_path)
    
    print(f"Found {len(available_images)} images with corresponding JSON records.")
    count_images = len(available_images)
    for noise_img_path in tqdm(available_images, desc="Processing images", total=count_images):
        if noise_img_path.name in processed_filenames:
            continue
        
        # Extract hashed filename and page number from image filename
        image_filename = noise_img_path.name  # e.g., 1c13544248fda6951a76950a332e334a_10.png
        base_name = image_filename.split('.')[0]  # e.g., 1c13544248fda6951a76950a332e334a_10
        
        # Look up corresponding JSON record (guaranteed to exist)
        json_record = json_lookup[base_name]
        
        # Extract page number and other info from JSON
        page_number = json_record['page_number']
        hashed_filename = json_record['hashed_filename']
        company = json_record['company']
        language = json_record['language']
        
        # Process QA pairs
        qa_pairs = json_record.get('qa_pairs', [])
        
        chunks = _extract_page_chunks(noise_img_path)
        
        if qa_pairs:
            # Create rows for each QA pair with all chunks
            for qa_pair in qa_pairs:
                query = qa_pair.get('question', '')
                original_answer = ', '.join(qa_pair.get('original_answer', []))
                
                for text, ctype, cid in chunks:
                    all_rows.append({
                        "query":            query,
                        "original_answer":  original_answer,
                        "image":            str(noise_img_path),
                        "image_filename":   image_filename,
                        "page_number":      page_number,
                        "chunk_type":       ctype,
                        "text_description": text,
                        "chunk_id":         cid,
                        "hashed_filename":  hashed_filename,
                        "company":          company,
                        "language":         language,
                    })
        else:
            # No QA pairs, create rows with empty query/answer
            for text, ctype, cid in chunks:
                all_rows.append({
                    "query":            "",
                    "original_answer":  "",
                    "image":            str(noise_img_path),
                    "image_filename":   image_filename,
                    "page_number":      page_number,
                    "chunk_type":       ctype,
                    "text_description": text,
                    "chunk_id":         cid,
                    "hashed_filename":  hashed_filename,
                    "company":          company,
                    "language":         language,
                })

    df = pd.DataFrame(all_rows, columns=COLUMNS)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Wrote {len(df)} rows → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()