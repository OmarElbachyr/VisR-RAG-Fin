from __future__ import annotations

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
NOISE_DIR    = Path("data/noise_pages")
OUTPUT_CSV   = Path("src/dataset/chunks/chunked_noise_pages.csv")
STRATEGY = "hi_res"

CHUNK_IMG_DIR = Path("chunks_images/noise")
CHUNK_IMG_DIR.mkdir(parents=True, exist_ok=True)
PADDING = 0.05


COLUMNS = [
    "query",
    "original_query",
    "answer",
    "original_answer",
    "image",
    "image_filename",
    "page_number",
    "chunk_type",
    "text_description",
    "chunk_id",
    "hashed_filename",
    "relevancy",
    "correctness",
    "evidence",
    "question_type",
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
    processed_filenames = set()  # or load from existing CSV if desired
    all_rows = []

    count_images = len(list(NOISE_DIR.glob("*.png")))
    for noise_img_path in tqdm(NOISE_DIR.glob("*.png"), desc="Processing noise images", total=count_images):
        if noise_img_path.name in processed_filenames:
            continue
        splitted = noise_img_path.stem.rsplit("_", 1)
        page_number = splitted[-1] if splitted and splitted[-1].isdigit() else ""
        chunks = _extract_page_chunks(noise_img_path)
        for text, ctype, cid in chunks:
            all_rows.append({
                "query":            "",
                "original_query":   "",
                "answer":           "",
                "original_answer":  "",
                "image":            str(noise_img_path),
                "image_filename":   noise_img_path.name,
                "page_number":      page_number,
                "chunk_type":       ctype,
                "text_description": text,
                "chunk_id":         cid,
                "hashed_filename":  "",
                "relevancy":        "",
                "correctness":      "",
                "evidence":         "",
                "question_type":    "",
                "company":          "",
                "language":         "",
            })

    df = pd.DataFrame(all_rows, columns=COLUMNS)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Wrote {len(df)} rows → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()