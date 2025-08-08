from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from PIL import Image
from tqdm import tqdm
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.image import partition_image  # OCR / layout extraction
from dataset.utils import filter_qa_pairs

def extract_value(field):
    """Extract value from either string or nested dict structure"""
    if isinstance(field, str):
        return field
    elif isinstance(field, dict) and "text" in field:
        text_list = field["text"]
        if isinstance(text_list, list) and len(text_list) > 0:
            return text_list[-1]  # Return the last element
        elif isinstance(text_list, str):
            return text_list
    return None

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
JSON_PATH = Path("data/annotations/label-studio-data-min.json")
IMG_DIR = Path("data/pages")
CSV_PATH = Path("src/dataset/chunks/chunked_pages.csv")
STRATEGY = "hi_res"
LIMIT = -1

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
    # Read and filter JSON annotations
    records = filter_qa_pairs(JSON_PATH)
    

    if LIMIT > 0:
        records = records[:LIMIT]

    all_rows: list[dict] = []
    global_question_counter = 0
    
    for rec in tqdm(records, desc="Processing records"):
        # build page chunks once per record
        png_name = Path(str(rec.get("image_filename", "")).split("?d=")[-1]).name
        img_path = IMG_DIR / png_name
        chunks   = _extract_page_chunks(img_path)
        if not chunks:
            continue

        # scan q1, q2, …
        q_keys = sorted(k for k in rec if k.startswith("q") and k[1:].isdigit())
        for q_key in q_keys:
            idx = int(q_key[1:])

            # Extract question and answer using same logic as utils.py
            raw_q = rec.get(f"q{idx}", "")
            raw_a = rec.get(f"a{idx}", "")
            
            query = extract_value(raw_q)
            answer = extract_value(raw_a)
            
            # Skip if we can't extract valid question or answer
            if not query or not answer:
                continue

            # Increment global question counter for each valid QA pair
            global_question_counter += 1
            question_id = f"q{global_question_counter}"

            # Extract hint from qa_pairs if available
            hint = ""
            qa_list = rec.get("qa_pairs", [])
            for qa in qa_list:
                qa_question = extract_value(qa.get("question"))
                if qa_question == query:
                    hint = qa.get("hint", "")
                    break

            relevancy   = str(rec.get(f"relevancy{idx}", "")).strip()
            correctness = str(rec.get(f"correct{idx}", "")).strip()
            evidence    = str(rec.get(f"evidence{idx}", "")).strip()
            qtype       = str(rec.get(f"type{idx}", "")).strip()

            for text, ctype, cid in chunks:
                all_rows.append({
                    "query":            query,
                    "answer":           answer,
                    "hint":             hint,
                    "question_id":      question_id,
                    "image":            str(img_path),
                    "image_filename":   png_name,
                    "page_number":      rec.get("page_number"),
                    "chunk_type":       ctype,
                    "text_description": text,
                    "chunk_id":         cid,
                    "hashed_filename":  rec.get("hashed_filename"),
                    "relevancy":        relevancy,
                    "correctness":      correctness,
                    "evidence":         evidence,
                    "question_type":    qtype,
                    "company":          rec.get("company"),
                    "language":         rec.get("language"),
                })

    # Write out a fresh CSV with only the desired columns
    df = pd.DataFrame(all_rows, columns=COLUMNS)
    df.to_csv(CSV_PATH, index=False)
    print(f"✅ Wrote {len(df)} rows → {CSV_PATH}")

if __name__ == "__main__":
    main()