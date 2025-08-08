from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple
import shutil

from dataset.utils import filter_qa_pairs
import pandas as pd
from PIL import Image
from tqdm import tqdm
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.image import partition_image  # OCR / layout extraction

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
JSON_PATH = Path("data/annotations/label-studio-data-min.json")
IMG_DIR = Path("data/pages")
NOISE_DIR = Path("data/noise_pages")  # path for noise images
CSV_PATH = Path("src/dataset/chunks/chunked_pages_all.csv")
ALL_PAGES_DIR = Path("data/all_pages")  # folder to collect all pages
STRATEGY = "hi_res"
LIMIT = -1

CHUNK_IMG_DIR = Path("chunks_images")
CHUNK_IMG_DIR.mkdir(parents=True, exist_ok=True)
PADDING = 0.05

# ensure output folders exist
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
ALL_PAGES_DIR.mkdir(parents=True, exist_ok=True)

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
    "is_noise",
]

def _extract_page_chunks(img_path: Path) -> List[Tuple[str, str, str]]:
    """Partition a page into (text, type, id) tuples."""
    try:
        elements = partition_image(filename=str(img_path), strategy=STRATEGY)
    except Exception as exc:
        tqdm.write(f"OCR failed for {img_path.name}: {exc}")
        return []

    text_elements = [e for e in elements if getattr(e, "category", "") not in {"Image", "Table"}]
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

        left = max(int(x1 - pad_x), 0)
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
    processed_images: set[str] = set()
    all_pages_images: set[Path] = set()

    # Process JSON records
    for rec in tqdm(records, desc="Processing records"):
        png_name = Path(str(rec.get("image_filename", "")).split("?d=")[-1]).name
        processed_images.add(png_name)
        img_path = IMG_DIR / png_name
        all_pages_images.add(img_path)

        chunks = _extract_page_chunks(img_path)
        if not chunks:
            continue

        # scan q1, q2, …
        q_keys = sorted(k for k in rec if k.startswith("q") and k[1:].isdigit())
        for q_key in q_keys:
            idx = int(q_key[1:])

            raw_q = rec.get(f"q{idx}", "")
            if isinstance(raw_q, dict):
                q_text = raw_q.get("text", "")
                query = str(q_text[0]).strip() if isinstance(q_text, list) and q_text else str(q_text).strip()
            else:
                query = str(raw_q).strip()

            answer = str(rec.get(f"a{idx}", "")).strip()

            original_query = ""
            original_answer = ""
            qa_list = rec.get("qa_pairs", [])
            if isinstance(qa_list, list) and 1 <= idx <= len(qa_list):
                original_query = str(qa_list[idx-1].get("question", "")).strip()
                oa = qa_list[idx-1].get("original_answer", "")
                original_answer = ", ".join(str(x) for x in oa).strip() if isinstance(oa, list) else str(oa).strip()

            relevancy = str(rec.get(f"relevancy{idx}", "")).strip()
            correctness = str(rec.get(f"correct{idx}", "")).strip()
            evidence = str(rec.get(f"evidence{idx}", "")).strip()
            qtype = str(rec.get(f"type{idx}", "")).strip()

            for text, ctype, cid in chunks:
                all_rows.append({
                    "query": query,
                    "original_query": original_query,
                    "answer": answer,
                    "original_answer": original_answer,
                    "image": str(img_path),
                    "image_filename": png_name,
                    "page_number": rec.get("page_number"),
                    "chunk_type": ctype,
                    "text_description": text,
                    "chunk_id": cid,
                    "hashed_filename": rec.get("hashed_filename"),
                    "relevancy": relevancy,
                    "correctness": correctness,
                    "evidence": evidence,
                    "question_type": qtype,
                    "company": rec.get("company"),
                    "language": rec.get("language"),
                    "is_noise": False,
                })

    # Process noise images
    if NOISE_DIR.exists():
        count_images = len(list(NOISE_DIR.glob("*.png")))
        for img_path in tqdm(NOISE_DIR.glob("*.png"), desc="Processing noise images", total=count_images):
            if not img_path.is_file() or img_path.name in processed_images:
                continue
            if img_path.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
                continue

            chunks = _extract_page_chunks(img_path)
            if not chunks:
                continue

            # extract page number from filename suffix (e.g., 'file_6.png')
            stem = img_path.stem
            try:
                page_number = int(stem.rsplit('_', 1)[-1])
            except ValueError:
                page_number = None

            all_pages_images.add(img_path)

            for text, ctype, cid in chunks:
                all_rows.append({
                    "query": "",
                    "original_query": "",
                    "answer": "",
                    "original_answer": "",
                    "image": str(img_path),
                    "image_filename": img_path.name,
                    "page_number": page_number,
                    "chunk_type": ctype,
                    "text_description": text,
                    "chunk_id": cid,
                    "hashed_filename": "",
                    "relevancy": "",
                    "correctness": "",
                    "evidence": "",
                    "question_type": "",
                    "company": "",
                    "language": "",
                    "is_noise": True,
                })

    # Copy unique pages to all_pages
    for src_path in all_pages_images:
        dst_path = ALL_PAGES_DIR / src_path.name
        if not dst_path.exists():
            shutil.copy2(src_path, dst_path)

    # Write out a fresh CSV with only the desired columns
    df = pd.DataFrame(all_rows, columns=COLUMNS)
    df.to_csv(CSV_PATH, index=False)
    print(f"✅ Wrote {len(df)} rows → {CSV_PATH}")


if __name__ == "__main__":
    main()
