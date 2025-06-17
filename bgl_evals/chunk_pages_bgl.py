from __future__ import annotations

import json
import os
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
JSON_PATH = Path("bgl/document_qa_pairs.json")
IMG_DIR = Path("bgl/pages")
CSV_PATH = Path("bgl/chunked_pages.csv")
STRATEGY = "hi_res"
LIMIT = -1

CHUNK_IMG_DIR = Path("chunks_images")
CHUNK_IMG_DIR.mkdir(parents=True, exist_ok=True)
PADDING = 0.05

COLUMNS = [
    "query",
    "image",
    "image_filename",
    "answer",
    "page_number",
    "chunk_type",
    "text_description",
    "chunk_id",
]


def _already_done(df: pd.DataFrame) -> set[Tuple[str, str, str]]:
    if df.empty:
        return set()
    return set(
        zip(
            df["image_filename"].astype(str),
            df["query"].astype(str),
            df["chunk_id"].astype(str),
        )
    )


def _extract_page_chunks(img_path: Path) -> List[Tuple[str, str, str]]:
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
        chunk_id = f"{img_path.stem}_{idx}"
        chunks.append((txt, "text", chunk_id))
        idx += 1

    # figure chunks
    for el in figure_elements:
        desc = getattr(el, "text", "").strip()
        meta = getattr(el, "metadata", None)
        coords_attr = getattr(meta, "coordinates", None) if meta else None
        if not coords_attr:
            continue
        points = coords_attr.get("points") if isinstance(coords_attr, dict) else getattr(coords_attr, "points", None)
        if not points:
            continue

        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
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
    # always start fresh
    df_existing = pd.DataFrame(columns=COLUMNS)
    done = set()

    # load new-format JSON
    with open(JSON_PATH, encoding="utf-8") as fh:
        records = json.load(fh)
    if LIMIT > 0:
        records = records[:LIMIT]

    new_rows: List[dict] = []

    for rec in tqdm(records, desc="Processing records"):
        doc_name = rec["doc_name"]

        for qa in rec["qa_pairs"]:
            question = qa["question"]
            answer = qa["answer"]
            for page_number in qa["evidence_pages"]:
                png_name = f"{doc_name}_{page_number}.png"
                img_path = IMG_DIR / png_name

                page_chunks = _extract_page_chunks(img_path)
                if not page_chunks:
                    continue

                for chunk_text, chunk_type, chunk_id in page_chunks:
                    key = (png_name, question, chunk_id)
                    if key in done:
                        continue
                    new_rows.append({
                        "query":            question,
                        "image":            str(img_path),
                        "image_filename":   png_name,
                        "answer":           answer,
                        "page_number":      page_number,
                        "chunk_type":       chunk_type,
                        "text_description": chunk_text,
                        "chunk_id":         chunk_id,
                    })
                    done.add(key)

    if not new_rows:
        print("✅ No new rows; nothing to write.")
        return

    df_new = pd.DataFrame(new_rows, columns=COLUMNS)
    df_new = df_new.drop_duplicates(subset=["image_filename", "query", "chunk_id"])
    df_new.to_csv(CSV_PATH, index=False)
    print(f"✅ Wrote {len(df_new)} rows → {CSV_PATH}")


if __name__ == "__main__":
    main()
