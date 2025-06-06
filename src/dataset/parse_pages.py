from __future__ import annotations

"""
Incrementally build **qa_pairs.csv** from a Label‑Studio‑style JSON export that
contains QA pairs and page‑level PNGs.

Adds **text_description** — the plain‑text OCR obtained with Unstructured’s
`partition_image` — so each row now includes the raw page text, **excluding**
text extracted from Image or Table elements.

*   Output: `src/dataset/qa_pairs.csv` (appended or created)
*   Columns:
    - **query**           – question text
    - **image**           – relative PNG path (for later casting to `Image()`)
    - **image_filename**  – file name only
    - **answer**          – JSON‑encoded list of answers
    - **page_number**     – page number (string)
    - **text_description** – concatenated OCR text of the page (no Image/Table)

Running the script again adds only unseen `(image_filename, query)` rows.
"""

import json
import os
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm
from unstructured.partition.image import partition_image
# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
JSON_PATH = Path("data/label-studio-data.json")  # source annotations
IMG_DIR = Path("data/pages")                     # folder with PNG pages
CSV_PATH = Path("src/dataset/parsed_pages.csv")     # incremental CSV store
STRATEGY = "hi_res"                            # partition_image strategy
LIMIT = 3                                      # set e.g. 3 for a quick test
# ---------------------------------------------------------------------------

COLUMNS = [
    "query",
    "image",
    "image_filename",
    "answer",
    "page_number",
    "text_description",
]


def _already_done(df: pd.DataFrame) -> set[tuple[str, str]]:
    if df.empty:
        return set()
    return set(zip(df["image_filename"], df["query"]))


def _ocr_text(img_path: Path) -> str:
    """Return concatenated OCR text for a page, skipping Image and Table elements."""
    try:
        elements = partition_image(filename=str(img_path), strategy=STRATEGY)
        return " ".join(
            e.text.strip()
            for e in elements
            if getattr(e, "text", None) and getattr(e, "category", "") not in {"Image", "Table"}
        )
    except Exception as exc:  # noqa: BLE001
        tqdm.write(f"OCR failed for {img_path.name}: {exc}")
        return ""


def main() -> None:
    # Load existing CSV (if any) and guarantee all columns exist
    if CSV_PATH.exists():
        df_existing = pd.read_csv(CSV_PATH)
        for col in COLUMNS:
            if col not in df_existing.columns:
                df_existing[col] = ""
    else:
        df_existing = pd.DataFrame(columns=COLUMNS)

    done = _already_done(df_existing)

    # Read JSON annotations
    with open(JSON_PATH, encoding="utf-8") as fh:
        records = json.load(fh)

    if LIMIT:
        records = records[:LIMIT]

    new_rows: List[dict] = []

    for rec in tqdm(records, desc="Processing records"):
        data = rec["data"]
        png_name = os.path.basename(data["image_filename"].split("?d=")[-1])
        img_path = IMG_DIR / png_name

        page_text = _ocr_text(img_path)

        # Expand QA pairs → rows
        for qa in data["qa_pairs"]:
            key = (png_name, qa["question"])
            if key in done:
                continue
            new_rows.append(
                {
                    "query": qa["question"],
                    "image": str(img_path),
                    "image_filename": png_name,
                    "answer": json.dumps([a.strip() for a in qa["original_answer"]]),
                    "page_number": data["page_number"],
                    "text_description": page_text,
                }
            )

    if not new_rows:
        print("✅ No new rows; CSV is current.")
        return

    # Append and save
    df_new = pd.DataFrame(new_rows, columns=COLUMNS)
    df_all = (
        pd.concat([df_existing, df_new], ignore_index=True)
        .drop_duplicates(subset=["image_filename", "query"])
    )
    df_all.to_csv(CSV_PATH, index=False)
    print(f"✅ Added {len(df_new)} rows (total: {len(df_all)}) → {CSV_PATH}")


if __name__ == "__main__":
    main()
