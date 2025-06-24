from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple
import shutil
import pandas as pd
from PIL import Image
from tqdm import tqdm
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.image import partition_image  # OCR / layout extraction

# ---------------------------------------------------------------------------
# CONFIG (kept exactly as in the original script, plus chunk‑image settings)
# ---------------------------------------------------------------------------
JSON_PATH = Path("/home/laura/vqa-ir-qa/data/label-studio-data.json")  # source annotations
NOISE_IMG_DIR = Path("/home/laura/vqa-ir-qa/data/indexed_pages")
IMG_DIR = Path("/home/laura/vqa-ir-qa/data/pages")                     # folder with PNG pages
CSV_PATH = Path("/home/laura/vqa-ir-qa/src/dataset/chunked_pages_with_noise.csv")     # incremental CSV store
STRATEGY = "hi_res"                             # partition_image strategy
LIMIT = -1                                        # set e.g. 3 for a quick test

# Where to store the cropped figure chunks
CHUNK_IMG_DIR = Path("/home/laura/vqa-ir-qa/data/chunks_images")
CHUNK_IMG_DIR.mkdir(parents=True, exist_ok=True)

# Fractional padding to apply around figure bounding boxes (e.g. 0.05 → 5 %)
PADDING = 0.05
# ---------------------------------------------------------------------------

COLUMNS = [
    "query",
    "image",
    "image_filename",
    "answer",
    "page_number",
    "chunk_type",       # "text" | "figure"
    "text_description", # chunk content
    "chunk_id",         # image_filename_{chunk_index}
    "is_noise",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _already_done(df: pd.DataFrame) -> set[Tuple[str, str, str]]:
    """Return the set of (image_filename, query, chunk_id) already present."""
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
    """Partition a page into `(chunk_text, chunk_type, chunk_id)` tuples.

    * `chunk_type` = "text" for textual content, "figure" for Image/Table elements.
    * Figures are cropped (with `PADDING` %) and saved to `CHUNK_IMG_DIR` using the
      naming scheme `image_filename_{index}.png`.
    * For figures, `chunk_text` is `element.text` (may be empty).
    """
    try:
        elements = partition_image(filename=str(img_path), strategy=STRATEGY)
    except Exception as exc:  # noqa: BLE001
        tqdm.write(f"OCR failed for {img_path.name}: {exc}")
        return []

    text_elements = [e for e in elements if getattr(e, "category", "") not in {"Image", "Table"}]
    figure_elements = [e for e in elements if getattr(e, "category", "") in {"Image", "Table"}]

    chunks: List[Tuple[str, str, str]] = []
    idx = 0

    # Open the page image once for all crops
    page_img = Image.open(img_path)

    # --- textual chunks ----------------------------------------------------
    for chunk in chunk_by_title(text_elements):
        txt = getattr(chunk, "text", "").strip()
        if not txt:
            continue
        chunk_id = f"{img_path.stem}_{idx}"
        chunks.append((txt, "text", chunk_id))
        idx += 1

    # --- figure chunks -----------------------------------------------------
    for el in figure_elements:
        desc = getattr(el, "text", "").strip()

        # ElementMetadata object → coordinates dict/object → points
        meta = getattr(el, "metadata", None)
        coords_attr = getattr(meta, "coordinates", None) if meta else None
        if coords_attr is None:
            continue

        if isinstance(coords_attr, dict):
            points = coords_attr.get("points")
        else:
            points = getattr(coords_attr, "points", None)
        if not points:
            continue  # skip figures without geometry

        # Expect the four corners, take min/max to be safe
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        # Apply padding (percentage of box dimensions)
        w, h = x2 - x1, y2 - y1
        pad_x, pad_y = PADDING * w, PADDING * h
        left = max(int(x1 - pad_x), 0)
        upper = max(int(y1 - pad_y), 0)
        right = min(int(x2 + pad_x), page_img.width)
        lower = min(int(y2 + pad_y), page_img.height)

        # Crop and save
        chunk_id = f"{img_path.stem}_{idx}"
        crop_path = CHUNK_IMG_DIR / f"{chunk_id}.png"
        page_img.crop((left, upper, right, lower)).save(crop_path)

        chunks.append((desc, "figure", chunk_id))
        idx += 1

    return chunks


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

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

    new_rows: List[dict] = []

    # ----------------------------
    # 1. Process QA (annotated) data
    # ----------------------------
    with open(JSON_PATH, encoding="utf-8") as fh:
        records = json.load(fh)

    if LIMIT > 0:
        records = records[:LIMIT]

    for rec in tqdm(records, desc="Processing QA records"):
        data = rec["data"]
        png_name = os.path.basename(data["image_filename"].split("?d=")[-1])
        img_path = IMG_DIR / png_name

        page_chunks = _extract_page_chunks(img_path)
        if not page_chunks:
            continue

        for qa in data["qa_pairs"]:
            for chunk_text, chunk_type, chunk_id in page_chunks:
                key = (png_name, qa["question"], chunk_id)
                if key in done:
                    continue
                new_rows.append({
                    "query": qa["question"],
                    "image": str(img_path),
                    "image_filename": png_name,
                    "answer": json.dumps([a.strip() for a in qa["original_answer"]]),
                    "page_number": data["page_number"],
                    "chunk_type": chunk_type,
                    "text_description": chunk_text,
                    "chunk_id": chunk_id,
                    "is_noise": 0,
                })

    # ----------------------------
    # 2. Process NOISE (unannotated) images
    noise_images = sorted(NOISE_IMG_DIR.glob("*.png"))
    if LIMIT > 0:
        noise_images = noise_images[:LIMIT]

    for img_path in tqdm(noise_images, desc="Processing noise images"):
        png_name = img_path.name
        page_chunks = _extract_page_chunks(img_path)
        if not page_chunks:
            continue

        # Extract page number from filename (e.g., "file_045.png" → "045")
        try:
            page_number = png_name.split("_")[1].split(".")[0]
        except IndexError:
            page_number = ""  # fallback if filename format is unexpected

        for chunk_text, chunk_type, chunk_id in page_chunks:
            key = (png_name, "", chunk_id)
            if key in done:
                continue
            new_rows.append({
                "query": "",
                "image": str(img_path),
                "image_filename": png_name,
                "answer": "",
                "page_number": page_number,
                "chunk_type": chunk_type,
                "text_description": chunk_text,
                "chunk_id": chunk_id,
                "is_noise": 1,
            })

    # ----------------------------
    # Final CSV write
    # ----------------------------
    if not new_rows:
        print("✅ No new rows; CSV is current.")
        return

    df_new = pd.DataFrame(new_rows, columns=COLUMNS)
    df_all = (
        pd.concat([df_existing, df_new], ignore_index=True)
        .drop_duplicates(subset=["image_filename", "query", "chunk_id"])
    )
    df_all.to_csv(CSV_PATH, index=False)

    def copy_all_images_to_all_pages():
        all_pages_dir = Path("/home/laura/vqa-ir-qa/data/all_pages")
        all_pages_dir.mkdir(parents=True, exist_ok=True)

        src_dirs = [NOISE_IMG_DIR, IMG_DIR]

        for src_dir in src_dirs:
            for img_path in src_dir.glob("*.png"):
                dst_path = all_pages_dir / img_path.name
                try:
                    shutil.copy2(img_path, dst_path)
                except Exception as e:
                    tqdm.write(f"⚠️ Failed to copy {img_path} to {dst_path}: {e}")

    # Call the function
    copy_all_images_to_all_pages()
    print("📁 All images copied to /data/all_pages")

    print(f"✅ Added {len(df_new)} rows (total: {len(df_all)}) → {CSV_PATH}")


if __name__ == "__main__":
    main()
