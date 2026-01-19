#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Set, Tuple

from tqdm import tqdm
from unstructured.partition.image import partition_image


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

K = 10  # Top-K pages per query

ANNOTATIONS_FILE = Path(
    "data/annotations/final_annotations/filtered_annotations/by_category/second_pass_classified_qa_category_A.json"
)

RETRIEVAL_RESULTS = Path(
    "src/retrievers/results/chunked_pages_second_pass_rq1/0.5/"
    "retrieved_pages/nomic-ai_colnomic-embed-multimodal-7b_sorted_run.json"
)

PAGE_IMG_DIR = Path("data/noise_pages")
OUT_TXT_DIR = Path(f"src/generators/rq_5/topk_pages_txt/k_{K}")

STRATEGY = "hi_res"

OUT_TXT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# LOAD ANNOTATED QUERY IDS
# ---------------------------------------------------------------------------

def load_annotated_query_ids(annotations_path: Path) -> Set[str]:
    """
    Extract query IDs exactly like in generation scripts.
    """
    with open(annotations_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    qids = set()
    for rec in records:
        for qa in rec.get("qa_pairs", []):
            qid = qa.get("question_id")
            if qid:
                qids.add(qid)

    return qids


# ---------------------------------------------------------------------------
# LOAD TOP-K PAGES (ANNOTATION-SCOPED)
# ---------------------------------------------------------------------------

def load_topk_pages_for_queries(
    results_path: Path,
    annotated_qids: Set[str],
    k: int,
) -> Set[Tuple[str, int]]:
    """
    Only consider queries present in annotations.
    Expected run format:
    {
      "q1": {
        "query": "...",
        "results": {
          "hashed_pageid": score
        }
      }
    }
    """
    with open(results_path, "r", encoding="utf-8") as f:
        runs = json.load(f)

    selected_pages: Set[Tuple[str, int]] = set()

    for qid in annotated_qids:
        if qid not in runs:
            continue

        results = runs[qid]["results"]

        for page_id in list(results.keys())[:k]:
            hashed, page = page_id.rsplit("_", 1)
            selected_pages.add((hashed, int(page)))

    return selected_pages


# ---------------------------------------------------------------------------
# PARSE PAGE IMAGE → TEXT
# ---------------------------------------------------------------------------

def parse_page_to_text(img_path: Path) -> str:
    elements = partition_image(
        filename=str(img_path),
        strategy=STRATEGY,
    )

    texts = []
    for el in elements:
        if getattr(el, "category", "") in {"Image", "Table"}:
            continue
        txt = getattr(el, "text", "")
        if txt:
            texts.append(txt.strip())

    return "\n\n".join(texts)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    annotated_qids = load_annotated_query_ids(ANNOTATIONS_FILE)
    print(f"📌 Loaded {len(annotated_qids)} annotated queries")

    pages = load_topk_pages_for_queries(
        RETRIEVAL_RESULTS,
        annotated_qids,
        K,
    )
    print(f"🔎 Selected {len(pages)} unique pages (K={K})")

    for hashed_filename, page_number in tqdm(pages, desc="Parsing pages"):
        img_name = f"{hashed_filename}_{page_number}.png"
        img_path = PAGE_IMG_DIR / img_name

        if not img_path.exists():
            tqdm.write(f"⚠️ Missing image: {img_name}")
            continue

        try:
            text = parse_page_to_text(img_path)
        except Exception as e:
            tqdm.write(f"❌ Failed parsing {img_name}: {e}")
            continue

        if not text.strip():
            continue

        out_path = OUT_TXT_DIR / f"{hashed_filename}_{page_number}.txt"
        out_path.write_text(text, encoding="utf-8")

    print(f"✅ Done. Files written to: {OUT_TXT_DIR}")


if __name__ == "__main__":
    main()
