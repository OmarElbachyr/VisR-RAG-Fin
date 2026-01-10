"""
Hard Negative Mining using Text Retriever (SentenceTransformer + fitz)

For each query, retrieves similar pages from the same PDF based on text
and selects top N pages (excluding the positive) as hard negatives.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import fitz  # PyMuPDF
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer

# =============================================================================
# CONFIGURATION - Edit these variables
# =============================================================================

# Number of hard negatives per query
N_SAVE = 5   # Number of pages to actually save to disk
N_JSON = 5   # Number of pages to record in JSON

# Paths
BENCHMARK_CSV = "src/dataset/chunks/final_chunks/chunked_pages_category_A.csv"
PDF_DIR = "data/indexed_pdfs"
OUTPUT_IMAGES_DIR = "data/hard_negative_pages_text"
OUTPUT_JSON = "data/hard_negatives_mining/query_hard_negatives_text.json"

# Model settings
MODEL_NAME = "BAAI/bge-m3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32

# Minimum text length to consider a page valid
MIN_TEXT_LENGTH = 50

# Limit number of PDFs to process (for testing)
LIMIT = None  # Set to an integer, e.g. 3, to process only first N PDFs

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_benchmark(csv_path: str) -> pd.DataFrame:
    """Load benchmark CSV and get unique queries with their positive pages."""
    df = pd.read_csv(csv_path)
    
    # Get unique queries (one row per question_id)
    unique_queries = df.groupby("question_id").first().reset_index()
    unique_queries = unique_queries[["question_id", "query", "hashed_filename", "image_filename", "page_number"]]
    
    print(f"Loaded {len(unique_queries)} unique queries from {len(df['hashed_filename'].unique())} PDFs")
    return unique_queries


def group_queries_by_pdf(df: pd.DataFrame) -> dict:
    """Group queries by their source PDF."""
    grouped = defaultdict(list)
    for _, row in df.iterrows():
        grouped[row["hashed_filename"]].append({
            "question_id": row["question_id"],
            "query": row["query"],
            "positive_page": row["image_filename"],
            "positive_page_number": int(row["page_number"]),
        })
    return dict(grouped)


def extract_text_from_pdf(pdf_path: str) -> list[str]:
    """Extract text from each page of a PDF using fitz."""
    doc = fitz.open(pdf_path)
    texts = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        texts.append(text.strip())
    
    doc.close()
    return texts


def save_page_image(pdf_path: str, page_idx: int, output_path: str):
    """Save a single page from PDF as an image."""
    doc = fitz.open(pdf_path)
    page = doc[page_idx]
    
    # Render at 150 DPI
    mat = fitz.Matrix(150 / 72, 150 / 72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    img.save(output_path)
    doc.close()


def embed_texts(model, texts: list[str], batch_size: int) -> np.ndarray:
    """Embed texts using SentenceTransformer."""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True
    )
    return embeddings.astype("float32")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Hard Negative Mining with Text Retriever")
    print("=" * 60)
    
    # Create output directories
    os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    
    # Ask user if they want to clear previous results
    response = input("\nClear previous hard_negative_pages and JSON? (y/n): ").strip().lower()
    if response == 'y':
        import shutil
        if os.path.exists(OUTPUT_IMAGES_DIR):
            shutil.rmtree(OUTPUT_IMAGES_DIR)
            os.makedirs(OUTPUT_IMAGES_DIR)
            print(f"  ✓ Cleared {OUTPUT_IMAGES_DIR}/")
        if os.path.exists(OUTPUT_JSON):
            os.remove(OUTPUT_JSON)
            print(f"  ✓ Cleared {OUTPUT_JSON}")
    
    # Load benchmark
    print("\n[1/4] Loading benchmark...")
    benchmark = load_benchmark(BENCHMARK_CSV)
    queries_by_pdf = group_queries_by_pdf(benchmark)
    print(f"  → {len(queries_by_pdf)} PDFs to process")
    
    # Load model
    print(f"\n[2/4] Loading model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    print(f"  → Using device: {DEVICE}")
    
    # Process each PDF
    print(f"\n[3/4] Mining hard negatives (N_SAVE={N_SAVE}, N_JSON={N_JSON})...")
    
    all_hard_negatives = {}  # question_id -> list of hard negative filenames
    saved_pages = set()  # Track already saved pages to avoid duplicates
    skipped_queries = 0
    processed_pdfs = 0
    failed_pdfs = []
    pdf_items = list(queries_by_pdf.items())
    if LIMIT is not None:
        pdf_items = pdf_items[:LIMIT]
    
    try:
        for pdf_name, queries in tqdm(pdf_items, desc="Processing PDFs"):
            pdf_path = os.path.join(PDF_DIR, pdf_name)
            
            if not os.path.exists(pdf_path):
                print(f"  ⚠️  PDF not found: {pdf_path}")
                continue
            
            try:
                pdf_hash = pdf_name.replace(".pdf", "")
                
                # Extract text from all pages
                page_texts = extract_text_from_pdf(pdf_path)
                num_pages = len(page_texts)
                num_queries = len(queries)
                print(f"  📄 PDF: {pdf_name}")
                print(f"  → Pages in PDF: {num_pages} | Queries: {num_queries}")
                
                if num_pages == 0:
                    print(f"  ⚠️  No pages in PDF: {pdf_name}")
                    failed_pdfs.append((pdf_name, "No pages in PDF"))
                    continue
                
                # Find pages with sufficient text
                valid_pages = []  # (page_idx, text)
                for idx, text in enumerate(page_texts):
                    if len(text) >= MIN_TEXT_LENGTH:
                        valid_pages.append((idx, text))
                
                if len(valid_pages) < 2:
                    # Need at least 2 pages (1 positive + 1 negative)
                    print(f"  ⚠️  Insufficient text pages: {len(valid_pages)} (need >= 2)")
                    skipped_queries += len(queries)
                    continue
                
                # Embed valid page texts
                valid_page_indices = [p[0] for p in valid_pages]
                valid_page_texts = [p[1] for p in valid_pages]
                page_embeddings = embed_texts(model, valid_page_texts, BATCH_SIZE)
                
                # Get all query texts for this PDF
                query_texts = [q["query"] for q in queries]
                query_embeddings = embed_texts(model, query_texts, BATCH_SIZE)
                
                processed_pdfs += 1
                
                # For each query, find hard negatives
                for idx, query_info in enumerate(queries):
                    qid = query_info["question_id"]
                    positive_page_num = query_info["positive_page_number"]
                    positive_page_idx = positive_page_num - 1  # 0-indexed
                    
                    # Compute similarity scores (query vs all valid pages)
                    q_emb = query_embeddings[idx]  # [dim]
                    scores = page_embeddings @ q_emb  # [num_valid_pages]
                    
                    # Rank pages by score (descending)
                    ranked_local_indices = scores.argsort()[::-1]
                    
                    # Select top N pages excluding the positive page
                    hard_neg_pages = []  # [(page_idx, score), ...]
                    for local_idx in ranked_local_indices:
                        page_idx = valid_page_indices[local_idx]
                        
                        # Skip the positive page
                        if page_idx == positive_page_idx:
                            continue
                        
                        hard_neg_pages.append((page_idx, float(scores[local_idx])))
                        
                        if len(hard_neg_pages) >= N_JSON:
                            break
                    
                    # Save hard negative page images (only N_SAVE to disk)
                    for page_idx, score in hard_neg_pages[:N_SAVE]:
                        page_num = page_idx + 1  # 1-indexed
                        filename = f"{pdf_hash}_{page_num}.png"
                        output_path = os.path.join(OUTPUT_IMAGES_DIR, filename)
                        
                        # Save if not already saved
                        if filename not in saved_pages:
                            save_page_image(pdf_path, page_idx, output_path)
                            saved_pages.add(filename)
                    
                    # Record all N_JSON pages in JSON (with scores)
                    hard_neg_filenames = []
                    for page_idx, score in hard_neg_pages:
                        page_num = page_idx + 1  # 1-indexed
                        filename = f"{pdf_hash}_{page_num}.png"
                        hard_neg_filenames.append({"page": filename, "score": score})
                    
                    all_hard_negatives[qid] = hard_neg_filenames
                    
            except Exception as e:
                print(f"  ❌ Error processing PDF {pdf_name}: {e}")
                failed_pdfs.append((pdf_name, str(e)))
                continue
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user. Saving progress...")
    
    finally:
        # Always save JSON, even if interrupted or errors occurred
        print(f"\n[4/4] Saving results...")
        output_data = {
            "metadata": {
                "model": MODEL_NAME,
                "n_save": N_SAVE,
                "n_json": N_JSON,
                "date": datetime.now().isoformat(),
                "total_queries": len(all_hard_negatives),
                "total_hard_negative_pages": len(saved_pages),
                "min_text_length": MIN_TEXT_LENGTH,
                "skipped_queries": skipped_queries,
                "failed_pdfs": len(failed_pdfs),
            },
            "queries": all_hard_negatives
        }
        
        with open(OUTPUT_JSON, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✅ Done!")
        print(f"   → Hard negative images: {OUTPUT_IMAGES_DIR}/ ({len(saved_pages)} pages)")
        print(f"   → Mapping JSON: {OUTPUT_JSON}")
        print(f"   → Processed PDFs: {processed_pdfs}")
        if skipped_queries > 0:
            print(f"   → Skipped queries (insufficient text): {skipped_queries}")
        if failed_pdfs:
            print(f"   → Failed PDFs: {len(failed_pdfs)}")
            for pdf_name, error in failed_pdfs:
                print(f"      - {pdf_name}: {error}")


if __name__ == "__main__":
    main()
