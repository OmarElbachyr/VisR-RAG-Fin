"""
Hard Negative Mining using ColQwen2.5 (Vision-based)

For each query, retrieves similar pages from the same PDF and selects
top N pages (excluding the positive) as hard negatives.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import fitz  # PyMuPDF
import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

# =============================================================================
# CONFIGURATION - Edit these variables
# =============================================================================

# Number of hard negatives per query
N_SAVE = 5   # Number of pages to actually save to disk
N_JSON = 50   # Number of pages to record in JSON

# Paths
BENCHMARK_CSV = "src/dataset/chunks/final_chunks/chunked_pages_category_A.csv"
PDF_DIR = "data/indexed_pdfs"
OUTPUT_IMAGES_DIR = "data/hard_negative_pages_colqwen"
OUTPUT_JSON = "data/hard_negatives_mining/query_hard_negatives_colqwen.json"

# Model settings
MODEL_NAME = "nomic-ai/colnomic-embed-multimodal-7b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
RESIZE_RATIO = 0.5
MIN_IMAGE_DIMENSION = 64  # Minimum width/height (Qwen2 requires >= 28, use 64 for safety)

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


def validate_image(img: Image.Image, min_dim: int = MIN_IMAGE_DIMENSION) -> bool:
    """Check if image meets minimum dimension requirements."""
    return img.width >= min_dim and img.height >= min_dim


def pdf_to_images(pdf_path: str, resize_ratio: float = 1.0, min_dim: int = MIN_IMAGE_DIMENSION) -> tuple[list[Image.Image], list[int], list[int]]:
    """
    Convert all pages of a PDF to PIL Images.
    
    Returns:
        images: List of valid images
        valid_indices: Original page indices (0-based) for valid images
        skipped_indices: Original page indices (0-based) for skipped images
    """
    doc = fitz.open(pdf_path)
    images = []
    valid_indices = []
    skipped_indices = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Render at 150 DPI for good quality
        mat = fitz.Matrix(150 / 72, 150 / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Resize if needed
        if resize_ratio != 1.0:
            new_w = int(img.width * resize_ratio)
            new_h = int(img.height * resize_ratio)
            img = img.resize((new_w, new_h), resample=Image.BILINEAR)
        
        # Validate image dimensions
        if validate_image(img, min_dim):
            images.append(img)
            valid_indices.append(page_num)
        else:
            skipped_indices.append(page_num)
    
    doc.close()
    return images, valid_indices, skipped_indices


def save_page_image(pdf_path: str, page_number: int, output_path: str, resize_ratio: float = 1.0):
    """Save a single page from PDF as an image."""
    doc = fitz.open(pdf_path)
    page = doc[page_number]
    
    mat = fitz.Matrix(150 / 72, 150 / 72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    
    if resize_ratio != 1.0:
        new_w = int(img.width * resize_ratio)
        new_h = int(img.height * resize_ratio)
        img = img.resize((new_w, new_h), resample=Image.BILINEAR)
    
    img.save(output_path)
    doc.close()


def embed_images(model, processor, images: list[Image.Image], batch_size: int, device: str) -> torch.Tensor:
    """Embed a list of images using ColQwen2.5."""
    embs = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        inputs = processor.process_images(batch)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            emb = model(**inputs)
        embs.extend(emb)
    
    # Pad to same length
    max_chunks = max(e.size(0) for e in embs)
    dim = embs[0].shape[-1]
    padded = embs[0].new_zeros(len(embs), max_chunks, dim)
    
    for i, e in enumerate(embs):
        seq_len = min(e.size(0), max_chunks)
        padded[i, :seq_len] = e[:seq_len]
    
    return padded


def embed_queries(model, processor, queries: list[str], batch_size: int, device: str) -> list[torch.Tensor]:
    """Embed a list of queries using ColQwen2.5."""
    embs = []
    
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        inputs = processor.process_queries(batch)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            emb = model(**inputs)
        embs.extend(emb)
    
    return embs


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Hard Negative Mining with ColQwen2.5")
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
    model = ColQwen2_5.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        attn_implementation='eager',
    ).eval()
    processor = ColQwen2_5_Processor.from_pretrained(MODEL_NAME)
    print(f"  → Using device: {DEVICE}")
    
    # Process each PDF
    print(f"\n[3/4] Mining hard negatives (N_SAVE={N_SAVE}, N_JSON={N_JSON})...")
    
    all_hard_negatives = {}  # question_id -> list of hard negative filenames
    saved_pages = set()  # Track already saved pages to avoid duplicates
    processed_pdfs = 0
    skipped_pages_total = 0
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
                # Load and embed all pages from this PDF
                pdf_hash = pdf_name.replace(".pdf", "")
                images, valid_indices, skipped_indices = pdf_to_images(pdf_path, RESIZE_RATIO)
                num_pages = len(images)
                num_queries = len(queries)
                print(f"  📄 PDF: {pdf_name}")
                print(f"  → Valid pages: {num_pages} | Skipped (too small): {len(skipped_indices)} | Queries: {num_queries}")
                
                if skipped_indices:
                    skipped_pages_total += len(skipped_indices)
                    print(f"  ⚠️  Skipped pages (dim < {MIN_IMAGE_DIMENSION}): {[i+1 for i in skipped_indices]}")
                
                if num_pages == 0:
                    print(f"  ⚠️  No valid pages in PDF: {pdf_name}")
                    failed_pdfs.append((pdf_name, "No valid pages after filtering"))
                    continue
                
                # Embed pages
                page_embeddings = embed_images(model, processor, images, BATCH_SIZE, DEVICE)
                
                # Get all query texts for this PDF
                query_texts = [q["query"] for q in queries]
                query_embeddings = embed_queries(model, processor, query_texts, BATCH_SIZE, DEVICE)
                
                processed_pdfs += 1
                
                # For each query, find hard negatives
                for idx, query_info in enumerate(queries):
                    qid = query_info["question_id"]
                    positive_page_num = query_info["positive_page_number"]
                    
                    # Score query against all pages
                    q_emb = query_embeddings[idx].unsqueeze(0)  # [1, seq_len, dim]
                    scores = processor.score_multi_vector(q_emb, page_embeddings)  # [1, num_pages]
                    scores = scores.squeeze(0).cpu().numpy()  # [num_pages]
                    
                    # Rank pages by score (descending)
                    ranked_indices = scores.argsort()[::-1]
                    
                    # Select top N pages excluding the positive page
                    # Note: valid_indices maps embedding index -> original page index (0-based)
                    hard_neg_pages = []  # [(embedding_idx, original_page_num, score), ...]
                    for emb_idx in ranked_indices:
                        original_page_num = valid_indices[emb_idx] + 1  # 1-indexed page numbers
                        
                        # Skip the positive page
                        if original_page_num == positive_page_num:
                            continue
                        
                        hard_neg_pages.append((emb_idx, original_page_num, float(scores[emb_idx])))
                        
                        if len(hard_neg_pages) >= N_JSON:
                            break
                    
                    # Save hard negative page images (only N_SAVE to disk)
                    for emb_idx, original_page_num, score in hard_neg_pages[:N_SAVE]:
                        filename = f"{pdf_hash}_{original_page_num}.png"
                        output_path = os.path.join(OUTPUT_IMAGES_DIR, filename)
                        
                        # Save if not already saved
                        if filename not in saved_pages:
                            images[emb_idx].save(output_path)
                            saved_pages.add(filename)
                    
                    # Record all N_JSON pages in JSON (with scores)
                    hard_neg_filenames = []
                    for emb_idx, original_page_num, score in hard_neg_pages:
                        filename = f"{pdf_hash}_{original_page_num}.png"
                        hard_neg_filenames.append({"page": filename, "score": score})
                    
                    all_hard_negatives[qid] = hard_neg_filenames
                
                # Clear memory
                del images, page_embeddings, query_embeddings
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  ❌ Error processing PDF {pdf_name}: {e}")
                failed_pdfs.append((pdf_name, str(e)))
                torch.cuda.empty_cache()
                continue
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user. Saving progress...")
    
    finally:
        # Always save JSON, even if interrupted or errors occurred
        print(f"\n[4/4] Saving results...")
    
        # Save mapping JSON
        output_data = {
            "metadata": {
                "model": MODEL_NAME,
                "n_save": N_SAVE,
                "n_json": N_JSON,
                "min_image_dimension": MIN_IMAGE_DIMENSION,
                "date": datetime.now().isoformat(),
                "total_queries": len(all_hard_negatives),
                "total_hard_negative_pages": len(saved_pages),
                "resize_ratio": RESIZE_RATIO,
                "skipped_pages_total": skipped_pages_total,
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
        print(f"   → Skipped pages (too small): {skipped_pages_total}")
        if failed_pdfs:
            print(f"   → Failed PDFs: {len(failed_pdfs)}")
            for pdf_name, error in failed_pdfs:
                print(f"      - {pdf_name}: {error}")


if __name__ == "__main__":
    main()
