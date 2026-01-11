import json
import time
import sys 
import os 
import argparse
from io import StringIO
from contextlib import redirect_stdout

from retrievers.classes.bm25 import BM25Retriever
from retrievers.classes.sentence_transformer import SentenceTransformerRetriever
from retrievers.classes.colbert import ColBERTRetriever
from retrievers.classes.splade import SpladeRetriever

from evaluation.classes.document_provider import DocumentProvider
from evaluation.classes.query_qrel_builder import QueryQrelsBuilder

def test_retriever(retriever_class, provider, queries, qrels, results, results_dir, agg="max", txt_file_path=None, **kwargs):
    # Use model_name as key if provided, otherwise use class name
    key = kwargs.get("model_name", retriever_class.__name__)
    
    # Skip if model result already exists
    if key in results:
        print(f"\nSkipping {key} - results already exist")
        return
    
    print(f"\nTesting {key}...")
    
    # --- measure indexing time (constructor) ---
    idx_start = time.time()
    retriever = retriever_class(provider, **kwargs)
    indexing_time = time.time() - idx_start

    # --- measure retrieval time (search) ---
    search_start = time.time()
    run = retriever.search(queries, agg=agg)
    retrieval_time = time.time() - search_start
    
    # Sort run dictionary by values for each query and save to data/retrieved_pages
    sorted_run = {}
    for qid, page_scores in run.items():
        # Sort pages by score in descending order
        sorted_pages = dict(sorted(page_scores.items(), key=lambda x: x[1], reverse=True))
        sorted_run[qid] = {
            "query": queries[qid],  # Include the actual query text
            "results": sorted_pages
        }
    
    # Save sorted run to data/retrieved_pages
    safe_key = key.replace("/", "_").replace(":", "_")  # Make filename safe
    run_file_path = f"{results_dir}/retrieved_pages/{safe_key}_sorted_run.json"
    os.makedirs(os.path.dirname(run_file_path), exist_ok=True)
    with open(run_file_path, "w") as run_file:
        json.dump(sorted_run, run_file, indent=4)
    print(f"Sorted run saved to {run_file_path}")
    
    # Capture verbose output from evaluate method
    verbose_output = StringIO()
    with redirect_stdout(verbose_output):
        metrics = retriever.evaluate(run, qrels, verbose=True)
    
    # Print to console as well
    print(verbose_output.getvalue())

    # Save printed output to txt file if path provided
    if txt_file_path:
        with open(txt_file_path, "a") as txtf:
            txtf.write(f"\nTesting {key}...\n")
            txtf.write(verbose_output.getvalue())
            txtf.write(f"Indexing time: {indexing_time:.4f}s\n")
            txtf.write(f"Retrieval time: {retrieval_time:.4f}s\n")
            # Write index stats if available
            if hasattr(retriever, 'index_stats'):
                txtf.write(f"Index stats: {retriever.index_stats}\n")
            txtf.write(f"Sorted run saved to: {run_file_path}\n")
            txtf.write("-" * 80 + "\n")

    # Build results dict
    result_data = {
        "metrics": metrics,
        "indexing_time": indexing_time,
        "retrieval_time": retrieval_time,
        "sorted_run_path": run_file_path
    }
    
    # Add index stats if available
    if hasattr(retriever, 'index_stats'):
        result_data["index_stats"] = retriever.index_stats
    
    results[key] = result_data

def setup_paths_and_results(chunks_path):
    """Setup file paths and load existing results."""
    # Extract a name from the chunks_path for results directory
    chunks_filename = os.path.splitext(os.path.basename(chunks_path))[0]
    
    results_dir = f"src/retrievers/results/{chunks_filename}"
    os.makedirs(results_dir, exist_ok=True)
    results_path = f"{results_dir}/text_retrievers_results_{chunks_filename}.json"
    txt_results_path = f"{results_dir}/text_retrievers_results_{chunks_filename}.txt"
    
    return results_path, txt_results_path

def load_or_create_results(results_path, txt_results_path, provider):
    """Load existing results or create new results structure."""
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
        # Ensure models key exists
        if "models" not in results:
            results["models"] = {}
    else:
        results = {
            "docs_stats": provider.stats,
            "models": {}
        }
        # Write provider stats to txt if new file
        with open(txt_results_path, "a") as txtf:
            txtf.write(f"Provider Stats:\n{provider.stats}\n")
            txtf.write("=" * 80 + "\n\n")
    
    return results

def run_all_tests(provider, queries, qrels, results, results_dir, txt_results_path):
    """Run tests for all retriever models."""
    hard_negatives_chunks_path = "src/dataset/chunks/noise_pages_chunks/hard_negative_chunks.csv"  # Set to None to disable
    batch_size = 32
    
    test_retriever(BM25Retriever, provider, queries, qrels, results["models"], results_dir, agg="max", 
                   hard_negatives_chunks_path=hard_negatives_chunks_path, txt_file_path=txt_results_path)
    
    test_retriever(SentenceTransformerRetriever, provider, queries, qrels, results["models"], results_dir, agg="max",
                   model_name="BAAI/bge-m3", hard_negatives_chunks_path=hard_negatives_chunks_path, device_map='cuda', txt_file_path=txt_results_path, batch_size=batch_size)
    
    test_retriever(SentenceTransformerRetriever, provider, queries, qrels, results["models"], results_dir, agg="max", 
                   model_name="intfloat/multilingual-e5-large", is_instruct=False, hard_negatives_chunks_path=hard_negatives_chunks_path, device_map='cuda', txt_file_path=txt_results_path, batch_size=batch_size)
    
    test_retriever(SentenceTransformerRetriever, provider, queries, qrels, results["models"], results_dir, agg="max", 
                   model_name="intfloat/multilingual-e5-large-instruct", is_instruct=True, hard_negatives_chunks_path=hard_negatives_chunks_path, device_map='cuda', txt_file_path=txt_results_path, batch_size=batch_size)
    
    test_retriever(ColBERTRetriever, provider, queries, qrels, results["models"], results_dir, agg="max", 
                   model_name="colbert-ir/colbertv2.0", index_folder="indexes/pylate-index", index_name="index", override=True, hard_negatives_chunks_path=hard_negatives_chunks_path, device_map='cuda', batch_size=batch_size, txt_file_path=txt_results_path)
    
    test_retriever(SpladeRetriever, provider, queries, qrels, results["models"], results_dir, agg="max", 
                   model_name="naver/splade-v3", hard_negatives_chunks_path=hard_negatives_chunks_path,
                   batch_size=batch_size, device_map="cuda", txt_file_path=txt_results_path)

def main():
    parser = argparse.ArgumentParser(description="Test text retrievers")
    parser.add_argument("--chunks_path", type=str,
                       help="Path to the CSV file containing chunks data")
    
    args = parser.parse_args()
    
    # args.chunks_path = "src/dataset/chunks/final_chunks/chunked_all_pages_windowed.csv"
    # args.chunks_path = "src/dataset/chunks/final_chunks/chunked_pages_category_A.csv"
    args.chunks_path = "src/dataset/chunks/second_pass/chunked_pages_second_pass.csv"
    
    # Setup paths and load results
    results_path, txt_results_path = setup_paths_and_results(args.chunks_path)
    results_dir = os.path.dirname(results_path)
    
    # Initialize provider and load/create results
    provider = DocumentProvider(args.chunks_path, use_nltk_preprocessor=True)
    results = load_or_create_results(results_path, txt_results_path, provider)
    
    print(provider.stats)
    queries, qrels = QueryQrelsBuilder(args.chunks_path).build()
    print(f">>> Results directory: {results_dir}\n")

    # Run all tests
    run_all_tests(provider, queries, qrels, results, results_dir, txt_results_path)

    # Save results to JSON file
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")
    print(f"Printed output appended to {txt_results_path}")

if __name__ == "__main__":
    main()