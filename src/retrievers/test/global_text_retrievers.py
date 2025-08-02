import json
import time
import sys 
import os 
from io import StringIO
from contextlib import redirect_stdout

from retrievers.classes.bm25 import BM25Retriever
from retrievers.classes.sentence_transformer import SentenceTransformerRetriever
from retrievers.classes.colbert import ColBERTRetriever
from retrievers.classes.splade import SpladeRetriever

from evaluation.classes.document_provider import DocumentProvider
from evaluation.classes.query_qrel_builder import QueryQrelsBuilder

def test_retriever(retriever_class, provider, queries, qrels, results, agg="max", txt_file_path=None, **kwargs):
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
    # Get data_option from main module if available, else default to "default"
    data_option = getattr(sys.modules["__main__"], "data_option", "default")
    run_file_path = f"data/retrieved_pages/{data_option}/{safe_key}_sorted_run.json"
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
            txtf.write(f"Sorted run saved to: {run_file_path}\n")
            txtf.write("-" * 80 + "\n")

    # Store mertrics
    results[key] = {
        "metrics": metrics,
        "indexing_time": indexing_time,
        "retrieval_time": retrieval_time,
        "sorted_run_path": run_file_path
    }

if __name__ == "__main__":
    data_option = "annotated_pages"  # Set to "annotated_pages" for annotated data, "all_pages" for all sampled data

    if data_option == "annotated_pages":
        csv_path = "src/dataset/chunks/chunked_pages.csv"
    elif data_option == "all_pages":
        csv_path = "src/dataset/chunks/chunked_sampled_pages.csv"
    else:
        raise ValueError("Invalid data_option. Choose 'annotated_pages' or 'all_pages'.")
    
    image_dir = "data/pages"
    results_dir = f"src/results/{data_option}"
    os.makedirs(results_dir, exist_ok=True)
    results_path = f"{results_dir}/text_retrievers_results_{data_option}.json"
    txt_results_path = f"{results_dir}/text_retrievers_results_{data_option}.txt"

    provider = DocumentProvider(csv_path, use_nltk_preprocessor=True)
    
    # Load existing results if present, else initialize
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
    
    print(provider.stats)
    queries, qrels = QueryQrelsBuilder(csv_path).build()

    test_retriever(BM25Retriever, provider, queries, qrels, results["models"], agg="max", 
                   txt_file_path=txt_results_path)
    
    test_retriever(SentenceTransformerRetriever, provider, queries, qrels, results["models"], agg="max",
                   model_name="BAAI/bge-m3", device_map='cuda', txt_file_path=txt_results_path, batch_size=32)
    
    test_retriever(SentenceTransformerRetriever, provider, queries, qrels, results["models"], agg="max", 
                   model_name="intfloat/multilingual-e5-large", is_instruct=False, device_map='cuda', txt_file_path=txt_results_path, batch_size=32)
    
    test_retriever(SentenceTransformerRetriever, provider, queries, qrels, results["models"], agg="max", 
                   model_name="intfloat/multilingual-e5-large-instruct", is_instruct=True, device_map='cuda', txt_file_path=txt_results_path, batch_size=32)
    
    test_retriever(ColBERTRetriever, provider, queries, qrels, results["models"], agg="max", 
                   model_name="colbert-ir/colbertv2.0", index_folder="indexes/pylate-index", index_name="index", override=True, device_map='cuda', batch_size=32, txt_file_path=txt_results_path)
    
    test_retriever(SpladeRetriever, provider, queries, qrels, results["models"], agg="max", 
                   model_name="naver/splade-v3",
                   batch_size=32, device_map="cuda", txt_file_path=txt_results_path)

    # Save results to JSON file (append/update)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")
    print(f"Printed output appended to {txt_results_path}")