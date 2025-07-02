import json
import time
import sys 
import os 
from io import StringIO
from contextlib import redirect_stdout
sys.path.append(os.path.abspath("/home/omar/projects/vqa-ir-qa/src"))

from retrievers.siglip import SigLIPRetriever
from retrievers.clip import ClipRetriever

from evaluation.document_provider import DocumentProvider
from evaluation.query_qrel_builder import QueryQrelsBuilder

def test_retriever(retriever_class, provider, queries, qrels, results, agg="max", txt_file_path=None, **kwargs):
    # Use model_name as key if provided, otherwise use class name
    key = kwargs.get("model_name", retriever_class.__name__)
    print(f"\nTesting {key}...")
    
    # --- measure indexing time (constructor) ---
    idx_start = time.time()
    retriever = retriever_class(provider, **kwargs)
    indexing_time = time.time() - idx_start

    # --- measure retrieval time (search) ---
    search_start = time.time()
    if "agg" in retriever.search.__code__.co_varnames:
        run = retriever.search(queries, agg=agg)
    else:
        run = retriever.search(queries)
    retrieval_time = time.time() - search_start
    
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
            txtf.write("-" * 80 + "\n")

    # Store mertrics
    results[key] = {
        "metrics": metrics,
        "indexing_time": indexing_time,
        "retrieval_time": retrieval_time
    }

if __name__ == "__main__":
    data_option = "all_pages"  # Set to "annotated_pages" for annotated data, "all_pages" for all sampled data

    if data_option == "annotated_pages":
        csv_path = "src/dataset/chunks/chunked_pages.csv"
    elif data_option == "all_pages":
        csv_path = "src/dataset/chunks/chunked_sampled_pages.csv"
    else:
        raise ValueError("Invalid data_option. Choose 'annotated_pages' or 'all_pages'.")
    
    image_dir = "data/pages"
    results_path = f"src/results/vision_retrievers_results_{data_option}.json"
    txt_results_path = f"src/results/vision_retrievers_results_{data_option}.txt"

    provider = DocumentProvider(csv_path, use_nltk_preprocessor=True)
    
    with open(txt_results_path, "w") as txtf:
        txtf.write(f"Provider Stats:\n{provider.stats}\n")
        txtf.write("=" * 80 + "\n\n")
    
    print(provider.stats)
    queries, qrels = QueryQrelsBuilder(csv_path).build()

    results = {
        "docs_stats": provider.stats,
        "models": {}
    }

    test_retriever(
        ClipRetriever, provider, queries, qrels, results["models"],
        model_name="openai/clip-vit-base-patch32", image_dir=image_dir, txt_file_path=txt_results_path, batch_size=32, device_map="cuda"
    )
    test_retriever(
        SigLIPRetriever, provider, queries, qrels, results["models"],
        model_name="google/siglip-base-patch16-224", image_dir=image_dir, txt_file_path=txt_results_path, batch_size=32, device_map="cuda"
    )

    # Save results to JSON file
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")
    print(f"Printed output saved to {txt_results_path}")