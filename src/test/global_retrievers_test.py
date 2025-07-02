import json
import time
import sys 
import os 
sys.path.append(os.path.abspath("/home/laura/vqa-ir-qa/src"))
from retrievers.bm25 import BM25Retriever
from retrievers.sentence_transformer import SentenceTransformerRetriever
from retrievers.colbert import ColBERTRetriever
from retrievers.splade import SpladeRetriever
from retrievers.colpali import ColPaliRetriever
from retrievers.clip import ClipRetriever
from retrievers.colsmol import ColSmol
from retrievers.colqwen2 import ColQwen2Retriever
from retrievers.colqwen2_5 import ColQwen2_5Retriever
from retrievers.siglip import SigLIPRetriever
from evaluation.document_provider import DocumentProvider
from evaluation.query_qrel_builder import QueryQrelsBuilder

def test_retriever(retriever_class, provider, queries, qrels, results, agg="max", **kwargs):
    print(f"Testing {retriever_class.__name__}...")
    
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
    
    metrics = retriever.evaluate(run, qrels, verbose=True)
    print(metrics)

    # Store mertrics
    results[retriever_class.__name__] = {
        "metrics": metrics,
        "indexing_time": indexing_time,
        "retrieval_time": retrieval_time
    }

if __name__ == "__main__":
    csv_path = "/home/laura/vqa-ir-qa/src/dataset/chunks/chunked_pages.csv"
    image_dir = "data/pages"
    results_path = "results/retriever_results_annotated.json"

    provider = DocumentProvider(csv_path, use_nltk_preprocessor=True)
    print(provider.stats)
    queries, qrels = QueryQrelsBuilder(csv_path).build()

    results = {
        "docs_stats": provider.stats,
        "models": {}
    }

    test_retriever(BM25Retriever, provider, queries, qrels, results["models"])
    test_retriever(SentenceTransformerRetriever, provider, queries, qrels, results["models"], model_name="BAAI/bge-m3")
    test_retriever(ColBERTRetriever, provider, queries, qrels, results["models"], model_name="colbert-ir/colbertv2.0", index_folder="indexes/pylate-index", index_name="index", override=True)
    test_retriever(SpladeRetriever, provider, queries, qrels, results["models"], model_name="naver/splade-v3",
                   batch_size=32, device="cuda")
    test_retriever(ClipRetriever, provider, queries, qrels, results["models"], model_name="openai/clip-vit-base-patch32",
                   image_dir=image_dir, batch_size=32, device_map="cuda")
    test_retriever(SigLIPRetriever, provider, queries, qrels, results["models"], model_name="google/siglip-base-patch16-224",
                   image_dir=image_dir, batch_size=32, device_map="cuda")
    test_retriever(ColPaliRetriever, provider, queries, qrels, results["models"], model_name="vidore/colpali-v1.3",
                   image_dir=image_dir, batch_size=32)
    # ColQwen2 3B and 7B models
    # test_retriever(ColQwen2_5Retriever, provider, queries, qrels, results["models"], model_name="nomic-ai/colnomic-embed-multimodal-3b",
    #                image_dir=image_dir, batch_size=8)
    # test_retriever(ColQwen2_5Retriever, provider, queries, qrels, results["models"], model_name="nomic-ai/colnomic-embed-multimodal-7b",
    #                image_dir=image_dir, batch_size=8)
    # 
    # test_retriever(ColQwen2Retriever, provider, queries, qrels, results["models"], model_name="vidore/colqwen2-v1.0",
    #                image_dir=image_dir, batch_size=4)
    # test_retriever(ColSmol, provider, queries, qrels, results["models"], model_name="vidore/colSmol-500M",
    #                image_dir=image_dir, batch_size=16)
    # Save results to JSON file
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")