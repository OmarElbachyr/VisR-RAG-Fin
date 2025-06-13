import sys
import os
import json
import time

sys.path.append(os.path.abspath("src"))

from retrievers.bm25 import BM25Retriever
from retrievers.sentence_transformer import SentenceTransformerRetriever
from retrievers.colbert import ColBERTRetriever
from retrievers.splade import SpladeRetriever
from evaluation.document_provider import DocumentProvider
from evaluation.query_qrel_builder import QueryQrelsBuilder

def test_retriever(retriever_class, provider, queries, qrels, results, **kwargs):
    print(f"Testing {retriever_class.__name__}...")
    start_time = time.time()
    retriever = retriever_class(provider, **kwargs)
    run = retriever.search(queries, agg="max")
    retrieval_time = time.time() - start_time
    metrics = retriever.evaluate(run, qrels, verbose=True)
    print(metrics)

    results[retriever_class.__name__] = {
        "metrics": metrics,
        "retrieval_time": retrieval_time
    }

if __name__ == "__main__":
    csv_path = "src/dataset/chunked_pages.csv"

    provider = DocumentProvider(csv_path, use_nltk_preprocessor=True)
    print(provider.stats)
    queries, qrels = QueryQrelsBuilder(csv_path).build()

    results = {}

    test_retriever(BM25Retriever, provider, queries, qrels, results)
    test_retriever(SentenceTransformerRetriever, provider, queries, qrels, results, model_name="BAAI/bge-m3")
    test_retriever(ColBERTRetriever, provider, queries, qrels, results, model_name="lightonai/GTE-ModernColBERT-v1")
    test_retriever(SpladeRetriever, provider, queries, qrels, results, model_name="naver/splade-v3")

    with open("src/results/retriever_results.json", "w") as f:
        json.dump(results, f, indent=4)
