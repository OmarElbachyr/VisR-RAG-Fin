import sys 
import os 
sys.path.append(os.path.abspath("/home/omar/projects/vqa-ir-qa/src"))

from retrievers.bm25 import BM25Retriever
from evaluation.document_provider import DocumentProvider
from evaluation.query_qrel_builder import QueryQrelsBuilder


if __name__ == "__main__":
    csv_path = "src/dataset/chunks/chunked_pages.csv"
    k_values = [1, 3, 5, 10]

    provider = DocumentProvider(csv_path)
    print(provider.stats)
    queries, qrels = QueryQrelsBuilder(csv_path).build()

    bm25 = BM25Retriever(provider)
    run = bm25.search(queries, agg="max")  # max, mean, sum
    
    metrics = bm25.evaluate(run, qrels, k_values, verbose=True)