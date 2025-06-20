import sys 
import os 
from retrievers.bm25 import BM25Retriever
from evaluation.document_provider import DocumentProvider
from evaluation.query_qrel_builder import QueryQrelsBuilder


if __name__ == "__main__":
    csv_path = "src/dataset/chunked_pages.csv"

    provider = DocumentProvider(csv_path, use_nltk_preprocessor=True)
    print(provider.stats)
    queries, qrels = QueryQrelsBuilder(csv_path).build()

    bm25 = BM25Retriever(provider)
    run = bm25.search(queries, agg="max")  # max, mean, sum

    # qid = 'q1'
    # print(qid, list(run[qid].items())[:5])  

    metrics = bm25.evaluate(run, qrels, verbose=True)