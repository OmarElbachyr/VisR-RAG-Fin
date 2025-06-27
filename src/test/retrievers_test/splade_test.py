import os
import sys

sys.path.append(os.path.abspath("/home/omar/projects/vqa-ir-qa/src"))

from retrievers.splade import SpladeRetriever
from evaluation.document_provider import DocumentProvider
from evaluation.query_qrel_builder import QueryQrelsBuilder

if __name__ == "__main__":
    csv_path = "src/dataset/chunks/chunked_pages.csv"
    k_values = [1, 3, 5, 10]
    provider = DocumentProvider(csv_path)
    print(provider.stats)
    queries, qrels = QueryQrelsBuilder(csv_path).build()
    splade = SpladeRetriever(provider, model_name="naver/splade-v3")
    run = splade.search(queries, agg="max")  # max, mean, sum
    metrics = splade.evaluate(run, qrels, k_values, verbose=True)
