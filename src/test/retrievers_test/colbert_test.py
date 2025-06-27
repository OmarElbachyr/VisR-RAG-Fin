import os
import sys
sys.path.append(os.path.abspath("/home/omar/projects/vqa-ir-qa/src"))

from retrievers.colbert import ColBERTRetriever 
from evaluation.query_qrel_builder import QueryQrelsBuilder
from evaluation.document_provider import DocumentProvider


if __name__ == "__main__":
    csv_path = "src/dataset/chunks/chunked_pages.csv"
    k_values = [1, 3, 5, 10]
    provider = DocumentProvider(csv_path)
    print(provider.stats)
    queries, qrels = QueryQrelsBuilder(csv_path).build()
    retriever = ColBERTRetriever(provider, 
                                model_name="colbert-ir/colbertv2.0", 
                                index_folder="indexes/pylate-index", 
                                index_name="index", 
                                override=True,
                                batch_size=32,
                                device_map="cuda")
    run = retriever.search(queries, k=-1, agg='max')
    metrics = retriever.evaluate(run, qrels, k_values, verbose=True)