import sys 
import os 
sys.path.append(os.path.abspath("C:/Users/laura.bernardy/OneDrive - University of Luxembourg/Documents/GitHub/vqa-ir-qa/src"))
from retrievers.colipali import ColipaliRetriever
from evaluation.document_provider import DocumentProvider
from evaluation.query_qrel_builder import QueryQrelsBuilder


if __name__ == "__main__":
    csv_path = "src/dataset/chunked_pages.csv"

    provider = DocumentProvider(csv_path, use_nltk_preprocessor=False)
    queries, qrels = QueryQrelsBuilder(csv_path).build()

    colipali = ColipaliRetriever(provider)
    run = colipali.search(queries, agg="max")  # max, mean, sum

    metrics = colipali.evaluate(run, qrels, verbose=True)