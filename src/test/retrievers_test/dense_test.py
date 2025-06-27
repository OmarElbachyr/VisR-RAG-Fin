import os
import sys

sys.path.append(os.path.abspath("/home/omar/projects/vqa-ir-qa/src"))

from retrievers.sentence_transformer import SentenceTransformerRetriever
from evaluation.query_qrel_builder import QueryQrelsBuilder
from evaluation.document_provider import DocumentProvider


if __name__ == "__main__":
    csv_path = "src/dataset/chunks/chunked_pages.csv"
    k_values = [1, 3, 5, 10]
    provider = DocumentProvider(csv_path)
    print(provider.stats)
    queries, qrels = QueryQrelsBuilder(csv_path).build()
    retriever = SentenceTransformerRetriever(provider, model_name='intfloat/multilingual-e5-large', is_instruct=False, device_map='cuda')
    run = retriever.search(queries, agg='max')
    metrics = retriever.evaluate(run, qrels, k_values, verbose=True)