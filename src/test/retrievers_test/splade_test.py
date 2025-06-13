from retrievers.splade import SpladeRetriever
from evaluation.document_provider import DocumentProvider
from evaluation.query_qrel_builder import QueryQrelsBuilder

if __name__ == "__main__":
    csv_path = "src/dataset/chunked_pages.csv"

    provider = DocumentProvider(csv_path, use_nltk_preprocessor=False)
    queries, qrels = QueryQrelsBuilder(csv_path).build()

    splade = SpladeRetriever(provider, model_name="naver/splade-v3")
    run = splade.search(queries, agg="max")  # max, mean, sum


    metrics = splade.evaluate(run, qrels, verbose=True)
