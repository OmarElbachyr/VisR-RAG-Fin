# test/retrievers_test/colqwen2vl_test.py
from retrievers.colqwen2_5 import ColQwen25Retriever
from evaluation.document_provider import DocumentProvider
from evaluation.query_qrel_builder import QueryQrelsBuilder

if __name__ == "__main__":
    csv_path = "src/dataset/chunked_pages.csv"
    image_dir = "data/pages"

    provider = DocumentProvider(csv_path)
    print(f"Stats: {provider.stats}")

    queries, qrels = QueryQrelsBuilder(csv_path).build()

    retriever = ColQwen25Retriever(
        provider,
        image_dir=image_dir,
        model_name="Metric-AI/ColQwen2.5-3b-multilingual-v1.0",
        device_map="cuda",
        batch_size=16
    )

    run = retriever.search(queries, batch_size=8)
    retriever.evaluate(run, qrels, verbose=True)
