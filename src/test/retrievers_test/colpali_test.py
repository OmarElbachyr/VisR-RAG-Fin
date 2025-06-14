from retrievers.colpali import ColPaliRetriever
from evaluation.document_provider import DocumentProvider
from evaluation.query_qrel_builder import QueryQrelsBuilder

if __name__ == "__main__":
    csv_path = "src/dataset/chunked_pages.csv"
    image_dir = "data/pages"

    # load provider and print stats
    provider = DocumentProvider(csv_path)
    print(f"Stats: {provider.stats}")

    # build queries & qrels
    queries, qrels = QueryQrelsBuilder(csv_path).build()

    # initialize ColPali retriever (pages indexed directly)
    colpali = ColPaliRetriever(
        provider,
        image_dir=image_dir,
        model_name="vidore/colpali-v1.3",
        device_map="cuda",
        batch_size=32
    )

    # run search (one score per page)
    run = colpali.search(queries, batch_size=8)

    # evaluate
    colpali.evaluate(run, qrels, verbose=True)
