from evaluation.classes.document_provider import DocumentProvider
from evaluation.classes.query_qrel_builder import QueryQrelsBuilder


if __name__ == "__main__":
    csv_path = "src/dataset/chunks/chunked_pages.csv"
    provider = DocumentProvider(csv_path, use_nltk_preprocessor=False)
    # print(len(provider.ids), "docs", len(provider.chunk_to_page), "chunk→page pairs")
    # first_id = provider.ids[0]
    # print(first_id, "-> page", provider.chunk_to_page[first_id])
    print(provider.stats)
    queries, qrels = QueryQrelsBuilder(csv_path).build()
