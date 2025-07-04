from retrievers.bm25 import BM25Retriever
from evaluation.document_provider import DocumentProvider
from evaluation.query_qrel_builder import QueryQrelsBuilder


if __name__ == "__main__":
    data_option = "annotated_pages"  # Set to "annotated_pages" for annotated data, "all_pages" for all sampled data

    if data_option == "annotated_pages":
        csv_path = "src/dataset/chunks/chunked_pages.csv"
    elif data_option == "all_pages":
        csv_path = "src/dataset/chunks/chunked_sampled_pages.csv"
    else:
        raise ValueError("Invalid data_option. Choose 'annotated_pages' or 'all_pages'.")
    k_values = [1, 3, 5, 10]
    eval_lib = 'ir_measures'  # 'ir_measures', 'pytrec_eval'

    provider = DocumentProvider(csv_path, use_nltk_preprocessor=True)
    print(provider.stats)
    queries, qrels = QueryQrelsBuilder(csv_path).build()

    bm25 = BM25Retriever(provider)
    run = bm25.search(queries, agg="max")  # max, mean, sum
    
    print(f"\n=== BM25 Retriever Results ===")
    metrics = bm25.evaluate(run, qrels, k_values, verbose=True, eval_lib=eval_lib)