from retrievers.classes.splade import SpladeRetriever
from evaluation.classes.document_provider import DocumentProvider
from evaluation.classes.query_qrel_builder import QueryQrelsBuilder

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
    provider = DocumentProvider(csv_path)
    print(provider.stats)
    queries, qrels = QueryQrelsBuilder(csv_path).build()
    model_name = "naver/splade-v3"
    splade = SpladeRetriever(provider, model_name=model_name, batch_size=1, device_map="cuda")
    run = splade.search(queries, agg="max")  # max, mean, sum
    
    print(f"\n=== Splade ({model_name}) Results ===")
    metrics = splade.evaluate(run, qrels, k_values, verbose=True, eval_lib=eval_lib)
