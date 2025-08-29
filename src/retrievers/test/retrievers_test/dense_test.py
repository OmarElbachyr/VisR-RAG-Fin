from retrievers.classes.sentence_transformer import SentenceTransformerRetriever
from evaluation.classes.query_qrel_builder import QueryQrelsBuilder
from evaluation.classes.document_provider import DocumentProvider
from evaluation.utils.ir_dataset import export_ir_dataset


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
    
    # Export IR dataset (one-time, centralized)
    export_ir_dataset(csv_path, output_dir=f"ir_export_{data_option}", format_type="csv")
    
    model_name = 'BAAI/bge-m3'  # "intfloat/multilingual-e5-large"
    retriever = SentenceTransformerRetriever(provider, model_name=model_name, is_instruct=False, device_map='cuda', batch_size=1)
    run = retriever.search(queries, agg='max')
    
    print(f"\n=== Dense Retriever ({model_name}) Results ===")
    metrics = retriever.evaluate(run, qrels, k_values, verbose=True, eval_lib=eval_lib)