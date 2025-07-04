
from retrievers.biqwen2_5 import BiQwen2_5Retriever
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
    image_dir = "data/pages"
    provider = DocumentProvider(csv_path)
    print(provider.stats)
    queries, qrels = QueryQrelsBuilder(csv_path).build()
    model_name = 'nomic-ai/nomic-embed-multimodal-7b'
    retriever = BiQwen2_5Retriever(
        provider=provider,
        image_dir=image_dir,
        model_name=model_name,
        device_map="cuda",        
        batch_size=16,            # for image-embedding
    )
    run = retriever.search(queries, batch_size=8)
    
    print(f"\n=== BiQwen2.5 ({model_name}) Results ===")
    metrics = retriever.evaluate(run, qrels, k_values, verbose=True, eval_lib=eval_lib)
