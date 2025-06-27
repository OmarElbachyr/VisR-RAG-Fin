import sys 
import os 
sys.path.append(os.path.abspath("/home/omar/projects/vqa-ir-qa/src"))

from retrievers.colqwen2_5 import ColQwen2_5Retriever
from evaluation.document_provider import DocumentProvider
from evaluation.query_qrel_builder import QueryQrelsBuilder

if __name__ == "__main__":
    csv_path = "src/dataset/chunks/chunked_pages.csv"
    k_values = [1, 3, 5, 10]
    image_dir = "data/pages"
    provider = DocumentProvider(csv_path)
    print(provider.stats)
    queries, qrels = QueryQrelsBuilder(csv_path).build()
    retriever = ColQwen2_5Retriever(
        provider=provider,
        image_dir=image_dir,
        model_name='tsystems/colqwen2.5-3b-multilingual-v1.0',
        device_map="cuda",        
        batch_size=8,            # for image-embedding
    )
    run = retriever.search(queries, batch_size=8)
    metrics = retriever.evaluate(run, qrels, k_values, verbose=True)
