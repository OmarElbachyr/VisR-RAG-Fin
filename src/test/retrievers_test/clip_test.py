import os
import sys
sys.path.append(os.path.abspath("/home/omar/projects/vqa-ir-qa/src"))

from retrievers.clip import ClipRetriever
from evaluation.document_provider import DocumentProvider
from evaluation.query_qrel_builder import QueryQrelsBuilder

if __name__ == "__main__":
    csv_path = "src/dataset/chunks/chunked_pages.csv"
    k_values = [1, 3, 5, 10]
    image_dir = "data/pages"
    provider = DocumentProvider(csv_path)
    print(provider.stats)
    queries, qrels = QueryQrelsBuilder(csv_path).build()
    clip = ClipRetriever(
        provider,
        image_dir=image_dir,
        model_name="openai/clip-vit-base-patch32", 
        device_map="cuda",
        batch_size=32
    )
    run = clip.search(queries, batch_size=8)
    metrics = clip.evaluate(run, qrels, k_values, verbose=True)