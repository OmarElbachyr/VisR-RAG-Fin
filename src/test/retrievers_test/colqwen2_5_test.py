import sys 
import os 
sys.path.append(os.path.abspath("/home/omar/projects/vqa-ir-qa/src"))

from retrievers.colqwen2_5 import ColQwen2_5Retriever
from evaluation.document_provider import DocumentProvider
from evaluation.query_qrel_builder import QueryQrelsBuilder

if __name__ == "__main__":
    csv_path = "src/dataset/chunked_pages.csv"
    image_dir = "data/pages"

    # Load document‐to‐page mapping and stats
    provider = DocumentProvider(csv_path)
    print(f"Stats: {provider.stats}")

    # Build your queries and relevance judgments
    queries, qrels = QueryQrelsBuilder(csv_path).build()

    names = [
        'vidore/colqwen2-v0.1-merged', # needs different processor ColQwen2 processor -> ColQwen2_Retriever class
        "vidore/colqwen2.5-v0.2", # works
        "tsystems/colqwen2.5-3b-multilingual-v1.0",  # works (the only multilingual model) and they have colqwen2 models, 
        "nomic-ai/colnomic-embed-multimodal-3b", # works, they have single vector multimodal models, e.g., nomic-ai/nomic-embed-multimodal-7b
        "nomic-ai/colnomic-embed-multimodal-7b", # works
        "Metric-AI/ColQwen2.5-3b-multilingual-v1.0", # works
        "Metric-AI/ColQwen2.5-7b-multilingual-v1.0", # out of memory
    ]

    small_model_names = [  # to test (needs different processor) -> ColSmol_Retriever class
        "vidore/colSmol-256M",
        "vidore/colSmol-500M",
    ]
    # Initialize the ColQwen2.5 retriever
    retriever = ColQwen2_5Retriever(
        provider=provider,
        image_dir=image_dir,
        model_name='Metric-AI/ColQwen2.5-7b-multilingual-v1.0',
        device_map="cuda",        
        batch_size=4,            # for image‐embedding
    )

    # Run retrieval and evaluate
    run = retriever.search(queries, batch_size=8)
    retriever.evaluate(run, qrels, verbose=True)
