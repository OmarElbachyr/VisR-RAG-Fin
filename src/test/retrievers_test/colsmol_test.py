import sys 
import os 
sys.path.append(os.path.abspath("/home/omar/projects/vqa-ir-qa/src"))

from retrievers.colsmol import ColSmol
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

    # Initialize the ColSmol retriever
    retriever = ColSmol(
        provider=provider,
        image_dir=image_dir,
        model_name='vidore/colSmol-500M',  
        device_map="cuda",        
        batch_size=32,            # for image‐embedding
    )

    # Run retrieval and evaluate
    run = retriever.search(queries, batch_size=8)
    retriever.evaluate(run, qrels, verbose=True)
