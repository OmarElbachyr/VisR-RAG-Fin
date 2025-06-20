from retrievers.siglip import SigLIPRetriever
from evaluation.document_provider import DocumentProvider
from evaluation.query_qrel_builder import QueryQrelsBuilder

if __name__ == "__main__":
    csv_path = "src/dataset/chunked_pages.csv"
    image_dir = "data/pages"

    provider = DocumentProvider(csv_path)
    print(f"Stats: {provider.stats}")

    queries, qrels = QueryQrelsBuilder(csv_path).build()

    siglip = SigLIPRetriever(
        provider,
        image_dir=image_dir,
        model_name="google/siglip-base-patch16-224",
        device_map="cuda",
        batch_size=32
    )

    run = siglip.search(queries, batch_size=8)
    siglip.evaluate(run, qrels, verbose=True)
