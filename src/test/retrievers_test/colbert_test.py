from retrievers.colbert import ColBERTRetriever 
from evaluation.query_qrel_builder import QueryQrelsBuilder
from evaluation.document_provider import DocumentProvider


if __name__ == "__main__":
    
    csv_path = "src/dataset/chunked_pages.csv"
    provider = DocumentProvider(csv_path)
    print(f'Stats: {provider.stats}')
    
    queries, qrels = QueryQrelsBuilder(csv_path).build()


    retriever = ColBERTRetriever(provider, 
                                model_name="lightonai/GTE-ModernColBERT-v1", 
                                index_folder="indexes/pylate-index", 
                                index_name="index", 
                                override=True,
                                batch_size=32,
                                device_map="cuda")
    
    run = retriever.search(queries, k=-1, agg='max') # -1 for all documents

    # qid = 'q1'
    # print(qid, list(run[qid].items())[:5])

    # metrics = retriever.evaluate(queries, qrels)
    retriever.evaluate(run, qrels, verbose=True)