from retrievers.sentence_transformer import SentenceTransformerRetriever
from evaluation.query_qrel_builder import QueryQrelsBuilder
from evaluation.document_provider import DocumentProvider


if __name__ == "__main__":
    
    csv_path = "src/dataset/chunked_pages.csv"
    provider = DocumentProvider(csv_path)
    print(f'Stats: {provider.stats}')
    
    queries, qrels = QueryQrelsBuilder(csv_path).build()

    # instruct models test
    # retriever = SentenceTransformerRetriever(provider, model_name='intfloat/multilingual-e5-large-instruct', is_instruct=True, device_map='cuda')
    
    # non instruct models test
    retriever = SentenceTransformerRetriever(provider, model_name='intfloat/multilingual-e5-large', is_instruct=False, device_map='cuda')
    
    run = retriever.search(queries, agg='max')
    # qid = 'q1'
    # print(qid, list(run[qid].items())[:5])

    retriever.evaluate(run, qrels, verbose=True)