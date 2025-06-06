import pandas as pd
from rank_bm25 import BM25Okapi
import pytrec_eval
from collections import defaultdict
import numpy as np
#from vidore import ColPali, ColPaliProcessor
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel


# === Load dataset ===
df = pd.read_csv("chunked_pages.csv")
df = df.drop_duplicates(subset=["query", "chunk_id", "text_description"])

query_text = df["query"].iloc[0]
query_id = "q1"
k_values = [1, 3, 5, 10]
metrics = {'ndcg_cut', 'map_cut', 'recall', 'P'}

# Prepare chunk-level info
chunk_ids = df["chunk_id"].tolist()
chunk_texts = df["text_description"].astype(str).tolist()
chunk_to_page = dict(zip(df["chunk_id"], df["page_number"]))

# Qrels at chunk-level: all chunks from relevant pages are relevant
relevant_pages = set(df["page_number"].unique())
qrels_chunk = {
    query_id: {cid: 1 for cid, page in chunk_to_page.items() if page in relevant_pages}
}
# Qrels at page-level
qrels_page = {
    query_id: {str(p): 1 for p in relevant_pages}
}


def evaluate(run, qrels, label):
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)
    results = evaluator.evaluate(run)

    print(f"\n=== {label} Evaluation ===")
    for k in k_values:
        ndcg = np.mean([results[qid].get(f"ndcg_cut_{k}", 0.0) for qid in results])
        ap = np.mean([results[qid].get(f"map_cut_{k}", 0.0) for qid in results])
        recall = np.mean([results[qid].get(f"recall_{k}", 0.0) for qid in results])
        precision = np.mean([results[qid].get(f"P_{k}", 0.0) for qid in results])
        print(f"NDCG@{k}: {ndcg:.4f} | MAP@{k}: {ap:.4f} | Recall@{k}: {recall:.4f} | P@{k}: {precision:.4f}")
    print("-" * 50)


def get_bm25_scores(query, texts):
    tokenized_corpus = [doc.split() for doc in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split()
    return bm25.get_scores(tokenized_query)


#def get_colpali_scores(query, texts):
#    model = ColPali.from_pretrained(
#        "vidore/colpali-v1.2",
#        torch_dtype=torch.bfloat16,
#        device_map="cuda"
#    ).eval()
#    processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")
#
#    q_input = processor([query], return_tensors="pt").to("cuda")
#    with torch.no_grad():
#        q_feat = model.get_text_features(**q_input)
#
#    scores = []
#    for text in texts:
#        d_input = processor([text], return_tensors="pt").to("cuda")
#        with torch.no_grad():
#            d_feat = model.get_text_features(**d_input)
#        score = torch.nn.functional.cosine_similarity(q_feat, d_feat).item()
#        scores.append(score)
#    return scores


def get_dense_scores(query, documents, model_name="sentence-transformers/all-MiniLM-L6-v2", top_k=None):
    """
    Initialize dense retriever with SentenceTransformer, encode documents,
    then rank documents based on similarity to query.

    Args:
        query (str): The query string.
        documents (list of str): List of document strings.
        model_name (str): SentenceTransformer model name.
        top_k (int, optional): Number of top results to return. Return all if None.

    Returns:
        list of (int, float): List of tuples (doc_index, similarity_score) sorted by score desc.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)

    # Encode documents
    embeddings = model.encode(documents, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)

    # Encode query
    query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    # Compute cosine similarity scores
    scores = cosine_similarity(query_vec, embeddings)[0]
    return scores


def get_colbert_scores(query, documents, model_name="bert-base-uncased", max_length=128, device=None):
    """
    Loads ColBERT-style BERT encoder and tokenizer, encodes documents and query,
    ranks documents by ColBERT max-similarity scoring.

    Args:
        query (str): Query string.
        documents (list of str): List of documents.
        model_name (str): HuggingFace transformer model to use.
        max_length (int): Max tokens for documents/query.
        device (str or None): 'cuda' or 'cpu'. Auto-detect if None.

    Returns:
        list of (int, float): Document indices and similarity scores sorted descending.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    def encode(texts):
        # Tokenize and encode to get token embeddings: (batch_size, seq_len, hidden_dim)
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Use last hidden state as token embeddings
            token_embeddings = outputs.last_hidden_state  # (B, L, H)
            # Normalize embeddings for cosine similarity
            token_embeddings = torch.nn.functional.normalize(token_embeddings, p=2, dim=-1)
        return token_embeddings

    # Encode query and docs
    query_emb = encode([query])[0]           # (query_len, hidden_dim)
    doc_embs = encode(documents)              # (num_docs, doc_len, hidden_dim)

    scores = []
    # For each document, compute maxsim:
    # For each query token, find max cosine sim with any doc token, then sum over query tokens
    for i in range(len(documents)):
        doc_emb = doc_embs[i]                 # (doc_len, hidden_dim)
        # Cosine similarity matrix: (query_len, doc_len)
        sim_matrix = torch.matmul(query_emb, doc_emb.T)
        # Max over doc tokens per query token
        max_sim_per_query_token, _ = sim_matrix.max(dim=1)
        score = max_sim_per_query_token.sum().item()
        scores.append(score)
    return scores

def evaluate_chunk_level(model_name, scores):
    run = {query_id: {cid: float(score) for cid, score in zip(chunk_ids, scores)}}
    evaluate(run, qrels_chunk, f"{model_name} (Chunk-Level)")


def evaluate_page_level(model_name, scores, agg="max"):
    # Aggregate chunk scores to pages
    page_scores = defaultdict(list)
    for cid, score in zip(chunk_ids, scores):
        page = chunk_to_page[cid]
        page_scores[page].append(score)

    if agg == "max":
        agg_scores = {str(p): float(np.max(s)) for p, s in page_scores.items()}
    elif agg == "mean":
        agg_scores = {str(p): float(np.mean(s)) for p, s in page_scores.items()}
    elif agg == "sum":
        agg_scores = {str(p): float(np.sum(s)) for p, s in page_scores.items()}
    else:
        raise ValueError("Unknown aggregation method")

    run = {query_id: agg_scores}
    evaluate(run, qrels_page, f"{model_name} (Page-Level, {agg})")


# === Run both models ===
bm25_scores = get_bm25_scores(query_text, chunk_texts)
dense_scores = get_dense_scores(query_text, chunk_texts)
colbert_scores = get_colbert_scores(query_text,chunk_texts )
#colpali_scores = get_colpali_scores(query_text, chunk_texts)

# === Evaluate ===
evaluate_chunk_level("BM25", bm25_scores)
#evaluate_chunk_level("ColPali", colpali_scores)
evaluate_chunk_level("Dense", dense_scores)
evaluate_chunk_level("Colbert", colbert_scores)

evaluate_page_level("BM25", bm25_scores, agg="max")
#evaluate_page_level("ColPali", colpali_scores)
evaluate_page_level("Dense", dense_scores, agg="max")
evaluate_chunk_level("Colbert", colbert_scores)