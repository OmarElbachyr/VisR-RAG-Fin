import pandas as pd
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import pytrec_eval
from collections import defaultdict
import nltk
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM

# === 1. Load and deduplicate =================================================
df = (
    pd.read_csv("chunked_pages.csv")
      .drop_duplicates(subset=["chunk_id", "text_description"])
)

# === 2. Build chunk corpus ===================================================
chunk_ids = df["chunk_id"].tolist()
chunk_texts = df["text_description"].astype(str).tolist()
chunk_to_page = dict(zip(chunk_ids, df["page_number"]))
doc_tokens = [doc.split() for doc in chunk_texts]

# === 3. Prepare queries and qrels ============================================
queries = {}
qrels_chunk = {}
qrels_page = {}

for idx, (query_text, group) in enumerate(df.groupby("query"), start=1):
    qid = f"q{idx}"
    queries[qid] = query_text
    relevant_pages = set(group["page_number"])
    relevant_chunks = group["chunk_id"]
    qrels_chunk[qid] = {cid: 1 for cid in relevant_chunks}
    qrels_page[qid] = {str(p): 1 for p in relevant_pages}

# === 4. Scoring functions =====================================================
def score_bm25(queries, tokenized_docs):
    bm25 = BM25Okapi(tokenized_docs)
    run_chunk = {}
    for qid, qtext in queries.items():
        scores = bm25.get_scores(qtext.split())
        run_chunk[qid] = {cid: float(score) for cid, score in zip(chunk_ids, scores)}
    return run_chunk

def score_dense(queries, docs, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    doc_embs = model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
    run_chunk = {}
    for qid, qtext in queries.items():
        query_emb = model.encode([qtext], convert_to_numpy=True, normalize_embeddings=True)
        scores = cosine_similarity(query_emb, doc_embs)[0]
        run_chunk[qid] = {cid: float(score) for cid, score in zip(chunk_ids, scores)}
    return run_chunk

def score_hybrid_all(queries, documents, dense_model_name="sentence-transformers/all-MiniLM-L6-v2", bm25_weight=0.5, dense_weight=0.5):
    nltk.download("punkt", quiet=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenized_documents = [nltk.word_tokenize(doc.lower()) for doc in documents]
    bm25 = BM25Okapi(tokenized_documents)

    model = SentenceTransformer(dense_model_name, device=device)
    doc_embeddings = model.encode(documents, convert_to_numpy=True, normalize_embeddings=True)

    run_chunk = {}
    for qid, qtext in queries.items():
        tokenized_query = nltk.word_tokenize(qtext.lower())
        bm25_scores = bm25.get_scores(tokenized_query)

        query_embedding = model.encode([qtext], convert_to_numpy=True, normalize_embeddings=True)
        dense_scores = cosine_similarity(query_embedding, doc_embeddings)[0]

        bm25_scores_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
        dense_scores_norm = (dense_scores + 1) / 2

        hybrid_scores = bm25_weight * bm25_scores_norm + dense_weight * dense_scores_norm
        run_chunk[qid] = {cid: float(score) for cid, score in zip(chunk_ids, hybrid_scores)}

    return run_chunk


def score_splade_all(queries, documents, model_name="naver/splade-cocondenser-ensembledistil"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()

    def encode_texts(texts):
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            sparse_rep = torch.log1p(F.relu(logits))
            sparse_rep = torch.max(sparse_rep, dim=1).values
        return sparse_rep.cpu()

    doc_embs = encode_texts(documents)
    run_chunk = {}
    for qid, qtext in queries.items():
        query_emb = encode_texts([qtext])[0]
        scores = torch.matmul(doc_embs, query_emb).numpy()
        run_chunk[qid] = {cid: float(score) for cid, score in zip(chunk_ids, scores)}
    return run_chunk


def score_colbert(queries, docs, model_name="bert-base-uncased", max_length=128):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    def encode(texts):
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            token_embs = outputs.last_hidden_state
            return torch.nn.functional.normalize(token_embs, p=2, dim=-1)

    doc_embs = encode(docs)  # (N, L, H)
    run_chunk = {}

    for qid, qtext in queries.items():
        q_emb = encode([qtext])[0]  # (Lq, H)
        scores = []
        for d in doc_embs:
            sim = torch.matmul(q_emb, d.T)
            max_sim, _ = sim.max(dim=1)
            scores.append(max_sim.sum().item())
        run_chunk[qid] = {cid: float(score) for cid, score in zip(chunk_ids, scores)}
    return run_chunk

# === 5. Utility: chunk → page aggregation ====================================
def aggregate_chunk_scores(run_chunk, agg="max"):
    run_page = {}
    for qid, scores in run_chunk.items():
        page_scores = defaultdict(list)
        for cid, score in scores.items():
            page = chunk_to_page.get(cid)
            if page is not None:
                page_scores[page].append(score)

        if agg == "max":
            run_page[qid] = {str(p): float(np.max(v)) for p, v in page_scores.items()}
        elif agg == "mean":
            run_page[qid] = {str(p): float(np.mean(v)) for p, v in page_scores.items()}
        elif agg == "sum":
            run_page[qid] = {str(p): float(np.sum(v)) for p, v in page_scores.items()}
        else:
            raise ValueError("Unsupported aggregation method.")
    return run_page

# === 6. Evaluation function ===================================================
def evaluate(run, qrels, label):
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut", "map_cut", "recall", "P"})
    results = evaluator.evaluate(run)
    print(f"\n=== {label} ===")
    for k in [1, 3, 5, 10]:
        ndcg = np.mean([res.get(f"ndcg_cut_{k}", 0.0) for res in results.values()])
        m_ap = np.mean([res.get(f"map_cut_{k}", 0.0) for res in results.values()])
        recall = np.mean([res.get(f"recall_{k}", 0.0) for res in results.values()])
        prec = np.mean([res.get(f"P_{k}", 0.0) for res in results.values()])
        print(f"K={k:<2}  NDCG:{ndcg:.4f}  MAP:{m_ap:.4f}  Recall:{recall:.4f}  Precision:{prec:.4f}")

# === 7. Run models and evaluate ===============================================
bm25_chunk = score_bm25(queries, doc_tokens)
dense_chunk = score_dense(queries, chunk_texts)
colbert_chunk = score_colbert(queries, chunk_texts)
hybrid_chunk = score_hybrid_all(queries, chunk_texts)
splade_chunk = score_splade_all(queries, chunk_texts)

# Chunk-level eval
evaluate(bm25_chunk, qrels_chunk, "BM25 (Chunk-Level)")
evaluate(dense_chunk, qrels_chunk, "Dense (Chunk-Level)")
evaluate(colbert_chunk, qrels_chunk, "ColBERT (Chunk-Level)")
evaluate(hybrid_chunk, qrels_chunk, "Hybrid (Chunk-Level)")
evaluate(splade_chunk, qrels_chunk, "SPLADE (Chunk-Level)")

# Page-level eval
evaluate(aggregate_chunk_scores(bm25_chunk), qrels_page, "BM25 (Page-Level)")
evaluate(aggregate_chunk_scores(dense_chunk), qrels_page, "Dense (Page-Level)")
evaluate(aggregate_chunk_scores(colbert_chunk), qrels_page, "ColBERT (Page-Level)")
evaluate(aggregate_chunk_scores(hybrid_chunk),qrels_page,"Hybrid (Page-Level)")
evaluate(aggregate_chunk_scores(splade_chunk),qrels_page,"SPLADE (Page-Level)")