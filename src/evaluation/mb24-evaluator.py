import pandas as pd
from rank_bm25 import BM25Okapi
import pytrec_eval
from collections import defaultdict
import numpy as np

# === 1. Load and deduplicate =================================================
df = (
    pd.read_csv("src/dataset/chunked_pages.csv")
      .drop_duplicates(subset=["chunk_id"])
)

# === 2. Build chunk corpus ===================================================
doc_ids = df["chunk_id"].tolist()
docs    = df["text_description"].astype(str).tolist()
tokenized_corpus = [doc.split() for doc in docs]

bm25 = BM25Okapi(tokenized_corpus)

# Map chunk → page once for fast look-up
chunk_to_page = dict(zip(df["chunk_id"], df["page_number"]))

# === 3. Prepare queries and qrels ============================================
queries = {}
qrels   = {}

for idx, (query_text, group) in enumerate(df.groupby("query", sort=False), start=1):
    qid = f"q{idx}"
    queries[qid] = query_text
    page_ids = (
        group["image_filename"].astype(str)
            .str.rsplit(".", n=1)
            .str[0]
            .unique()
    )
    qrels[qid] = {pid: 1 for pid in page_ids}

# === 4. Score & aggregate ====================================================
run = {}                # {qid: {page_id: score}}
for qid, qtext in queries.items():
    tokenized_q = qtext.split()
    chunk_scores = bm25.get_scores(tokenized_q)  # same order as doc_ids

    # roll up to pages
    page_scores = defaultdict(list)
    for doc_id, score in zip(doc_ids, chunk_scores):
        page = chunk_to_page.get(doc_id)
        if page is not None:
            page_scores[page].append(score)

    run[qid] = {str(p): float(np.max(scores)) for p, scores in page_scores.items()}

# === 5. Evaluate with pytrec_eval ============================================
k_values = [1, 3, 5, 10]
metrics_requested = {"ndcg_cut", "map_cut", "recall", "P"}
evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics_requested)
results = evaluator.evaluate(run)

# === 6. Aggregate and print ===================================================
print("=== BM25 Evaluation (Page Level) ===")
for k in k_values:
    ndcg   = np.mean([res.get(f"ndcg_cut_{k}", 0.0) for res in results.values()])
    m_ap   = np.mean([res.get(f"map_cut_{k}", 0.0)  for res in results.values()])
    recall = np.mean([res.get(f"recall_{k}", 0.0)   for res in results.values()])
    prec   = np.mean([res.get(f"P_{k}", 0.0)        for res in results.values()])

    print(f"K={k:<2}  NDCG:{ndcg:.4f}  MAP:{m_ap:.4f}  R:{recall:.4f}  P:{prec:.4f}")
