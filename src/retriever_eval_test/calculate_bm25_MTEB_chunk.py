import pandas as pd
from rank_bm25 import BM25Okapi
import pytrec_eval
from collections import defaultdict
import numpy as np

# === Step 1: Load your dataset ===
df = pd.read_csv("chunked_pages.csv")

# Optional: clean duplicated rows
df = df.drop_duplicates(subset=["query", "chunk_id", "text_description"])

# === Step 2: Prepare corpus ===
corpus = {
    row["chunk_id"]: {
        "text": str(row["text_description"]),
        "metadata": {
            "image_filename": row["image_filename"],
            "page_number": row["page_number"]
        }
    }
    for _, row in df.iterrows()
}

# === Step 3: Prepare queries ===
# For simplicity, assume only one query_id for now
query_text = df["query"].iloc[0]
query_id = "q1"
queries = {query_id: query_text}

# === Step 4: Prepare qrels ===
qrels = {
    query_id: {row["chunk_id"]: 1 for _, row in df.iterrows()}
}

# === Step 5: BM25 retrieval ===
doc_ids = list(corpus.keys())
docs = [corpus[doc_id]["text"] for doc_id in doc_ids]
tokenized_corpus = [doc.split() for doc in docs]

bm25 = BM25Okapi(tokenized_corpus)

# Compute BM25 scores for the query
tokenized_query = query_text.split()
scores = bm25.get_scores(tokenized_query)
doc_scores = {doc_ids[i]: float(scores[i]) for i in range(len(doc_ids))}

# BM25 run format
run = {query_id: doc_scores}

# === Step 6: Evaluate using pytrec_eval ===
k_values = [1, 3, 5, 10]
evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut', 'map_cut', 'recall', 'P'})
score_results = evaluator.evaluate(run)

# === Step 7: Metric Collection (your snippet) ===
all_ndcgs = defaultdict(list)
all_aps = defaultdict(list)
all_recalls = defaultdict(list)
all_precisions = defaultdict(list)

for qid in score_results.keys():
    for k in k_values:
        all_ndcgs[f"NDCG@{k}"].append(score_results[qid].get(f"ndcg_cut_{k}", 0.0))
        all_aps[f"MAP@{k}"].append(score_results[qid].get(f"map_cut_{k}", 0.0))
        all_recalls[f"Recall@{k}"].append(score_results[qid].get(f"recall_{k}", 0.0))
        all_precisions[f"P@{k}"].append(score_results[qid].get(f"P_{k}", 0.0))

# === Step 8: Print Results ===
print("=== BM25 Evaluation Results ===")
for k in k_values:
    print(f"NDCG@{k}: {np.mean(all_ndcgs[f'NDCG@{k}']):.4f}")
    print(f"MAP@{k}: {np.mean(all_aps[f'MAP@{k}']):.4f}")
    print(f"Recall@{k}: {np.mean(all_recalls[f'Recall@{k}']):.4f}")
    print(f"P@{k}: {np.mean(all_precisions[f'P@{k}']):.4f}")
    print("-" * 30)
