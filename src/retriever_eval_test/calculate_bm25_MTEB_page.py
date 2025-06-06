import pandas as pd
from rank_bm25 import BM25Okapi
import pytrec_eval
from collections import defaultdict
import numpy as np

# === Step 1: Load dataset ===
df = pd.read_csv("chunked_pages.csv")
df = df.drop_duplicates(subset=["query", "chunk_id", "text_description"])

# === Step 2: Build corpus ===
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

# === Step 3: Prepare queries (only one for now) ===
query_text = df["query"].iloc[0]
query_id = "q1"
queries = {query_id: query_text}

# === Step 4: Prepare chunk-to-page map ===
chunk_id_to_page = {
    row["chunk_id"]: row["page_number"]
    for _, row in df.iterrows()
}

# === Step 5: Prepare qrels at page-level ===
# Assumes all pages in this query are relevant
relevant_pages = set(df["page_number"].unique())
qrels = {
    query_id: {str(p): 1 for p in relevant_pages}  # str keys for pytrec_eval
}

# === Step 6: BM25 scoring (chunk-level) ===
doc_ids = list(corpus.keys())
docs = [corpus[doc_id]["text"] for doc_id in doc_ids]
tokenized_corpus = [doc.split() for doc in docs]

bm25 = BM25Okapi(tokenized_corpus)
tokenized_query = query_text.split()
chunk_scores = bm25.get_scores(tokenized_query)

# === Step 7: Aggregate chunk scores to page-level ===
page_scores = defaultdict(list)
for i, chunk_id in enumerate(doc_ids):
    page = chunk_id_to_page.get(chunk_id)
    if page is not None:
        page_scores[page].append(chunk_scores[i])

# Use max/mean/sum aggregation (here: max)
aggregated_page_scores = {
    str(page): float(np.max(scores)) for page, scores in page_scores.items()
}

# === Step 8: Format run for pytrec_eval ===
run = {query_id: aggregated_page_scores}

# === Step 9: Evaluate with pytrec_eval ===
k_values = [1, 3, 5, 10]
evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'ndcg_cut', 'map_cut', 'recall', 'P'})
score_results = evaluator.evaluate(run)

# === Step 10: Collect metrics ===
all_ndcgs = defaultdict(list)
all_aps = defaultdict(list)
all_recalls = defaultdict(list)
all_precisions = defaultdict(list)

for qid in score_results:
    for k in k_values:
        all_ndcgs[f"NDCG@{k}"].append(score_results[qid].get(f"ndcg_cut_{k}", 0.0))
        all_aps[f"MAP@{k}"].append(score_results[qid].get(f"map_cut_{k}", 0.0))
        all_recalls[f"Recall@{k}"].append(score_results[qid].get(f"recall_{k}", 0.0))
        all_precisions[f"P@{k}"].append(score_results[qid].get(f"P_{k}", 0.0))

# === Step 11: Print metrics ===
print("=== BM25 Evaluation (Page-Level) ===")
for k in k_values:
    print(f"NDCG@{k}: {np.mean(all_ndcgs[f'NDCG@{k}']):.4f}")
    print(f"MAP@{k}: {np.mean(all_aps[f'MAP@{k}']):.4f}")
    print(f"Recall@{k}: {np.mean(all_recalls[f'Recall@{k}']):.4f}")
    print(f"P@{k}: {np.mean(all_precisions[f'P@{k}']):.4f}")
    print("-" * 30)
