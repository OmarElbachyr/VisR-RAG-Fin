import json
import os
import nltk
from rank_bm25 import BM25Okapi
from tqdm import tqdm

nltk.download("punkt_tab")

def normalize_answer(s):
    import re, string
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', string.punctuation))
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punctuation(lower(s))))

def word_lcs(gold, pred):
    A = normalize_answer(gold).split()
    B = normalize_answer(pred).split()
    dp = [[0] * (len(B)+1) for _ in range(len(A)+1)]
    for i in range(1, len(A)+1):
        for j in range(1, len(B)+1):
            if A[i-1] == B[j-1]:
                dp[i][j] = dp[i-1][j-1]+1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs_len = dp[-1][-1]
    precision = lcs_len / len(A) if A else 0.0
    return precision

def bm25_rank(query, corpus):
    tokenized_corpus = [nltk.word_tokenize(doc.lower()) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = nltk.word_tokenize(query.lower())
    scores = bm25.get_scores(tokenized_query)
    return scores

def evaluate_retrieval(chunks, qa_items, top_k=5):
    filenames = [chunk['filename'] for chunk in chunks]
    texts = [chunk['chunk'] for chunk in chunks]
    
    results = []
    for qa in tqdm(qa_items):
        gold_filename = os.path.basename(qa['image_filename'])  # just the file part
        questions = json.loads(qa['question'])  # list of questions

        for q in questions:
            scores = bm25_rank(q, texts)
            ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            ranked_filenames = [filenames[i] for i in ranked_indices]

            try:
                rank = ranked_filenames.index(gold_filename)
            except ValueError:
                rank = -1  # not found

            results.append({
                'question': q,
                'gold_filename': gold_filename,
                'rank': rank,
                'hit@1': int(rank == 0),
                'hit@5': int(rank >= 0 and rank < 5),
                'hit@10': int(rank >= 0 and rank < 10)
            })
    
    return results

def load_jsonl_or_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# Usage
if __name__ == "__main__":
    chunks = load_jsonl_or_json("chunks.json")
    qa_items = load_jsonl_or_json("./data/qa_pairs/qa_single_page_visual_bla_gpt-4o.json")

    results = evaluate_retrieval(chunks, qa_items, top_k=10)

    # Simple report
    total = len(results)
    hit1 = sum(r['hit@1'] for r in results)
    hit5 = sum(r['hit@5'] for r in results)
    hit10 = sum(r['hit@10'] for r in results)

    print(f"\nTotal questions: {total}")
    print(f"Hit@1: {hit1/total:.2%}")
    print(f"Hit@5: {hit5/total:.2%}")
    print(f"Hit@10: {hit10/total:.2%}")

    # Optional: save results to JSON
    with open("bm25_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
