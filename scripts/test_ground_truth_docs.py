import json
from collections import OrderedDict
import os
import pandas as pd

#--- check if answer of generator is linked to retrieved documents ---

with open('/home/laura/vqa-ir-qa/data/ls_single_page_visual_onpoint_gpt-4o-2024-08-06.json', 'r') as f:
    data = json.load(f, object_pairs_hook=OrderedDict)

extracted = []

for sample in data:
    image_filename = os.path.basename(sample["image_filename"])
    
    for qa in sample["qa_pairs"]:
        question = qa.get("question", "")
        # Use formatted_answer if present, otherwise fallback to original_answer
        answer = qa.get("formatted_answer") or qa.get("original_answer", [])
        answer_text = ", ".join(answer) if isinstance(answer, list) else answer
        
        extracted.append({
            "image_filename": image_filename,
            "question": question,
            "answer": answer_text
        })

print(extracted)



samples = extracted       
scores_path  = "/home/laura/vqa-ir-qa/data/colnomic_gemma4b_output_topk1.json"
k           = 1
with open(scores_path, "r", encoding="utf‑8") as f:
    score_dict = json.load(f)

rows = []
question_topk_matches = {}

for sample in samples:
    file_id = os.path.splitext(os.path.basename(sample["image_filename"]))[0]
    q = sample.get("question")
    ans = sample.get("answer")
    if isinstance(ans, list):
        ans = ", ".join(ans)

    present = q in score_dict
    in_topk = False
    if present:
        topk_ids = sorted(score_dict[q].items(), key=lambda x: x[1], reverse=True)[:k]
        topk_ids = [fid for fid, _ in topk_ids]
        in_topk = file_id in topk_ids

    rows.append({
        "question": q,
        "file_id": file_id,
        "in_score_json": present,
        f"in_top_{k}": in_topk,
        "answer": ans,
    })

    if q not in question_topk_matches:
        question_topk_matches[q] = {"total": 0, "in_topk": 0}
    question_topk_matches[q]["total"] += 1
    if in_topk:
        question_topk_matches[q]["in_topk"] += 1

df = pd.DataFrame(rows)

missing_q = df[~df["in_score_json"]]
not_topk_q = df[df["in_score_json"] & ~df[f"in_top_{k}"]]

unique_questions = set(q for q in question_topk_matches.keys())
questions_found_in_score = [q for q in unique_questions if q in score_dict]
questions_with_topk = [q for q, stats in question_topk_matches.items() if stats["in_topk"] > 0]

print(f"Total QA pairs analysed: {len(df)}")
print(f"Unique questions found in score JSON: {len(questions_found_in_score)}")
print(f"samples NOT in top-{k}: {len(not_topk_q)}")
print(f"Number of questions with at least one sample in top-{k}: {len(questions_with_topk)}")

question_summary = [
    {"question": q, "total_samples": stats["total"], "samples_in_topk": stats["in_topk"]}
    for q, stats in question_topk_matches.items()
]

pd.DataFrame(question_summary).to_csv("question_topk_summary.csv", index=False)
df.to_csv("detailed_samples.csv", index=False)