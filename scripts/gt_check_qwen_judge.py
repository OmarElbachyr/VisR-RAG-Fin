# --- script to use qwen or GPT as LLM as a judge or Bertscore for evaluation --- 
import json
from collections import OrderedDict
import os
import pandas as pd
from evaluate import load
from bert_score import score
import tiktoken
from contextlib import redirect_stdout
import numpy as np
import time
from dotenv import load_dotenv
import openai
import subprocess
# ---------------------- CONFIG ------------------------------------------------
LOGFILE      = "/home/laura/vqa-ir-qa/data/VLM_nomic_7b_molmoe_1b_cases_annotated.txt"
load_dotenv("/home/laura/vqa-ir-qa/.env")
openai.api_key = os.getenv("OPENAI_API_KEY")
# -----------------------------------------------------------------------------

running_time_start = time.time()

#bertscore = load("bertscore")
#encoding  = tiktoken.encoding_for_model("gpt-4")

MODEL_NAME = "gpt-4"              # or "gpt-4o-preview"

SYSTEM_PROMPT = """
You are an expert VQA grader.
Given a question, reference answer, and proposed answer, decide if the proposed
answer is fully correct.
Respond ONLY with the single word "yes" (fully correct) or "no" (anything
less than fully correct). No other text.
""".strip()

def qwen_ollama_judge_binary(question: str, reference: str, prediction: str) -> bool:
    """
    Use Ollama CLI to query Qwen 2.2 model for binary VQA grading.
    Returns True if judge says 'yes', else False.
    """

    user_prompt = f"""Question: {question}

Reference answer: {reference}

Proposed answer: {prediction}

yes or no:"""

    full_prompt = SYSTEM_PROMPT + "\n\n" + user_prompt

    try:
        # Using `ollama run` to send the prompt to the model
        result = subprocess.run(
            ["ollama", "run", "qwen2.5:14b"],
            input=full_prompt,
            capture_output=True,
            text=True,
            check=True,
        )
        reply = result.stdout.strip().lower().replace(".", "").replace("\n", "")
        if reply not in {"yes", "no"}:
            raise ValueError(f"Unexpected Ollama output: {reply!r}")
        return reply == "yes"

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Ollama CLI failed: {e.stderr.strip()}") from e


#def gpt4o_judge_binary(question: str, reference: str, prediction: str) -> bool:
#    """
#    Return True if GPT‑4o says the prediction is fully correct, else False.
#    Compatible with OpenAI Python SDK v1.x.
#    """
#    user_prompt = f"""Question: {question}
#
#Reference answer: {reference}
#
#Proposed answer: {prediction}
#
#yes or no:"""
#
#    response = openai.chat.completions.create(
#        model=MODEL_NAME,
#        messages=[
#            {"role": "system", "content": SYSTEM_PROMPT},
#            {"role": "user",   "content": user_prompt}
#        ],
#        temperature=0.0,
#        max_tokens=1        # expects "yes" or "no"
#    )
#
#    # Normalise and validate
#    reply = response.choices[0].message.content.strip().lower().replace(".", "")
#    if reply not in {"yes", "no"}:
#        raise ValueError(f"Unexpected GPT‑4o output: {reply!r}")
#
#    return reply == "yes"
# -----------------------------------------------------------------------------


def average_token_length(data, answer_key):
    total_tokens = 0
    for item in data:
        tokens = encoding.encode(item[answer_key])
        total_tokens += len(tokens)
    return total_tokens / len(data) if data else 0


# ---------------------- LOAD DATA -------------------------------------------
with open('/home/laura/vqa-ir-qa/data/label-studio-data-min.json', 'r') as f:
    data1 = json.load(f, object_pairs_hook=OrderedDict)

extracted = []
for sample in data1:
    image_filename = os.path.basename(sample["image_filename"])
    for qa in sample["qa_pairs"]:
        question = qa.get("question", "")
        answer   = qa.get("formatted_answer") or qa.get("original_answer", [])
        answer_text = ", ".join(answer) if isinstance(answer, list) else answer
        extracted.append({
            "image_filename": image_filename,
            "question"      : question,
            "answer"        : answer_text
        })


with open('/home/laura/vqa-ir-qa/data/nomic_7b_molmoe_1b_cases_annotated.txt', 'r') as f:
    data2 = [json.loads(line, object_pairs_hook=OrderedDict) for line in f if line.strip()]


# Align references and predictions
reference_dict = {item['question']: item['answer'] for item in extracted}

predictions       = []
references        = []
queries_filtered  = []

for item in data2:
    query       = item['question']
    pred_answer = item['model_answer']
    ref_answer  = reference_dict.get(query, None)

    if ref_answer is None:
        print(f"No reference answer found for query: {query}")
        continue

    predictions.append(pred_answer)
    references.append(ref_answer)
    queries_filtered.append(query)


# ---------------------- EVALUATION & LOGGING --------------------------------
with open(LOGFILE, "w", encoding="utf-8") as f:

    # Log to both terminal and file
    def log(msg=""):
        print(msg)
        print(msg, file=f)

    # ---- GPT‑4o Binary Judge ----------------------------------------------
    log("### GPT‑4o BINARY JUDGE ###\n")
    judge_results = []

    for q, pred, ref in zip(queries_filtered, predictions, references):
        try:
            #correct = gpt4o_judge_binary(q, ref, pred)
            correct = qwen_ollama_judge_binary(q, ref, pred)
        except Exception as e:
            log(f"⚠️  qwen failed on '{q[:60]}...': {e}")
            correct = False
        judge_results.append(correct)

        log(f"Query     : {q}")
        log(f"Reference : {ref}")
        log(f"Prediction: {pred}")
        log(f"qwen verdict: {'yes' if correct else 'no'}")
        log("----------------------------------------------------")

    if judge_results:
        accuracy = np.mean(judge_results) * 100
        log(f"\nOverall qwen accuracy: {accuracy:.2f}% "
            f"({sum(judge_results)}/{len(judge_results)})")

    # ---- Token‑length stats -----------------------------------------------
    avg_len_data1 = average_token_length(extracted, 'answer')
    avg_len_data2 = average_token_length(data2, 'model_answer')
    log(f"\nAverage token length (ground‑truth answers): {avg_len_data1:.2f}")
    log(f"Average token length (generated answers)   : {avg_len_data2:.2f}")

# ---------------------- WALL‑CLOCK TIME -------------------------------------
running_time = time.time() - running_time_start
print(f"Finished. Time: {running_time:.1f}s")
