import ollama
import json
from collections import OrderedDict
import time
import os

running_time_start = time.time()

# Load sorted scores from JSON
with open('/home/laura/vqa-ir-qa/data/label-studio-data-min.json', 'r') as f:
    documents = json.load(f, object_pairs_hook=OrderedDict)

# Parameters
base_path = "/home/laura/vqa-ir-qa/data/pages/"
file_ending = ".png"
output_path = "/home/laura/vqa-ir-qa/data/output_qwen3b_answers_annotated.jsonl"
error_log_path = "/home/laura/vqa-ir-qa/data/failed_cases_qwen3b_annotated.txt"

# Prompt
prompt = """# Task description

The task is the following:
Given a PDF page as an image, you will have to generate questions that can be asked by a user to retrieve information from a large documentary corpus.

# Instructions 
- Do not repeat the question or it's parts.
- Do not refer textually to the visual elements (example to avoid: "based on the diagram").
- Answering should be possible with the page or pages.
- The question is asked by a user without knowing the corpus existence or content.
- Make the answer short, so it does just consist of the retrieved information you find in the page for the answer. 
- Dont use punctuation at the end of your answer.
- don't create own answers, just use the information from the pages. Just extract it. 
- Don't answer in full sentences, just answer with the extracted information. Minimal answer.  

# Context: here are the question and the pages with the answer inside:"""

# Iterate through each document and its QA pairs
for doc in documents:
    image_filename = os.path.basename(doc['image_filename'])
    image_path = base_path + image_filename
    qa_pairs = doc['qa_pairs']

    assert os.path.isfile(image_path), f"Missing image file: {image_path}"

    for qa in qa_pairs:
        question = qa['question']

        try:
            response = ollama.chat(
                model='qwen2.5vl:3b',
                messages=[{
                    'role': 'user',
                    'content': prompt + "\n" + question,
                    'images': [image_path],
                }]
            )

            answer = response['message']['content']

            # Write each result immediately to the .jsonl file
            with open(output_path, "a", encoding="utf-8") as f:
                json.dump({
                    "question": question,
                    "image": image_path,
                    "model_answer": answer,
                    "original_answer": qa.get("formatted_answer", None),
                    "hint": qa.get("hint", None),
                    "page_number": doc.get("page_number"),
                    "document": doc.get("original_filename")
                }, f, ensure_ascii=False)
                f.write('\n')

        except Exception as e:
            # Log the failure to a text file
            with open(error_log_path, "a", encoding="utf-8") as error_file:
                error_file.write(f"{image_filename}\t{question}\n")
            print(f"Error processing {image_filename} with question: {question}")
            print(f"Exception: {e}")
            continue  # Skip to the next question

running_time = time.time() - running_time_start
print("Total time:", running_time)
