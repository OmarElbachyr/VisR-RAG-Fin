import ollama
import json
from collections import OrderedDict
import time

running_time_start = time.time()
# Load sorted scores from JSON
with open('/home/laura/vqa-ir-qa/data/sorted_scores_colipali.json', 'r') as f:
    data = json.load(f, object_pairs_hook=OrderedDict)

# Parameters
base_path = "/home/laura/vqa-ir-qa/data/pages/"
file_ending = ".png"
top_k = 3
query_input = {}

# Prompt for model
prompt= """ # Task description

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

# Prepare the image list per query
for question, doc_scores in data.items():
    top_docs = list(doc_scores.keys())[:top_k]
    full_paths = [f"{base_path}{doc_id}{file_ending}" for doc_id in top_docs]
    query_input[question] = full_paths

# Output list to store results
output_data = []

# Query the model and collect results
for query, image_list in query_input.items():
    response = ollama.chat(
        model='granite3.2-vision',
        messages=[{
            'role': 'user',
            'content': prompt + "\n" + query,
            'images': image_list,
        }],
        # options={"temperature": 0.7}
    )

    answer = response['message']['content']

    # Print result
    # print(query + "\n" + str(image_list) + "\n\n" + answer + "\n" + "_" * 46)

    # Save result
    output_data.append({
        "query": query,
        "image_list": image_list,
        "answer": answer
    })

# Save all results to a JSON file
with open("/home/laura/vqa-ir-qa/data/granite_topk3.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)

running_time = time.time() - running_time_start
print(running_time)
