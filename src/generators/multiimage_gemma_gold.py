#####in progress

import ollama
import json
from collections import OrderedDict

with open('/home/laura/vqa-ir-qa/data/sorted_scores_colipali.json', 'r') as f:
    data = json.load(f, object_pairs_hook=OrderedDict)

base_path = "vqa-ir-qa/data/pages/"
file_ending = ".png"
query_input = {}

for question, doc_scores in data.items():
    # Get the first 3 document IDs from the OrderedDict keys
    top_docs = list(doc_scores.keys())[:3]
    
    # Build the full paths for those docs
    full_paths = [f"{base_path}{doc_id}{file_ending}" for doc_id in top_docs]
    
    query_input[question] = full_paths


for query, image_list in query_input.items():
    response = ollama.chat(model='gemma3:4b', 
        messages=[{
            'role': 'user', 
            'content': query,
            'images': image_list,
        }],
        # options={"temperature":0.7}
        )

    print(query + f"\n" + str(image_list) + f"\n\n", response['message']['content'] + "\n______________________________________________")
