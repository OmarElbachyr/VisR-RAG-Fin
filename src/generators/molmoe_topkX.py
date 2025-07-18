import json
from collections import OrderedDict
import time
from PIL import Image, ImageStat
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image, ImageStat
import requests
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
import argparse

### Molmoe does not support multiimage input, so not topk2 or more

parser = argparse.ArgumentParser(description="VQA binary grading with Qwen")
parser.add_argument("--scores", required=True, help="Path to output log file")
parser.add_argument("--output", required=True, help="Path to JSON file with model answers")
parser.add_argument("--erroroutput", required=True, help="Path to JSON file with model answers")
args = parser.parse_args()

SCORES = args.scores
OUTPUT = args.output
ERROROUTPUT = args.erroroutput


def preprocess_image(path: str) -> Image.Image:
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Add white or dark background for transparency
    if image.mode == "RGBA" or 'A' in image.getbands():
        gray = image.convert('L')
        stat = ImageStat.Stat(gray)
        avg = stat.mean[0]
        bg = (0, 0, 0) if avg > 127 else (255, 255, 255)
        new_img = Image.new('RGB', image.size, bg)
        new_img.paste(image, mask=image.split()[3])
        image = new_img
    return image


running_time_start = time.time()
# Load sorted scores from JSON
with open(SCORES, 'r') as f:
    data = json.load(f, object_pairs_hook=OrderedDict)

# Parameters
base_path = "/home/laura/vqa-ir-qa/data/pages/"
file_ending = ".png"
top_k = 0 ##start at 0 for topk 1
query_input = {}

# Prompt for model
prompt= """ # Task description

The task is the following:
Given a PDF page as an image, you will have to generate questions that can be asked by a user to retrieve information from a large documentary corpus.

# Instructions 
- Do not repeat the question or its parts.
- Do not refer textually to the visual elements (example to avoid: "based on the diagram").
- Answering should be possible with the page or pages.
- The question is asked by a user without knowing the corpus existence or content.
- Make the answer short, so it does just consist of the retrieved information you find in the page for the answer. 
- Dont use punctuation at the end of your answer.
- don't create own answers, just use the information from the pages. Just extract it. 
- Don't answer in full sentences, just answer with the extracted information. Minimal answer.  

# Context: here are the question and the pages with the answer inside: """
start = time.time()

query_inputs = []  

for question, doc_scores in data.items():
    quest = doc_scores['query']
    docs = doc_scores['results'].keys()
    imagename = list(docs)
    imagename = imagename[top_k]
    full_paths = [f"{base_path}{imagename}{file_ending}"]

    query_input = {
        'doc_scores': full_paths,
        'query': quest
    }

    query_inputs.append(query_input)

output_data = []


processor = AutoProcessor.from_pretrained(
    'allenai/MolmoE-1B-0924', trust_remote_code=True, torch_dtype='auto', device_map='auto'
)
model = AutoModelForCausalLM.from_pretrained(
    'allenai/MolmoE-1B-0924', trust_remote_code=True, torch_dtype='auto', device_map='auto'
)



print(query_inputs)
# Query the model and collect results
for item in query_inputs:
    image_list = item['doc_scores']
    query = item['query']
    imagelist = preprocess_image(image_list[0])
    
    try:
        prompte = prompt + "\n" + query
        inputs = processor.process(images=[imagelist], text=prompte + "\n" + question)
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )
        gen = output[0, inputs['input_ids'].size(1):]
        answer = processor.tokenizer.decode(gen, skip_special_tokens=True)

        
        with open(OUTPUT, "a", encoding="utf-8") as out:
            json.dump({
                "question": query,
                "image": str(image_list[0]),
                "model_answer": answer,
            }, out, ensure_ascii=False)
            out.write("\n")
            
    except Exception as e:
        with open(ERROROUTPUT, "a", encoding="utf-8") as err:
            err.write(f"{image_list}\t{question}\t{e}\n")
        print(f"Error {image_list}: {e}")

print("Completed in", time.time() - start, "s")

