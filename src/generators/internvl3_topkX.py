import ollama
import json
from collections import OrderedDict
import time
from PIL import Image, ImageStat
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


def preprocess_image(path):
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Handle alpha channel
    if img.mode == "RGBA" or "A" in img.getbands():
        gray = img.convert("L")
        avg = ImageStat.Stat(gray).mean[0]
        bg = (0, 0, 0) if avg > 127 else (255, 255, 255)
        new = Image.new("RGB", img.size, bg)
        new.paste(img, mask=img.split()[3])
        img = new

    # Resize if larger than 1000x1200
    max_width, max_height = 1000, 1200
    if img.width > max_width or img.height > max_height:
        img.thumbnail((max_width, max_height), Image.LANCZOS)

    return img


running_time_start = time.time()
# Load sorted scores from JSON
with open('/home/laura/vqa-ir-qa/data/retrieved_pages/vidore_colSmol-500M_sorted_run.json', 'r') as f:
    data = json.load(f, object_pairs_hook=OrderedDict)

# Parameters
base_path = "/home/laura/vqa-ir-qa/data/pages/"
file_ending = ".png"
top_k = 0
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
# Prepare the image list per query
query_inputs = []  # this will hold all your entries

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
# Output list to store results
output_data = []

repo = "OpenGVLab/InternVL3-1B-hf"
processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    repo,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16
).eval()

print(query_inputs)
# Query the model and collect results
for item in query_inputs:
    image_list = item['doc_scores']
    query = item['query']
    imagelist = [preprocess_image(image) for image in image_list]
    
    try:
        prompte = prompt + "\n" + query
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": imagelist[0]},
                    {"type": "text", "text": prompte}
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=model.dtype)
        generated_ids = model.generate(**inputs, max_new_tokens=200)
        answer = processor.decode(
            generated_ids[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        with open("/home/laura/vqa-ir-qa/data/colsmol500_internvl3_2b_answers_annotated.json", "a", encoding="utf-8") as out:
            json.dump({
                "question": query,
                "image": str(image_list[0]),
                "model_answer": answer,
            }, out, ensure_ascii=False)
            out.write("\n")
            
    except Exception as e:
        with open("/home/laura/vqa-ir-qa/data/failed_colsmol500_internvl3_2b_cases_annotated.txt", "a", encoding="utf-8") as err:
            err.write(f"{image_list}\t{question}\t{e}\n")
        print(f"Error {image_list}: {e}")

print("Completed in", time.time() - start, "s")

