import json
import time
import os
from collections import OrderedDict

from PIL import Image, ImageStat
import requests
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig

# 🗂 Paths
INPUT_JSON = '/home/laura/vqa-ir-qa/data/label-studio-data-min.json'
BASE_PATH = "/home/laura/vqa-ir-qa/data/pages/"
OUTPUT_PATH = "/home/laura/vqa-ir-qa/data/output_molmoe_answers_annotated.jsonl"
ERROR_LOG = "/home/laura/vqa-ir-qa/data/failed_molmoe_cases_annotated.txt"

# Prompt (same as before)
PROMPT = """# Task description

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

def main():
    start = time.time()
    documents = json.load(open(INPUT_JSON, encoding="utf-8"), object_pairs_hook=OrderedDict)

    # Load model and processor
    processor = AutoProcessor.from_pretrained(
        'allenai/MolmoE-1B-0924', trust_remote_code=True, torch_dtype='auto', device_map='auto'
    )
    model = AutoModelForCausalLM.from_pretrained(
        'allenai/MolmoE-1B-0924', trust_remote_code=True, torch_dtype='auto', device_map='auto'
    )

    for doc in documents:
        img_file = os.path.basename(doc['image_filename'])
        img_path = os.path.join(BASE_PATH, img_file)
        if not os.path.isfile(img_path):
            continue

        img = preprocess_image(img_path)
        for qa in doc['qa_pairs']:
            question = qa['question']
            try:
                inputs = processor.process(images=[img], text=PROMPT + "\n" + question)
                inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

                output = model.generate_from_batch(
                    inputs,
                    GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                    tokenizer=processor.tokenizer
                )
                gen = output[0, inputs['input_ids'].size(1):]
                answer = processor.tokenizer.decode(gen, skip_special_tokens=True)

                with open(OUTPUT_PATH, "a", encoding="utf-8") as out:
                    json.dump({
                        "question": question,
                        "image": img_path,
                        "model_answer": answer,
                        "original_answer": qa.get("formatted_answer"),
                        "hint": qa.get("hint"),
                        "page_number": doc.get("page_number"),
                        "document": doc.get("original_filename")
                    }, out, ensure_ascii=False)
                    out.write("\n")

            except Exception as e:
                with open(ERROR_LOG, "a", encoding="utf-8") as err:
                    err.write(f"{img_file}\t{question}\t{e}\n")
                print(f"Error {img_file}: {e}")

    print("Completed in", time.time() - start, "s")

if __name__ == "__main__":
    main()
