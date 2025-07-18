import json
import time
import os
from collections import OrderedDict

from PIL import Image, ImageStat
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# 🗂 Paths
INPUT_JSON = "/home/laura/vqa-ir-qa/data/label-studio-data-min.json"
BASE_PATH = "/home/laura/vqa-ir-qa/data/pages"
OUTPUT_PATH = "/home/laura/vqa-ir-qa/data/output_internvl3_1b_answers_annotated.jsonl"
ERROR_LOG = "/home/laura/vqa-ir-qa/data/failed_internvl3_1b_cases_annotated.txt"

# Prompt
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

def main():
    start = time.time()
    docs = json.load(open(INPUT_JSON, encoding="utf-8"), object_pairs_hook=OrderedDict)

    repo = "OpenGVLab/InternVL3-1B-hf"
    processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        repo,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16
    ).eval()

    for doc in docs:
        img_file = os.path.basename(doc["image_filename"])
        img_path = os.path.join(BASE_PATH, img_file)
        if not os.path.exists(img_path):
            continue

        img = preprocess_image(img_path)
        for qa in doc["qa_pairs"]:
            question = qa["question"]
            try:
                prompt = PROMPT + "\n" + question
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": prompt}
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