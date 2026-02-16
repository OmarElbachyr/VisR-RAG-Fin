#!/usr/bin/env python3
import json
import time
import argparse
import os
import io
from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv
from generators.prompt_utils import load_prompt

load_dotenv()


class GeminiTopKHybridGenerator:
    def __init__(
        self,
        text_data_file,
        image_data_file,
        annotations_file,
        models,
        topk_text=1,
        topk_image=1,
        filter_top_k_hits=False,
    ):
        self.text_data_file = text_data_file
        self.image_data_file = image_data_file
        self.annotations_file = annotations_file
        self.topk_text = topk_text
        self.topk_image = topk_image
        self.models = models
        self.filter_top_k_hits = filter_top_k_hits

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        self.client = genai.Client(api_key=api_key)

    # ------------------------------------------------------------------
    # LOADERS
    # ------------------------------------------------------------------

    def load_data(self):
        with open(self.text_data_file, "r", encoding="utf-8") as f:
            text_data = json.load(f)

        with open(self.image_data_file, "r", encoding="utf-8") as f:
            image_data = json.load(f)

        with open(self.annotations_file, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        print(f"Loaded {len(text_data)} text retrieval entries")
        print(f"Loaded {len(image_data)} image retrieval entries")
        print(f"Loaded {len(annotations)} annotations")

        return text_data, image_data, annotations

    def get_image_path(self, page_id):
        page_path = f"data/pages/{page_id}.png"
        noise_path = f"data/hard_negative_pages_text/{page_id}.png"
        return page_path if os.path.exists(page_path) else noise_path

    def get_text_path(self, page_id):
        return f"src/generators/rq_5/topk_pages_txt/k_10/{page_id}.txt"

    # ------------------------------------------------------------------
    # GEMINI QUERY (HYBRID)
    # ------------------------------------------------------------------

    def query_model(self, model, question, text_page_ids, image_page_ids):
        start_time = time.time()

        try:
            # -------- images --------
            image_parts = []
            for pid in image_page_ids:
                img_path = self.get_image_path(pid)
                image = Image.open(img_path)
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG")
                buffer.seek(0)
                image_parts.append(
                    types.Part.from_bytes(
                        data=buffer.read(),
                        mime_type="image/jpeg",
                    )
                )

            # -------- text --------
            page_texts = []
            for pid in text_page_ids:
                txt_path = self.get_text_path(pid)
                if os.path.exists(txt_path):
                    page_texts.append(
                        open(txt_path, "r", encoding="utf-8").read()
                    )

            prompt_template = load_prompt("topk_prompt.txt")
            prompt = prompt_template.format(
                question=question,
                image_count=len(image_parts),
            )

            text_context = "\n\n".join(
                f"[Page {i+1} – Text]\n{text}"
                for i, text in enumerate(page_texts)
            )

            contents = [prompt + "\n\n" + text_context] + image_parts

            response = self.client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                ),
            )

            return {
                "success": True,
                "response": response.text.strip() if response.text else "",
                "time": time.time() - start_time,
            }

        except Exception as e:
            return {
                "success": False,
                "response": None,
                "time": time.time() - start_time,
                "error": str(e),
            }

    # ------------------------------------------------------------------
    # GENERATION LOOP
    # ------------------------------------------------------------------

    def generate_answers(self, models, limit=None, output_dir=None):
        text_data, image_data, annotations = self.load_data()

        question_keys = sorted(
            [k for k in text_data.keys() if k.startswith("q")],
            key=lambda x: int(x[1:]),
        )

        if limit:
            question_keys = question_keys[:limit]

        # ------------------------
        # Build QA lookup
        # ------------------------
        qa_lookup = {}
        for ann in annotations:
            real_page = None
            if ann.get("image_filename"):
                real_page = ann["image_filename"].split("/")[-1].split(".")[0]

            for qa in ann.get("qa_pairs", []):
                qa_lookup[qa["question_id"]] = {
                    "answer": qa.get("answer"),
                    "type": qa.get("type"),
                    "evidence": qa.get("evidence"),
                    "company": ann.get("company"),
                    "file_category": ann.get("file_category"),
                    "language": ann.get("language"),
                    "real_relevant_page": real_page,
                }

        # ------------------------
        # FILTER TOP-K HITS (HYBRID)
        # ------------------------
        filtered_keys = []

        for q_key in question_keys:
            if q_key not in qa_lookup:
                continue

            text_entry = text_data[q_key]
            image_entry = image_data[q_key]

            text_page_ids = list(text_entry["results"].keys())[: self.topk_text]
            image_page_ids = list(image_entry["results"].keys())[: self.topk_image]

            real_page = qa_lookup[q_key]["real_relevant_page"]

            if not self.filter_top_k_hits:
                filtered_keys.append(q_key)
            else:
                if real_page in text_page_ids or real_page in image_page_ids:
                    filtered_keys.append(q_key)

        question_keys = filtered_keys

        print(f"Processing {len(question_keys)} queries")

        # ------------------------
        # GENERATION
        # ------------------------
        results_by_model = {m: [] for m in models}

        for model in models:
            print(f"Processing model: {model}")

            for i, q_key in enumerate(question_keys, 1):
                print(f"Entry {i}/{len(question_keys)} ({q_key})")

                text_entry = text_data[q_key]
                image_entry = image_data[q_key]

                text_page_ids = list(text_entry["results"].keys())[: self.topk_text]
                image_page_ids = list(image_entry["results"].keys())[: self.topk_image]

                qa = qa_lookup[q_key]
                question = text_entry["query"]

                result = self.query_model(
                    model,
                    question,
                    text_page_ids,
                    image_page_ids,
                )

                relevant_pages = (
                    [(pid, "text") for pid in text_page_ids] +
                    [(pid, "image") for pid in image_page_ids]
                )

                results_by_model[model].append({
                    "model": model,
                    "q_id": q_key,
                    "question": question,
                    "relevant_pages": relevant_pages,
                    "real_relevant_page": qa["real_relevant_page"],
                    "ground_truth": qa["answer"],
                    "evidence": qa["evidence"],
                    "type": qa["type"],
                    "predicted": result["response"],
                    "success": result["success"],
                    "time": result["time"],
                    "error": result.get("error"),
                    "company": qa["company"],
                    "file_category": qa["file_category"],
                    "language": qa["language"],
                })

                time.sleep(0.1)

            model_safe = model.replace(":", "-").replace("/", "-")
            out_file = os.path.join(
                output_dir,
                f"gemini_topk_hybrid_text{self.topk_text}_image{self.topk_image}_{model_safe}.json",
            )

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results_by_model[model], f, indent=2, ensure_ascii=False)

            print(f"Saved: {out_file}")

        for model, model_results in results_by_model.items():
            model_successful = sum(1 for r in model_results if r['success'])
            print(f"  {model}: {len(model_results)} total, {model_successful} successful")

        return list(results_by_model.values())[0]


# ------------------------------------------------------------------
# CLI (UNCHANGED)
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--text_retrieval_json")
    parser.add_argument("--image_retrieval_json")
    parser.add_argument("--annotations_file")
    parser.add_argument("--topk_text", type=int, default=5)
    parser.add_argument("--topk_image", type=int, default=5)
    parser.add_argument("--output_dir")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--filter_top_k_hits", action="store_true")

    args = parser.parse_args()

    if not args.models:
        # Add models you want to test here, or pass them via command line
        args.models = ["gemini-3-flash-preview"]

    if not args.limit:
        args.limit = None

    args.topk_text = 5
    args.topk_image = 5

    args.filter_top_k_hits = True

    if not args.annotations_file:
        args.annotations_file = (
            "data/annotations/final_annotations/filtered_annotations/"
            "by_category/first_pass_classified_qa_category_A.json"
        )

    if not args.text_retrieval_json:
        args.text_retrieval_json = (
            "src/retrievers/results/chunked_pages_second_pass_rq1/"
            "retrieved_pages/colbert-ir_colbertv2.0_sorted_run.json"
        )

    if not args.image_retrieval_json:
        args.image_retrieval_json = (
            "src/retrievers/results/chunked_pages_second_pass_rq1/0.5/"
            "retrieved_pages/nomic-ai_colnomic-embed-multimodal-7b_sorted_run.json"
        )

    if not args.output_dir:
        args.output_dir = "src/generators/results/category_a/retrieval_pipeline_hybrid"

    os.makedirs(args.output_dir, exist_ok=True)

    generator = GeminiTopKHybridGenerator(
        text_data_file=args.text_retrieval_json,
        image_data_file=args.image_retrieval_json,
        annotations_file=args.annotations_file,
        models=args.models,
        topk_text=args.topk_text,
        topk_image=args.topk_image,
        filter_top_k_hits=args.filter_top_k_hits,
    )

    generator.generate_answers(
        models=args.models,
        limit=args.limit,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
