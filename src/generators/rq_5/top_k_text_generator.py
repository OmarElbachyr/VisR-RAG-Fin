#!/usr/bin/env python3
import json
import time
import argparse
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from generators.prompt_utils import load_prompt

load_dotenv()


class GeminiTopKTextGenerator:
    def __init__(self, data_file, annotations_file, models, top_k=1, filter_top_k_hits=False):
        self.data_file = data_file
        self.annotations_file = annotations_file
        self.top_k = top_k
        self.models = models
        self.filter_top_k_hits = filter_top_k_hits

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        self.client = genai.Client(api_key=api_key)
        self.model_clients = {model: model for model in models}

    def load_data(self):
        with open(self.data_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        with open(self.annotations_file, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        print(f"Loaded {len(data)} retrieval entries")
        print(f"Loaded {len(annotations)} annotations")
        return data, annotations

    def get_text_path(self, page_id):
        return f"src/generators/rq_5/topk_pages_txt/k_{self.top_k}/{page_id}.txt"

    def query_model(self, model, question, page_texts):
        start_time = time.time()

        try:
            page_count = len(page_texts)

            prompt_template = load_prompt("topk_prompt_text.txt")
            prompt = prompt_template.format(
                question=question,
                image_count=page_count  # keep variable name for parity
            )

            full_input = (
                prompt
                + "\n\n"
                + "\n\n".join(
                    f"[Page {i+1}]\n{text}"
                    for i, text in enumerate(page_texts)
                )
            )

            response = self.client.models.generate_content(
                model=model,
                contents=[full_input],
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

    def generate_answers(self, models, limit=None, output_dir=None):
        data, annotations = self.load_data()

        question_keys = sorted(
            [k for k in data.keys() if k.startswith("q")],
            key=lambda x: int(x[1:]),
        )

        if limit:
            question_keys = question_keys[:limit]

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

        question_keys = [k for k in question_keys if k in qa_lookup]

        filtered_keys = []
        for q_key in question_keys:
            results_dict = data[q_key]["results"]
            top_k_pages = list(results_dict.keys())[:self.top_k]
            if (
                not self.filter_top_k_hits
                or qa_lookup[q_key]["real_relevant_page"] in top_k_pages
            ):
                filtered_keys.append(q_key)

        print(f"Processing {len(filtered_keys)} queries after filtering")

        results_by_model = {m: [] for m in models}

        for model in models:
            print(f"Processing model: {model}")

            for i, q_key in enumerate(filtered_keys, 1):
                print(f"Entry {i}/{len(filtered_keys)} ({q_key})")

                entry = data[q_key]
                page_ids = list(entry["results"].keys())[:self.top_k]

                page_texts = []
                for pid in page_ids:
                    txt_path = self.get_text_path(pid)
                    if not os.path.exists(txt_path):
                        break
                    page_texts.append(
                        open(txt_path, "r", encoding="utf-8").read()
                    )

                if len(page_texts) != self.top_k:
                    continue

                qa = qa_lookup[q_key]
                question = entry["query"]

                result = self.query_model(model, question, page_texts)

                results_by_model[model].append({
                    "model": model,
                    "q_id": q_key,
                    "question": question,
                    "relevant_pages": page_ids,
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
            out_file = f"{output_dir}/gemini_topk_text_{model_safe}.json"

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(results_by_model[model], f, indent=2, ensure_ascii=False)

            print(f"Saved: {out_file}")

        all_results = [r for model_results in results_by_model.values() for r in model_results]
        successful = sum(1 for r in all_results if r['success'])
        print(f"\nCompleted: {len(all_results)} total, {successful} successful")
        
        for model, model_results in results_by_model.items():
            model_successful = sum(1 for r in model_results if r['success'])
            print(f"  {model}: {len(model_results)} total, {model_successful} successful")

        return list(results_by_model.values())[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--retrievers", nargs="+")
    parser.add_argument("--annotations_file")
    parser.add_argument("--top_k", nargs="+", type=int)
    parser.add_argument("--output_dir")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--filter_top_k_hits", action="store_true")

    args = parser.parse_args()

    if not args.models:
        # Add models you want to test here, or pass them via command line
        args.models = ['gemini-3-flash-preview']

    if not args.retrievers:
        # Add retrievers you want to test here, or pass them via command line
        args.retrievers = ['colbert-ir_colbertv2.0']

    if not args.limit:
        args.limit = None

    if not args.top_k:
        args.top_k = [10]

    args.filter_top_k_hits = True

    args.annotations_file = 'data/annotations/final_annotations/filtered_annotations/by_category/first_pass_classified_qa_category_A.json'
    args.output_dir = "src/generators/results/category_a/retrieval_pipeline_text"

    generator = GeminiTopKTextGenerator(
        None,
        args.annotations_file,
        models=args.models,
        top_k=1,
        filter_top_k_hits=args.filter_top_k_hits,
    )

    base_folder = "src/retrievers/results/chunked_pages_second_pass_rq1/retrieved_pages"

    for top_k in args.top_k:
        generator.top_k = top_k
        for retriever in args.retrievers:
            generator.data_file = (
                f"{base_folder}/{retriever.replace('/', '_')}_sorted_run.json"
            )
            out_dir = os.path.join(
                args.output_dir, f"top_k_{top_k}", retriever.replace("/", "_")
            )
            os.makedirs(out_dir, exist_ok=True)
            generator.generate_answers(args.models, args.limit, out_dir)


if __name__ == "__main__":
    main()
