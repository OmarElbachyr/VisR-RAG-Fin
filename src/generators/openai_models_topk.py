#!/usr/bin/env python3
import json
import time
import argparse
import os
import base64
from openai import OpenAI
from dotenv import load_dotenv
from generators.prompt_utils import load_prompt

load_dotenv()


class OpenAITopKGenerator:
    def __init__(
        self,
        data_file,
        annotations_file,
        models,
        top_k=1,
        filter_top_k_hits=False,
    ):
        self.data_file = data_file
        self.annotations_file = annotations_file
        self.models = models
        self.top_k = top_k
        self.filter_top_k_hits = filter_top_k_hits

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)

    def load_data(self):
        with open(self.data_file, "r", encoding="utf-8") as f:
            retrieval_data = json.load(f)
        print(f"Loaded {len(retrieval_data)} retrieval entries")

        with open(self.annotations_file, "r", encoding="utf-8") as f:
            annotations = json.load(f)
        print(f"Loaded {len(annotations)} annotations")

        return retrieval_data, annotations

    def get_image_path(self, filename):
        fname = filename.split("/")[-1]
        page_path = f"data/pages/{fname}.png"
        noise_path = f"data/hard_negative_pages_text/{fname}.png"

        if os.path.exists(page_path):
            return page_path
        if os.path.exists(noise_path):
            return noise_path
        return None

    def encode_image(self, image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def query_model(self, model, question, image_paths):
        start_time = time.time()
        try:
            image_contents = []
            for img_path in image_paths:
                base64_img = self.encode_image(img_path)
                image_contents.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_img}"
                    }
                })

            prompt_template = load_prompt("topk_prompt.txt")
            prompt = prompt_template.format(
                question=question,
                image_count=len(image_paths)
            )

            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            [{"type": "text", "text": prompt}]
                            + image_contents
                        )
                    }
                ],
            )

            text = response.choices[0].message.content or ""

            return {
                "success": True,
                "response": text.strip(),
                "time": time.time() - start_time,
            }

        except Exception as e:
            return {
                "success": False,
                "response": None,
                "time": time.time() - start_time,
                "error": str(e),
            }

    def generate_answers(self, limit=None, output_dir="results"):
        retrieval_data, annotations = self.load_data()

        # Build QA lookup
        qa_lookup = {}
        for ann in annotations:
            image_filename = ann.get("image_filename")
            real_page = (
                image_filename.split("/")[-1].split(".")[0]
                if image_filename else None
            )

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

        # Sort and optionally limit questions
        q_keys = sorted(
            [k for k in retrieval_data.keys() if k in qa_lookup],
            key=lambda x: int(x[1:])
        )
        if limit:
            q_keys = q_keys[:limit]

        # Filter by Top-K hit (optional)
        filtered_keys = []
        for qk in q_keys:
            topk_pages = list(retrieval_data[qk]["results"].keys())[:self.top_k]
            if not self.filter_top_k_hits:
                filtered_keys.append(qk)
            else:
                if qa_lookup[qk]["real_relevant_page"] in topk_pages:
                    filtered_keys.append(qk)

        print(f"Processing {len(filtered_keys)} queries")

        os.makedirs(output_dir, exist_ok=True)
        results_by_model = {m: [] for m in self.models}

        for model in self.models:
            print(f"\nRunning model: {model}")
            for idx, qk in enumerate(filtered_keys, 1):
                entry = retrieval_data[qk]
                question = entry["query"]

                topk_pages = list(entry["results"].keys())[:self.top_k]
                image_paths = [self.get_image_path(p) for p in topk_pages]

                if any(p is None for p in image_paths):
                    print("Missing image → skipped:", qk)
                    continue

                print(f"[{idx}/{len(filtered_keys)}] {qk}")

                result = self.query_model(
                    model=model,
                    question=question,
                    image_paths=image_paths,
                )

                qa = qa_lookup[qk]
                results_by_model[model].append({
                    "model": model,
                    "q_id": qk,
                    "question": question,
                    "relevant_pages": topk_pages,
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

            model_safe = model.replace("/", "-").replace(":", "-").replace(".", "-")
            out_path = os.path.join(output_dir, f"openai_topk_{model_safe}.json")

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results_by_model[model], f, indent=2, ensure_ascii=False)

            print(f"Saved → {out_path}")

        for model, model_results in results_by_model.items():
            model_successful = sum(1 for r in model_results if r['success'])
            print(f"  {model}: {len(model_results)} total, {model_successful} successful")
        
        return results_by_model

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
        args.models = ["gpt-5.2-2025-12-11"]

    if not args.retrievers:
        # Add retrievers you want to test here, or pass them via command line
        args.retrievers = ["nomic-ai/colnomic-embed-multimodal-7b"]

    if not args.limit:
        args.limit = None

    if not args.top_k:
        args.top_k = [10, 5, 3, 1]

    args.filter_top_k_hits = True

    args.annotations_file = 'data/annotations/final_annotations/filtered_annotations/by_category/first_pass_classified_qa_category_A.json'
    args.output_dir = "src/generators/results/category_a/retrieval_pipeline"

    base_folder = (
        "src/retrievers/results/chunked_pages_second_pass_rq1/"
        "0.5/retrieved_pages"
    )

    generator = OpenAITopKGenerator(
        data_file=None,
        annotations_file=args.annotations_file,
        models=args.models,
        top_k=1,
        filter_top_k_hits=args.filter_top_k_hits,
    )

    for k in args.top_k:
        generator.top_k = k
        for retriever in args.retrievers:
            generator.data_file = (
                f"{base_folder}/{retriever.replace('/', '_')}_sorted_run.json"
            )
            out_dir = os.path.join(
                args.output_dir,
                f"top_k_{k}",
                retriever.replace("/", "_"),
            )
            os.makedirs(out_dir, exist_ok=True)
            generator.generate_answers(args.limit, out_dir)


if __name__ == "__main__":
    main()
