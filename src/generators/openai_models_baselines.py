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

class OpenAIBaselinesGenerator:
    def __init__(self, models, data_file="data/annotations/label-studio-data-min_filtered.json"):
        self.data_file = data_file
        self.models = models
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)

    def load_data(self):
        with open(self.data_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} entries")
        return data

    def get_image_path(self, filename):
        return f"data/pages/{filename.split('/')[-1]}"

    def encode_image(self, image_path):
        """Helper to convert local image to base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def query_model(self, model, question, image_path):
        """Query OpenAI Chat Completions with Vision."""
        start_time = time.time()
        try:
            base64_image = self.encode_image(image_path)
            prompt_template = load_prompt("baseline_prompt.txt")
            prompt = prompt_template.format(question=question)

            # Use Chat Completions API instead of Responses API
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
            )

            response_text = response.choices[0].message.content or ""

            return {
                "success": True,
                "response": response_text.strip(),
                "time": time.time() - start_time,
            }

        except Exception as e:
            return {
                "success": False,
                "response": None,
                "time": time.time() - start_time,
                "error": str(e),
            }

    def generate_baselines(self, models, limit=None, output_dir="src/generators/results/baselines"):
        print(f"Testing models: {models}")
        data = self.load_data()
        if limit:
            data = data[:limit]

        os.makedirs(output_dir, exist_ok=True)
        results_by_model = {model: [] for model in models}

        for model in models:
            print(f"\nProcessing model: {model}")
            for i, entry in enumerate(data, 1):
                image_path = self.get_image_path(entry["image_filename"])
                if not os.path.exists(image_path):
                    print(f"Image not found: {image_path}")
                    continue

                page_id = os.path.basename(image_path).split('.')[0]

                for qa in entry.get("qa_pairs", []):
                    print(f"Entry {i}/{len(data)} - Q: {qa['question'][:50]}...")
                    
                    result = self.query_model(
                        model=model,
                        question=qa["question"],
                        image_path=image_path,
                    )

                    results_by_model[model].append({
                        "model": model,
                        "q_id": qa["question_id"],
                        "question": qa["question"],
                        "relevant_pages": [page_id],
                        "ground_truth": qa["answer"],
                        "evidence": qa["evidence"],
                        "type": qa["type"],
                        "predicted": result["response"],
                        "success": result["success"],
                        "time": result["time"],
                        "error": result.get("error"),
                        **{k: entry.get(k) for k in ["company", "file_category", "language"]}
                    })
                    time.sleep(0.1) # Respectful rate limiting

        # Save Logic
        for model in models:
            model_safe = model.replace("/", "-").replace(":", "-").replace(".", "-")
            output_file = os.path.join(output_dir, f"openai_baselines_{model_safe}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results_by_model[model], f, indent=2, ensure_ascii=False)
            print(f"Saved results for {model}: {output_file}")

        # Summary
        all_results = [result for model_results in results_by_model.values() for result in model_results]
        successful = sum(1 for r in all_results if r['success'])
        print(f"\nCompleted: {len(all_results)} total, {successful} successful")
       
        for model, model_results in results_by_model.items():
            model_successful = sum(1 for r in model_results if r['success'])
            print(f"  {model}: {len(model_results)} total, {model_successful} successful")
        
        return results_by_model


# --------------------------------------------------
# Entry point
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", help="Models to test")
    parser.add_argument("--data_file", default="data/annotations/label-studio-data-min_filtered.json")
    parser.add_argument("--output_dir", default="src/generators/results/baselines")
    parser.add_argument("--limit", type=int)

    args = parser.parse_args()

    if not args.models:
        # Add models you want to test here, or pass them via command line
        args.models = ["gpt-5.2-2025-12-11"]

    if not args.limit:
        args.limit = None

    is_category_a = True
    is_category_b = False

    if is_category_a:
        args.data_file = "data/annotations/final_annotations/filtered_annotations/by_category/first_pass_classified_qa_category_A.json"
        args.output_dir = "src/generators/results/category_a/baselines"

    if is_category_b:
        args.data_file = "data/annotations/final_annotations/filtered_annotations/by_category/first_pass_classified_qa_category_B.json"
        args.output_dir = "src/generators/results/category_b/baselines"

    generator = OpenAIBaselinesGenerator(args.models, args.data_file)
    generator.generate_baselines(args.models, args.limit, args.output_dir)


if __name__ == "__main__":
    main()
