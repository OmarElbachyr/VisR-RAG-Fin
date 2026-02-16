#!/usr/bin/env python3
import subprocess
import sys

# Install required transformers version
# old was 4.53.3, colqwen-3vl needs 4.57.0
def ensure_transformers_version():
    transformers_version = "4.57.0"
    try:
        import transformers
        if transformers.__version__ != transformers_version:
            print(f"Current transformers version: {transformers.__version__}")
            print(f"Installing transformers=={transformers_version}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", f"transformers=={transformers_version}"])
            print(f"Successfully installed transformers=={transformers_version}")
            print("Please restart the script to use the new transformers version.")
            sys.exit(0)
        else:
            print(f"Using transformers version: {transformers.__version__}")
    except ImportError:
        print(f"Installing transformers=={transformers_version}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"transformers=={transformers_version}"])
        print(f"Successfully installed transformers=={transformers_version}")

ensure_transformers_version()

import json
import time
import argparse
import os
import gc
from transformers import pipeline
from PIL import Image
import torch
from dotenv import load_dotenv
from generators.prompt_utils import load_prompt

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')


class HuggingFaceTopKGenerator:
    def __init__(self, data_file, annotations_file, models, top_k=1, filter_top_k_hits=False):
        self.data_file = data_file
        self.annotations_file = annotations_file
        self.top_k = top_k
        self.models = models
        self.filter_top_k_hits = filter_top_k_hits
        self.model_pipelines = {}
        self.device = "cuda"

    def _load_model_pipeline(self, model):
        if model not in self.model_pipelines:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            self.model_pipelines[model] = pipeline(
                task="image-text-to-text",
                model=model,
                device=self.device,
                torch_dtype="auto",
                token=HF_TOKEN,
                trust_remote_code=True,
            )

    def _clear_model_pipeline(self, model):
        if model in self.model_pipelines:
            del self.model_pipelines[model]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

    def load_data(self):
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} entries")

        with open(self.annotations_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        print(f"Loaded {len(annotations)} annotations")
        return data, annotations

    def get_image_path(self, filename):
        page_path = f"data/pages/{filename.split('/')[-1]}.png"
        noise_path = f"data/hard_negative_pages_text/{filename.split('/')[-1]}.png"
        return page_path if os.path.exists(page_path) else noise_path

    def query_model(self, model, question, images):
        start_time = time.time()
        self._load_model_pipeline(model)
        pipe = self.model_pipelines[model]
        try:
            image_count = len(images)
            prompt_template = load_prompt("topk_prompt.txt")
            prompt = prompt_template.format(question=question, image_count=image_count)
            messages = [
                {
                    "role": "user",
                    "content": (
                        [{"type": "image"} for _ in images]
                        + [{"type": "text", "text": prompt}]
                    )
                }
            ]
            device_type = self.device.split(':')[0] if ':' in self.device else self.device
            with torch.inference_mode(), torch.autocast(device_type):
                output = pipe(text=messages, images=images, temperature=0.1, max_new_tokens=512)
            gen = output[0].get("generated_text", output[0])
            if isinstance(gen, list):
                last = gen[-1]
                if isinstance(last, dict) and "content" in last:
                    response = last["content"]
                else:
                    response = last
            elif isinstance(gen, dict):
                response = gen.get("content", gen.get("message", {}).get("content", str(gen)))
            else:
                response = gen
            return {
                'success': True,
                'response': response.strip() if isinstance(response, str) else str(response),
                'time': time.time() - start_time
            }
        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {
                'success': False,
                'response': None,
                'time': time.time() - start_time,
                'error': str(e)
            }
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate_answers(self, models, limit=None, output_dir="src/generators/results/retrieval_pipeline"):
        print(f"Testing models: {models}")
        data, annotations = self.load_data()
        question_keys = sorted([k for k in data.keys() if k.startswith('q')], key=lambda x: int(x[1:]))
        if limit:
            question_keys = question_keys[:limit]
        os.makedirs(output_dir, exist_ok=True)

        qa_lookup = {}
        for annotation in annotations:
            company = annotation.get('company')
            file_category = annotation.get('file_category')
            language = annotation.get('language')
            image_filename = annotation.get('image_filename')
            real_relevant_page = None
            if image_filename:
                real_relevant_page = image_filename.split('/')[-1].split('.')[0]

            for qa_pair in annotation.get('qa_pairs', []):
                q_id = qa_pair.get('question_id')
                qa_lookup[q_id] = {
                    'answer': qa_pair.get('answer'),
                    'type': qa_pair.get('type'),
                    'evidence': qa_pair.get('evidence'),
                    'company': company,
                    'file_category': file_category,
                    'language': language,
                    'real_relevant_page': real_relevant_page
                }

        question_keys = [k for k in question_keys if k in qa_lookup]

        filtered_keys = []
        for q_key in question_keys:
            results_dict = data[q_key].get('results', {})
            top_k_images = list(results_dict.keys())[:self.top_k]
            if not self.filter_top_k_hits or qa_lookup[q_key]['real_relevant_page'] in top_k_images:
                filtered_keys.append(q_key)

        print(f"Processing {len(filtered_keys)} queries after filtering")

        results_by_model = {model: [] for model in models}
        for model in models:
            print(f"Processing model: {model}")
            self._load_model_pipeline(model)
            for i, q_key in enumerate(filtered_keys, 1):
                print(f"Entry {i}/{len(filtered_keys)} ({q_key})")
                entry = data[q_key]
                results_dict = entry.get('results', {})
                top_k_images = list(results_dict.keys())[:self.top_k]
                images = [self.get_image_path(img) for img in top_k_images]
                if len(images) != self.top_k:
                    continue

                qa_data = qa_lookup.get(q_key, {})
                question = entry.get('query', '')
                result = self.query_model(model, question, images)

                results_by_model[model].append({
                    'model': model,
                    'q_id': q_key,
                    'question': question,
                    'relevant_pages': top_k_images,
                    'real_relevant_page': qa_data.get('real_relevant_page'),
                    'ground_truth': qa_data.get('answer'),
                    'evidence': qa_data.get('evidence'),
                    'type': qa_data.get('type'),
                    'predicted': result['response'],
                    'success': result['success'],
                    'time': result['time'],
                    'error': result.get('error'),
                    'company': qa_data.get('company'),
                    'file_category': qa_data.get('file_category'),
                    'language': qa_data.get('language')
                })
            self._clear_model_pipeline(model)
            model_safe = model.replace(":", "-").replace(".", "-").replace("/", "-").replace("_", "-")
            output_file = f"{output_dir}/hf_top-k_{model_safe}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results_by_model[model], f, indent=2, ensure_ascii=False)
            print(f"Results for {model} saved: {output_file}")

        all_results = [r for model_results in results_by_model.values() for r in model_results]
        successful = sum(1 for r in all_results if r['success'])

        print(f"\nCompleted: {len(all_results)} total, {successful} successful")
        for model, model_results in results_by_model.items():
            model_successful = sum(1 for r in model_results if r['success'])
            print(f"  {model}: {len(model_results)} total, {model_successful} successful")

        gc.collect()
        torch.cuda.empty_cache()
        return list(results_by_model.values())[0] if results_by_model else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--retrievers', nargs='+')
    parser.add_argument('--annotations_file')
    parser.add_argument('--top_k', nargs='+', type=int)
    parser.add_argument('--output_dir')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--filter_top_k_hits', action='store_true')

    args = parser.parse_args()

    device = "cuda:1"

    if not args.models:
        # Add models you want to test here, or pass them via command line
        args.models =  [
                        "OpenGVLab/InternVL3_5-2B-HF",
                        "OpenGVLab/InternVL3_5-4B-HF",
                       ]
    
    if not args.retrievers:
        # Add retrievers you want to test here, or pass them via command line
        args.retrievers = ['nomic-ai/colnomic-embed-multimodal-7b']

    if not args.limit:
        args.limit = 1

    if not args.top_k:
        args.top_k = [1]

    args.filter_top_k_hits = True

    args.annotations_file = 'data/annotations/final_annotations/filtered_annotations/by_category/first_pass_classified_qa_category_A.json'
    args.output_dir = 'src/generators/results/category_a/retrieval_pipeline'

    generator = HuggingFaceTopKGenerator(
        None,
        args.annotations_file,
        models=args.models,
        top_k=1,
        filter_top_k_hits=args.filter_top_k_hits
    )
    generator.device = device

    base_folder = "src/retrievers/results/chunked_pages_second_pass_rq1/0.5/retrieved_pages"

    for top_k in args.top_k:
        generator.top_k = top_k
        for retriever in args.retrievers:
            data_file = f"{base_folder}/{retriever.replace('/', '_')}_sorted_run.json"
            generator.data_file = data_file
            retriever_subdir = os.path.join(args.output_dir, f"top_k_{top_k}", retriever.replace("/", "_"))
            os.makedirs(retriever_subdir, exist_ok=True)
            generator.generate_answers(args.models, args.limit, retriever_subdir)


if __name__ == "__main__":
    main()