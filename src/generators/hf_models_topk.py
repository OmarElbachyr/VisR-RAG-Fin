#!/usr/bin/env python3
import json
import time
import argparse
import os
from transformers import pipeline
from PIL import Image
import torch
from dotenv import load_dotenv
 
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')


class HuggingFaceTopKGenerator:
    def __init__(self, data_file, models, top_k=1):
        self.data_file = data_file
        self.top_k = top_k
        self.model_pipelines = { #FIXME:this code works with transformers==4.53.3 but could break colpali code that uses (4.51.3)  
            model: pipeline(
                task="image-text-to-text",
                model=model,
                device="cuda",
                torch_dtype=torch.float16,
                token=HF_TOKEN,
                trust_remote_code=True,
            )
            for model in models
        } 
        
    def load_data(self):
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} entries")
        return data
    
    def get_image_path(self, filename):
        return f"data/pages/{filename.split('/')[-1]}.png"
    
    def query_model(self, model, question, images):
        start_time = time.time()
        pipe = self.model_pipelines[model]
        try:
            image_count = len(images)
            
            prompt = f"""You are a financial analyst expert at analyzing financial docuemnts.

    You'll be given:
    • Question: "{question}"
    • Context: {image_count} PDF page images that may contain the answer

    Instructions:
    1. Carefully examine all provided images for relevant information
    2. Answer the question using **only** information found in the images
    3. Be precise and concise in your response
    4. If the answer cannot be found in any of the images, respond exactly: "I don't know"
    5. Do not make assumptions or use external knowledge
    6. If information spans multiple images, synthesize it appropriately

    Question: {question}"""
            
            messages = [
                {
                    "role": "user",
                    "content": (
                        [{"type": "image"} for _ in images]
                        + [{"type": "text", "text": prompt}]
                    )
                }
            ]

            with torch.inference_mode(), torch.autocast("cuda"):
                output = pipe(text=messages, images=images, temperature=0.1, max_new_tokens=512)

            # normalize whatever shape generated_text has
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
            return {
                'success': False,
                'response': None,
                'time': time.time() - start_time,
                'error': str(e)
            }


    def generate_answers(self, models, limit=None, output_dir="src/generators/results"):
        print(f"Testing models: {models}")
        
        data = self.load_data()

        # Get sorted question keys and apply limit if specified
        question_keys = sorted([k for k in data.keys() if k.startswith('q')], 
                              key=lambda x: int(x[1:]))
        if limit:
            question_keys = question_keys[:limit]
            
        os.makedirs(output_dir, exist_ok=True)
        results_by_model = {model: [] for model in models}
        
        for i, q_key in enumerate(question_keys, 1):
            print(f"Entry {i}/{len(question_keys)} ({q_key})")
            
            # Get the top-k image filenames (keys) from the 'results' dict, sorted by score descending
            entry = data[q_key]
            results_dict = entry.get('results', {})
            top_k_images = list(results_dict.keys())[:self.top_k]
            images = [self.get_image_path(img) for img in top_k_images]

            if len(images) != self.top_k:
                print(f"Not enough images for top-k (got {len(images)}, need {self.top_k}), skipping entry")
                continue
            
            for model in models:
                print(f"  Model: {model}")
                question = entry.get('query', '')
                print(f"    {question[:50]}...")

                result = self.query_model(model, question, images)

                results_by_model[model].append({
                    'model': model,
                    'q_id': q_key,
                    'question': question,
                    'images': json.dumps(top_k_images),
                    'predicted': result['response'],
                    'success': result['success'],
                    'time': result['time'],
                    'error': result.get('error'),
                })
                
        # Save results - separate file for each model
        for model, model_results in results_by_model.items():
            model_safe = model.replace(":", "-").replace(".", "-").replace("/", "-").replace("_", "-")
            output_file = f"{output_dir}/hf_top-k_{model_safe}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(model_results, f, indent=2, ensure_ascii=False)
            print(f"Results for {model} saved: {output_file}")
        
        # Calculate and display summary statistics
        all_results = [result for model_results in results_by_model.values() for result in model_results]
        successful = sum(1 for r in all_results if r['success'])
        print(f"\nCompleted: {len(all_results)} total, {successful} successful")
        
        # Show per-model statistics
        for model, model_results in results_by_model.items():
            model_successful = sum(1 for r in model_results if r['success'])
            print(f"  {model}: {len(model_results)} total, {model_successful} successful")
 
        return list(results_by_model.values())[0] if results_by_model else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', help='Models to test')
    parser.add_argument('--retrievers', nargs='+', help='Retrievers to test')
    parser.add_argument('--top_k', nargs='+', type=int, default=[1], help='List of top-k values to provide to each model (e.g. --top_k 1 3 5)')
    parser.add_argument('--data_file', default='data/label-studio-data-min.json')
    parser.add_argument('--output_dir', default='src/generators/results')
    parser.add_argument('--limit', type=int, help='Limit entries')

    args = parser.parse_args()

    # Set default values in code (can still be overridden by command line)
    if not args.models:
        args.models =  ['OpenGVLab/InternVL3-8B-hf', 'OpenGVLab/InternVL3-2B-hf'] #, 'OpenGVLab/InternVL3-2B-hf']
    if not args.retrievers:
        args.retrievers = ['nomic-ai/colnomic-embed-multimodal-3b','nomic-ai/colnomic-embed-multimodal-7b']

    data_option = 'annotated_pages' # 'all_pages'

    if not args.limit:
        args.limit = None

    args.top_k = [1, 3]

    for top_k in args.top_k:
        print(f"\n=== Running for top_k: {top_k} ===")
        for retriever in args.retrievers:
            print(f"\n=== Running for retriever: {retriever} ===")
            data_file = f'data/retrieved_pages/{data_option}/{retriever.replace("/", "_")}_sorted_run.json'
            retriever_subdir = os.path.join(args.output_dir, f'top_k_{top_k}', retriever.replace("/", "_"))
            os.makedirs(retriever_subdir, exist_ok=True)
            generator = HuggingFaceTopKGenerator(data_file, models=args.models, top_k=top_k)
            generator.generate_answers(args.models, args.limit, retriever_subdir)


if __name__ == "__main__":
    main()