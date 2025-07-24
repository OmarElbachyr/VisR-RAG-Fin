#!/usr/bin/env python3
import json
import time
import argparse
import os
from pathlib import Path
import ollama


class OllamaTopKGenerator:
    def __init__(self, data_file, top_k):
        self.data_file = data_file
        self.top_k = top_k  # Default top-k value
        
    def load_data(self):
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} entries")
        return data
    
    def get_image_path(self, filename):
        return f"data/all_pages/{filename.split('/')[-1]}.png"
    
    def query_model(self, model, question, images):
        start_time = time.time()
        try:
            image_count = len(images)
            image_refs = ", ".join([f"image {i+1}" for i in range(image_count)])
            
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

            response = ollama.chat(
                model=model,
                messages=[
                    {'role': 'user', 
                     'content': prompt, 
                     'images': images
                     }
                ],
                options={"temperature": 0.1}
            )
            return {
                'success': True,
                'response': response['message']['content'],
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
            model_safe = model.replace(":", "_").replace(".", "_")
            output_file = f"{output_dir}/ollama_baselines_{model_safe}.json"
            
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
    parser.add_argument('--top_k', default='1', help='Number of top-k pages to provide to each model')
    parser.add_argument('--data_file', default='data/label-studio-data-min.json')
    parser.add_argument('--output_dir', default='src/generators/results')
    parser.add_argument('--limit', type=int, help='Limit entries')
    
    args = parser.parse_args()
    
    # Set default values in code (can still be overridden by command line)
    if not args.models:
        args.models = ['qwen2.5vl:3b', 'gemma3:4b-it-q4_K_M',
                       'qwen2.5vl:7b', 'gemma3:12b-it-q4_K_M']
    if not args.retrievers:
        args.retrievers = ['vidore/colqwen2.5-v0.2','nomic-ai/nomic-embed-multimodal-3b', 
                           'nomic-ai/nomic-embed-multimodal-7b']
        
    data_option = 'annotated_pages' # 'all_pages'
        
    if not args.limit:
        args.limit = None
    
    args.top_k = 3
    
    try:
        ollama.list()
        print("✓ Ollama connected")
    except Exception as e:
        print(f"✗ Ollama error: {e}")
        return
    
    for retriever in args.retrievers:
        print(f"\n=== Running for retriever: {retriever} ===")
        data_file = f'data/retrieved_pages/{data_option}/{retriever.replace("/", "_")}_sorted_run.json'
        # print(data_file)
        retriever_subdir = os.path.join(args.output_dir, f'top_k_{args.top_k}', retriever.replace("/", "_"))
        os.makedirs(retriever_subdir, exist_ok=True)
        generator = OllamaTopKGenerator(data_file, top_k=args.top_k)
        generator.generate_answers(args.models, args.limit, retriever_subdir)


if __name__ == "__main__":
    main()