#!/usr/bin/env python3
import subprocess
import sys

# Install required transformers version
def ensure_transformers_version():
    """Ensure transformers==4.53.3 is installed"""
    try:
        import transformers
        if transformers.__version__ != "4.53.3":
            print(f"Current transformers version: {transformers.__version__}")
            print("Installing transformers==4.53.3...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==4.53.3"])
            print("Successfully installed transformers==4.53.3")
            # Need to restart to use new version
            print("Please restart the script to use the new transformers version.")
            sys.exit(0)
        else:
            print(f"Using transformers version: {transformers.__version__}")
    except ImportError:
        print("Installing transformers==4.53.3...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==4.53.3"])
        print("Successfully installed transformers==4.53.3")

# Ensure correct transformers version before proceeding
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

class HFBaselinesGenerator:
    def __init__(self, models, data_file="/data/annotations/label-studio-data-min_filtered.json"):
        self.data_file = data_file
        self.models = models
        self.model_pipelines = {}
        
    def _load_model_pipeline(self, model):
        """Load a single model pipeline to manage GPU memory better"""
        if model not in self.model_pipelines:
            # Clear GPU memory before loading new model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            self.model_pipelines[model] = pipeline(
                task="image-text-to-text",
                model=model,
                device="cuda",
                torch_dtype="auto",
                token=HF_TOKEN,
                trust_remote_code=True,
            )
    
    def _clear_model_pipeline(self, model):
        """Clear a specific model pipeline from memory"""
        if model in self.model_pipelines:
            del self.model_pipelines[model]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect() 
 
    def load_data(self):
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} entries")
        return data
   
    def get_image_path(self, filename):
        return f"data/pages/{filename.split('/')[-1]}"
 
    def query_model(self, model, question, image_path):
        start_time = time.time()
        image = Image.open(image_path).convert("RGB")
        
        # Load model pipeline on demand
        self._load_model_pipeline(model)
        pipe = self.model_pipelines[model]
 
        try:
            # Load prompt template and format it
            prompt_template = load_prompt("baseline_prompt.txt")
            prompt = prompt_template.format(question=question)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
 
            with torch.inference_mode(), torch.autocast("cuda"):
                output = pipe(text=messages, temperature=0.1, max_new_tokens=512)
            
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
            # Clear cache on error to prevent memory issues
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {
                'success': False,
                'response': None,
                'time': time.time() - start_time,
                'error': str(e)
            }
        finally:
            # Always clear cache after inference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
   
    def generate_baselines(self, models, limit=None, output_dir="src/generators/results/baselines"):
        print(f"Testing models: {models}")
       
        data = self.load_data()
        if limit:
            data = data[:limit]
       
        os.makedirs(output_dir, exist_ok=True)
        results_by_model = {model: [] for model in models}
       
        # Process one model at a time to save memory
        for model in models:
            print(f"Processing model: {model}")
            
            # Load model pipeline for current model only
            self._load_model_pipeline(model)
            
            for i, entry in enumerate(data, 1):
                print(f"Entry {i}/{len(data)}")
                image_path = self.get_image_path(entry['image_filename'])
               
                if not os.path.exists(image_path):
                    print(f"Image not found: {image_path}")
                    continue
     
                page_id = os.path.basename(image_path).replace('.png', '')
                
                print(f"  Model: {model}")
                for qa in entry.get('qa_pairs', []):
                    print(f"    {qa['question'][:50]}...")
                   
                    result = self.query_model(model, qa['question'], image_path)
                   
                    results_by_model[model].append({
                        'model': model,
                        'q_id': qa['question_id'],
                        'question': qa['question'],
                        'relevant_pages': [page_id],
                        'ground_truth': qa['answer'],
                        'evidence': qa['evidence'],
                        'type': qa['type'],
                        'predicted': result['response'],
                        'success': result['success'],
                        'time': result['time'],
                        'error': result.get('error'),
                        **{k: entry.get(k) for k in ['company', 'file_category', 'language']}
                    })
                    
            # Clear current model pipeline to free memory before next model
            self._clear_model_pipeline(model)
            print(f"Completed processing for {model}")
            
            # Save results immediately for this model
            model_safe = model.replace("/", "-").replace(":", "-").replace(".", "-").replace("_", "-")
            output_file = f"{output_dir}/hf_baselines_{model_safe}.json"
           
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results_by_model[model], f, indent=2, ensure_ascii=False)
            print(f"Results for {model} saved: {output_file}")
       
        # Final cleanup and summary
        all_results = [result for model_results in results_by_model.values() for result in model_results]
        successful = sum(1 for r in all_results if r['success'])
        print(f"\nCompleted: {len(all_results)} total, {successful} successful")
       
        for model, model_results in results_by_model.items():
            model_successful = sum(1 for r in model_results if r['success'])
            print(f"  {model}: {len(model_results)} total, {model_successful} successful")
       
        # Clean up GPU memory
        gc.collect()
        torch.cuda.empty_cache()
        return list(results_by_model.values())[0] if results_by_model else None
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', help='Models to test')
    parser.add_argument('--data_file', default='data/annotations/label-studio-data-min_filtered.json')
    parser.add_argument('--output_dir', default='src/generators/results/baselines')
    parser.add_argument('--limit', type=int, help='Limit entries')
    parser.add_argument('--is_test', action='store_true', help='Use test dataset and save to test results directory')
   
    args = parser.parse_args()
   
    args.is_test = True 
    if not args.models:
        args.models = ['OpenGVLab/InternVL3-8B-hf', 'OpenGVLab/InternVL3-2B-hf'] #, don't work: 'Qwen/Qwen2.5-VL-3B-Instruct', 'Qwen/Qwen2.5-VL-7B-Instruct']
        args.models = ['google/gemma-3-12b-it']
       
    if not args.limit:
        args.limit = 1
    
    # Update data file and output directory if is_test is set
    if args.is_test:
        args.data_file = 'data/annotations/label-studio-data-min_filtered_sampled.json'
        args.output_dir = 'src/generators/results/test/baselines'
   
    generator = HFBaselinesGenerator(args.models, args.data_file)
    generator.generate_baselines(args.models, args.limit, args.output_dir)
 
if __name__ == "__main__":
    main()
