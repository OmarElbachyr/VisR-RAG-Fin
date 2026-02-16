#!/usr/bin/env python3
import json
import time
import argparse
import os
from google import genai
from google.genai import types
from PIL import Image
import io
from dotenv import load_dotenv
from generators.prompt_utils import load_prompt
 
load_dotenv()

class GeminiBaselinesGenerator:
    def __init__(self, models, data_file="/data/annotations/label-studio-data-min_filtered.json"):
        self.data_file = data_file
        self.models = models
        
        # Configure Gemini API
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        self.client = genai.Client(api_key=api_key)
        
        # Initialize model clients
        self.model_clients = {model: model for model in models}
        
    def load_data(self):
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} entries")
        return data
   
    def get_image_path(self, filename):
        return f"data/pages/{filename.split('/')[-1]}"
 
    def query_model(self, model, question, image_path):
        start_time = time.time()
        
        try:
            # Load and prepare the image
            image = Image.open(image_path)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            buffer.seek(0)
            image_part = types.Part.from_bytes(
                data=buffer.read(),
                mime_type="image/jpeg"
            )
            
            # Load prompt template and format it
            prompt_template = load_prompt("baseline_prompt.txt")
            prompt = prompt_template.format(question=question)
            
            response = self.client.models.generate_content(
                model=model,
                contents=[prompt, image_part],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                )
            )
            
            response_text = response.text if response.text else ""
           
            return {
                'success': True,
                'response': response_text.strip(),
                'time': time.time() - start_time
            }
       
        except Exception as e:
            return {
                'success': False,
                'response': None,
                'time': time.time() - start_time,
                'error': str(e)
            }
   
    def generate_baselines(self, models, limit=None, output_dir="src/generators/results/baselines"):
        print(f"Testing models: {models}")
       
        data = self.load_data()
        if limit:
            data = data[:limit]
       
        os.makedirs(output_dir, exist_ok=True)
        results_by_model = {model: [] for model in models}
       
        # Process all models
        for model in models:
            print(f"Processing model: {model}")

            for i, entry in enumerate(data, 1):
                print(f"Entry {i}/{len(data)}")
                image_path = self.get_image_path(entry['image_filename'])
               
                if not os.path.exists(image_path):
                    print(f"Image not found: {image_path}")
                    continue
     
                page_id = os.path.basename(image_path).replace('.png', '')
                
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
                    
                    # Add small delay to respect rate limits
                    time.sleep(0.1)
       
        # Save results for each model
        for model in models:
            model_safe = model.replace("/", "-").replace(":", "-").replace(".", "-").replace("_", "-")
            output_file = f"{output_dir}/gemini_baselines_{model_safe}.json"
           
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results_by_model[model], f, indent=2, ensure_ascii=False)
            print(f"Results for {model} saved: {output_file}")
       
        # Summary
        all_results = [result for model_results in results_by_model.values() for result in model_results]
        successful = sum(1 for r in all_results if r['success'])
        print(f"\nCompleted: {len(all_results)} total, {successful} successful")
       
        for model, model_results in results_by_model.items():
            model_successful = sum(1 for r in model_results if r['success'])
            print(f"  {model}: {len(model_results)} total, {model_successful} successful")
       
        return list(results_by_model.values())[0] if results_by_model else None
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', help='Models to test')
    parser.add_argument('--data_file', default='data/annotations/label-studio-data-min_filtered.json')
    parser.add_argument('--output_dir', default='src/generators/results/baselines')
    parser.add_argument('--limit', type=int, help='Limit entries')
   
    args = parser.parse_args()
   
    args.is_test = True 
    if not args.models:
        # Add models you want to test here, or pass them via command line
        args.models = ['gemini-3-pro-preview']
       
    if not args.limit:
        args.limit = 1

    is_category_a = True # set to True to consider only Category A QAs from the first pass classification
    is_category_b = False # set to True to consider only Category B QAs from the first pass classification after rewriting them and being classified as category A
    
    if is_category_a:
        args.data_file = 'data/annotations/final_annotations/filtered_annotations/by_category/first_pass_classified_qa_category_A.json'
        args.output_dir = 'src/generators/results/category_a/baselines'

    if is_category_b:
        args.data_file = 'data/annotations/final_annotations/filtered_annotations/by_category/first_pass_classified_qa_category_B.json'
        args.output_dir = 'src/generators/results/category_b/baselines'
   
    generator = GeminiBaselinesGenerator(args.models, args.data_file)
    generator.generate_baselines(args.models, args.limit, args.output_dir)
 
if __name__ == "__main__":
    main()
