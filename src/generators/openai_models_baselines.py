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
    def __init__(self, models, data_file="/data/annotations/label-studio-data-min_filtered.json"):
        self.data_file = data_file
        self.models = models
        self.client = OpenAI()
        
    def encode_image(self, image_path):
        """Encode image to base64 for OpenAI API"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
 
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
            # Encode the image
            base64_image = self.encode_image(image_path)
            
            # Load prompt template and format it
            prompt_template = load_prompt("baseline_prompt.txt")
            prompt = prompt_template.format(question=question)
            
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
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=512,
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content
           
            return {
                'success': True,
                'response': response_text.strip() if response_text else "",
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
       
        # Process all models (no memory management needed for API calls)
        for i, entry in enumerate(data, 1):
            print(f"Entry {i}/{len(data)}")
            image_path = self.get_image_path(entry['image_filename'])
           
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                continue
 
            page_id = os.path.basename(image_path).replace('.png', '')
            
            for model in models:
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
                    
                    # Add small delay to respect rate limits
                    time.sleep(0.1)
       
        # Save results for each model
        for model in models:
            model_safe = model.replace("/", "-").replace(":", "-").replace(".", "-").replace("_", "-")
            output_file = f"{output_dir}/openai_baselines_{model_safe}.json"
           
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
    parser.add_argument('--is_test', action='store_true', help='Use test dataset and save to test results directory')
   
    args = parser.parse_args()
   
    args.is_test = True 
    if not args.models:
        args.models = ['gpt-4o', 'gpt-4o-mini']
       
    if not args.limit:
        args.limit = 2
    
    # Update data file and output directory if is_test is set
    if args.is_test:
        args.data_file = 'data/annotations/label-studio-data-min_filtered_sampled.json'
        args.output_dir = 'src/generators/results/test/baselines'
   
    generator = OpenAIBaselinesGenerator(args.models, args.data_file)
    generator.generate_baselines(args.models, args.limit, args.output_dir)
 
if __name__ == "__main__":
    main()
