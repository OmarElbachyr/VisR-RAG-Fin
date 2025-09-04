#!/usr/bin/env python3
import json
import time
import argparse
import os
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
 
load_dotenv()


class GeminiTopKGenerator:
    def __init__(self, data_file, annotations_file, models, top_k=1):
        self.data_file = data_file
        self.annotations_file = annotations_file
        self.top_k = top_k
        self.models = models
        
        # Configure Gemini API
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        
        # Initialize model clients
        self.model_clients = {}
        for model in models:
            self.model_clients[model] = genai.GenerativeModel(model)
        
    def load_data(self):
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} entries")

        with open(self.annotations_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        print(f"Loaded {len(annotations)} annotations")
        return data, annotations
    
    def get_image_path(self, filename):
        return f"data/pages/{filename.split('/')[-1]}.png"
    
    def query_model(self, model, question, images):
        start_time = time.time()
        
        try:
            # Load all images
            pil_images = []
            for image_path in images:
                image = Image.open(image_path)
                pil_images.append(image)
            
            image_count = len(images)
            
            prompt = f"""You are a financial analyst expert at analyzing financial documents.

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
            
            # Build content list with prompt and all images
            content = [prompt] + pil_images
            
            model_client = self.model_clients[model]
            response = model_client.generate_content(
                content,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=512,
                    temperature=0.1
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

    def generate_answers(self, models, limit=None, output_dir="src/generators/results/retrieval_pipeline"):
        print(f"Testing models: {models}")
        
        data, annotations = self.load_data()

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

            # Check if all images exist
            existing_images = []
            for image_path in images:
                if os.path.exists(image_path):
                    existing_images.append(image_path)
                else:
                    print(f"Image not found: {image_path}")

            if len(existing_images) != self.top_k:
                print(f"Not enough images for top-k (got {len(existing_images)}, need {self.top_k}), skipping entry")
                continue

            # Find the ground truth answer and metadata in annotations
            ground_truth_answer = None
            ground_truth_type = None
            ground_truth_evidence = None
            company = None
            file_category = None
            language = None
            
            for annotation in annotations:
                for qa_pair in annotation.get('qa_pairs', []):
                    if qa_pair.get('question_id') == q_key:
                        ground_truth_answer = qa_pair.get('answer')
                        ground_truth_type = qa_pair.get('type')
                        ground_truth_evidence = qa_pair.get('evidence')
                        company = annotation.get('company')
                        file_category = annotation.get('file_category')
                        language = annotation.get('language')
                        break
                
                # Fallback: if qa_pairs is empty or missing, try to get from direct fields
                if ground_truth_answer is None:
                    answer_key = f"a{q_key[1:]}"  # Convert q1 -> a1, q2 -> a2, etc.
                    type_key = f"type{q_key[1:]}"  # Convert q1 -> type1, q2 -> type2, etc.
                    evidence_key = f"evidence{q_key[1:]}"  # Convert q1 -> evidence1, q2 -> evidence2, etc.
                    
                    if answer_key in annotation:
                        ground_truth_answer = annotation.get(answer_key)
                        ground_truth_type = annotation.get(type_key)
                        ground_truth_evidence = annotation.get(evidence_key)
                        company = annotation.get('company')
                        file_category = annotation.get('file_category')
                        language = annotation.get('language')
                        break
                
                if ground_truth_answer is not None:
                    break
            
            for model in models:
                print(f"  Model: {model}")
                question = entry.get('query', '')
                print(f"    {question[:50]}...")

                result = self.query_model(model, question, existing_images)

                results_by_model[model].append({
                    'model': model,
                    'q_id': q_key,
                    'question': question,
                    'relevant_pages': top_k_images,
                    'ground_truth': ground_truth_answer,
                    'evidence': ground_truth_evidence,
                    'type': ground_truth_type,
                    'predicted': result['response'],
                    'success': result['success'],
                    'time': result['time'],
                    'error': result.get('error'),
                    'company': company,
                    'file_category': file_category,
                    'language': language
                })
                
                # Add small delay to respect rate limits
                time.sleep(0.1)
        
        # Save results - separate file for each model
        for model, model_results in results_by_model.items():
            model_safe = model.replace(":", "-").replace(".", "-").replace("/", "-").replace("_", "-")
            output_file = f"{output_dir}/gemini_top-k_{model_safe}.json"
            
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
    parser.add_argument('--annotations_file', default='data/annotations/label-studio-data-min_filtered.json')
    parser.add_argument('--top_k', nargs='+', type=int, help='List of top-k values to provide to each model (e.g. --top_k 1 3 5)')
    parser.add_argument('--output_dir', default='src/generators/results/retrieval_pipeline', help='Directory containing end-to-end retrieval + generation pipeline results')
    parser.add_argument('--limit', type=int, help='Limit entries')
    parser.add_argument('--is_test', action='store_true', help='Use test dataset and save to test results directory')

    args = parser.parse_args()
    
    args.is_test = True 
    if not args.models:
        args.models = ['gemini-1.5-pro', 'gemini-1.5-flash']
    if not args.retrievers:
        args.retrievers = ['nomic-ai/colnomic-embed-multimodal-3b','nomic-ai/colnomic-embed-multimodal-7b']

    data_option = 'annotated_pages_test' if args.is_test else 'annotated_pages'

    if not args.limit:
        args.limit = 1

    if not args.top_k:
        args.top_k = [1, 3]

    # Update annotations file and output directory if is_test is set
    if args.is_test:
        args.annotations_file = 'data/annotations/label-studio-data-min_filtered_sampled.json'
        args.output_dir = 'src/generators/results/test/retrieval_pipeline'

    for top_k in args.top_k:
        print(f"\n=== Running for top_k: {top_k} ===")
        
        for retriever in args.retrievers:
            print(f"\n=== Running for retriever: {retriever} ===")
            data_file = f'src/retrievers/results/{data_option}/retrieved_pages/{retriever.replace("/", "_")}_sorted_run.json'
            
            # Create the generator for this specific configuration
            generator = GeminiTopKGenerator(data_file, args.annotations_file, models=args.models, top_k=top_k)
            
            retriever_subdir = os.path.join(args.output_dir, f'top_k_{top_k}', retriever.replace("/", "_"))
            os.makedirs(retriever_subdir, exist_ok=True)
            generator.generate_answers(args.models, args.limit, retriever_subdir)


if __name__ == "__main__":
    main()
