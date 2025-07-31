#!/usr/bin/env python3
import json
import time
import argparse
import os
from pathlib import Path
import ollama
from datetime import datetime


class ImageCompatibilityTester:
    def __init__(self, data_file="data/label-studio-data-min.json"):
        self.data_file = data_file
        
    def load_data(self):
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} entries")
        return data
    
    def get_image_path(self, filename):
        return f"data/all_pages/{filename.split('/')[-1]}"
    
    def get_image_info(self, image_path):
        """Get image dimensions and file size"""
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                width, height = img.size
                file_size = os.path.getsize(image_path)
            return width, height, file_size
        except Exception as e:
            print(f"Could not read image info for {image_path}: {e}")
            return None, None, None
    
    def test_model_with_image(self, model, image_path):
        """Test if a model can accept an image without generating full response"""
        try:
            start_time = time.time()
            response = ollama.chat(
                model=model,
                messages=[{
                    'role': 'user', 
                    'content': 'Can you see this image? Reply with just "yes" or "no".', 
                    'images': [image_path]
                }],
                options={
                    'num_predict': 10,  # Limit response length for speed
                    'temperature': 0    # Make it deterministic
                }
            )
            
            return {
                'success': True,
                'response': response['message']['content'].strip()[:50],
                'time': time.time() - start_time,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'response': None,
                'time': time.time() - start_time,
                'error': {
                    'type': type(e).__name__,
                    'message': str(e),
                    'args': e.args
                }
            }
    
    def test_image_compatibility(self, models, limit=None, output_dir="src/generators/baselines/results"):
        """Test if models can accept images of different dimensions/resolutions"""
        print(f"Testing image compatibility for models: {models}")
        
        data = self.load_data()
        if limit and limit > 0:
            data = data[:limit]
        
        os.makedirs(output_dir, exist_ok=True)
        compatibility_results = {model: {'compatible': [], 'incompatible': []} for model in models}
        
        # Filter out entries with missing images first
        valid_entries = []
        for entry in data:
            image_path = self.get_image_path(entry['image_filename'])
            if os.path.exists(image_path):
                valid_entries.append(entry)
            else:
                print(f"Image not found, skipping: {image_path}")
        
        print(f"Found {len(valid_entries)} valid images out of {len(data)} entries")
        
        for i, entry in enumerate(valid_entries, 1):
            print(f"Testing image {i}/{len(valid_entries)}")
            image_path = self.get_image_path(entry['image_filename'])

            page_id = os.path.basename(image_path).replace('.png', '')
            width, height, file_size = self.get_image_info(image_path)
            
            for model in models:
                print(f"  Testing {model} with {page_id}")
                
                result = self.test_model_with_image(model, image_path)
                
                image_info = {
                    'page_id': page_id,
                    'image_path': image_path,
                    'width': width,
                    'height': height,
                    'file_size_bytes': file_size,
                    'test_time': result['time']
                }
                
                if result['success']:
                    image_info['response'] = result['response']
                    compatibility_results[model]['compatible'].append(image_info)
                    print(f"    ✓ Compatible - Response: {result['response']}")
                else:
                    image_info['error'] = result['error']
                    compatibility_results[model]['incompatible'].append(image_info)
                    error_msg = result['error']['message'] if result['error'] else 'Unknown error'
                    print(f"    ✗ Error: {error_msg[:100]}")
        
        # Save incompatible images only to errors directory
        incompatible_dir = f"{output_dir}/errors/incompatible_images"
        os.makedirs(incompatible_dir, exist_ok=True)
        
        # Clear existing incompatible image files
        for f in os.listdir(incompatible_dir):
            if f.endswith('.json'):
                os.remove(os.path.join(incompatible_dir, f))
        
        # Save separate incompatible images file for each model
        for model, results in compatibility_results.items():
            if results['incompatible']:
                model_safe = model.replace(":", "_").replace(".", "_")
                incompatible_file = f"{incompatible_dir}/incompatible_images_{model_safe}.json"
                with open(incompatible_file, 'w', encoding='utf-8') as f:
                    json.dump(results['incompatible'], f, indent=2, ensure_ascii=False)
                print(f"Incompatible images for {model} saved: {incompatible_file}")
        
        # Print summary and save to text file
        stats_content = self.get_compatibility_summary_text(compatibility_results)
        print(stats_content)
        
        # Save stats to text file
        stats_file = f"{output_dir}/compatibility_stats.txt"
        with open(stats_file, 'w', encoding='utf-8') as f:
            f.write(f"Image Compatibility Test Results\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total entries loaded: {len(data)}\n")
            f.write(f"Valid images found: {len(valid_entries)}\n\n")
            f.write(stats_content)
        
        print(f"Compatibility stats saved to: {stats_file}")
        return compatibility_results
    
    def get_compatibility_summary_text(self, compatibility_results):
        """Get a summary of compatibility test results as text"""
        summary_lines = []
        summary_lines.append("="*60)
        summary_lines.append("IMAGE COMPATIBILITY TEST RESULTS")
        summary_lines.append("="*60)
        
        for model, results in compatibility_results.items():
            compatible_count = len(results['compatible'])
            incompatible_count = len(results['incompatible'])
            total = compatible_count + incompatible_count
            
            if total > 0:
                success_rate = (compatible_count / total) * 100
                summary_lines.append(f"\n{model}:")
                summary_lines.append(f"  Compatible: {compatible_count}/{total} images ({success_rate:.1f}%)")
                
                if results['compatible']:
                    # Get image dimension stats for compatible images
                    widths = [img['width'] for img in results['compatible'] if img['width']]
                    heights = [img['height'] for img in results['compatible'] if img['height']]
                    sizes = [img['file_size_bytes'] for img in results['compatible'] if img['file_size_bytes']]
                    
                    if widths and heights:
                        summary_lines.append(f"  Dimension range: {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")
                    if sizes:
                        summary_lines.append(f"  File size range: {min(sizes)//1024}KB to {max(sizes)//1024}KB")
                
                if results['incompatible']:
                    summary_lines.append(f"  Incompatible: {incompatible_count} images")
                    # Show most common error types
                    error_types = {}
                    for img in results['incompatible']:
                        if img.get('error') and img['error'].get('type'):
                            error_type = img['error']['type']
                            error_types[error_type] = error_types.get(error_type, 0) + 1
                    
                    if error_types:
                        summary_lines.append("  Common errors:")
                        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                            summary_lines.append(f"    {error_type}: {count} times")
            else:
                summary_lines.append(f"\n{model}: No images tested")
        
        return "\n".join(summary_lines)


def main():
    parser = argparse.ArgumentParser(description='Test image compatibility with Ollama vision models')
    parser.add_argument('--models', nargs='+', help='Models to test')
    parser.add_argument('--data_file', default='data/label-studio-data-min.json',
                        help='JSON file containing image data')
    parser.add_argument('--output_dir', default='src/generators/baselines/results',
                        help='Directory to save results')
    parser.add_argument('--limit', type=int, help='Limit number of images to test')
    
    args = parser.parse_args()
    
    # Set default values
    if not args.models:
        args.models = ['qwen2.5vl:3b', 'gemma3:4b-it-q4_K_M',
                       'qwen2.5vl:7b', 'gemma3:12b-it-q4_K_M']
    if not args.limit:
        args.limit = None
    
    # Check Ollama connection
    try:
        ollama.list()
        print("✓ Ollama connected")
    except Exception as e:
        print(f"✗ Ollama error: {e}")
        return
    
    # Run compatibility test
    tester = ImageCompatibilityTester(args.data_file)
    tester.test_image_compatibility(args.models, args.limit, args.output_dir)


if __name__ == "__main__":
    main()
