#!/usr/bin/env python3
"""
VLM Annotation Script for Page Evaluation

This script uses different Ollama VLMs to annotate pages from the data/pages folder.
Each VLM evaluates pages based on question-answer pairs and returns relevancy and correctness scores.
"""

import json
import base64
import requests
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union
import pymupdf
from PIL import Image
import io
import time
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class QAPair:
    """Question-Answer pair with optional hint and ground truth labels"""
    question: str
    answer: str
    hint: Optional[str] = None
    ground_truth_relevancy: Optional[int] = None  # 0 or 1
    ground_truth_correctness: Optional[int] = None  # 0 or 1


@dataclass
class AnnotationResult:
    """Result of VLM annotation"""
    page_id: str
    model_name: str
    qa_pair: QAPair
    relevancy: int  # Ground truth relevancy (0 or 1)
    correctness: int  # Ground truth correctness (0 or 1)
    predicted_relevancy: int  # VLM predicted relevancy (0 or 1)
    predicted_correctness: int  # VLM predicted correctness (0 or 1)
    confidence: int  # VLM confidence score (0-100)
    reasoning: Optional[str] = None
    processing_time: float = 0.0


class OllamaVLMClient:
    """Client for interacting with Ollama VLM models"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def list_models(self) -> List[str]:
        """List available Ollama models"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get('models', [])
            return [model['name'] for model in models]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def list_all_models(self) -> List[str]:
        """List all available Ollama models (alias for list_models)"""
        return self.list_models()
    
    def encode_image(self, image_path: Path) -> str:
        """Encode image to base64"""
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def query_vlm(self, model_name: str, prompt: str, image_path: Path, timeout: int = 120) -> Dict:
        """Query VLM with image and prompt"""
        try:
            image_data = self.encode_image(image_path)
            
            payload = {
                "model": model_name,
                "prompt": prompt,
                "images": [image_data],
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Low temperature for consistent results
                    "top_p": 0.9
                }
            }
            
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=timeout
            )
            processing_time = time.time() - start_time
            
            response.raise_for_status()
            result = response.json()
            result['processing_time'] = processing_time
            return result
            
        except Exception as e:
            logger.error(f"Failed to query {model_name}: {e}")
            return {"error": str(e), "processing_time": 0.0}


class PDFTextExtractor:
    """Extract text content from PDF pages"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: Path, page_number: int) -> str:
        """Extract text from a specific PDF page"""
        try:
            doc = pymupdf.open(pdf_path)
            if page_number >= len(doc):
                return ""
            
            page = doc[page_number]
            text = page.get_text()
            doc.close()
            return text.strip()
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}, page {page_number}: {e}")
            return ""


class VLMAnnotator:
    """Main annotation class using VLM models"""
    
    def __init__(self, ollama_client: OllamaVLMClient, pdf_dir: Optional[Path] = None):
        self.client = ollama_client
        self.pdf_dir = pdf_dir
        self.text_extractor = PDFTextExtractor()
    
    def extract_page_text(self, page_id: str) -> Optional[str]:
        """Extract text from specific PDF page based on page_id pattern"""
        if not self.pdf_dir or not self.pdf_dir.exists():
            return None
        
        try:
            # Parse page_id pattern: hash_pagenumber (e.g., 1c13544248fda6951a76950a332e334a_10)
            if '_' not in page_id:
                return None
            
            hash_part, page_num_str = page_id.rsplit('_', 1)
            page_number = int(page_num_str) - 1  # Convert to 0-based index
            
            # Look for PDF file with matching hash
            pdf_file = self.pdf_dir / f"{hash_part}.pdf"
            if not pdf_file.exists():
                logger.warning(f"PDF file not found: {pdf_file}")
                return None
            
            # Extract text from specific page
            text = self.text_extractor.extract_text_from_pdf(pdf_file, page_number)
            logger.info(f"Extracted {len(text)} characters from {pdf_file.name} page {page_number + 1}")
            return text
            
        except Exception as e:
            logger.error(f"Failed to extract text for page {page_id}: {e}")
            return None
    
    def create_evaluation_prompt(self, qa_pair: QAPair, page_text: Optional[str] = None) -> str:
        """Create evaluation prompt for VLM"""
        prompt = f'''You are an expert document analyst. Your task is to evaluate whether a page is relevant to a question and whether the provided answer is correct.

**Question:** {qa_pair.question}
**Provided Answer:** {qa_pair.answer}'''
        
        if qa_pair.hint:
            prompt += f'''
**Hint:** {qa_pair.hint}'''
        
        if page_text:
            prompt += f'''
**Page Text Content:** {page_text[:2000]}...''' if len(page_text) > 2000 else f'''
**Page Text Content:** {page_text}'''
        
        prompt += '''

Please analyze the page image and evaluate:

1. **RELEVANCY**: Does this page contain information that is relevant to answering the question? 
   - 1 = Page contains relevant information to answer the question
   - 0 = Page does not contain relevant information

2. **CORRECTNESS**: If the page is relevant, is the provided answer correct based on the visual and textual information?
   - 1 = The provided answer is correct
   - 0 = The provided answer is incorrect or cannot be verified

3. **CONFIDENCE**: How confident are you in your evaluation (0-100)?
   - 0-20 = Very low confidence, uncertain about the evaluation
   - 21-40 = Low confidence, some uncertainty
   - 41-60 = Moderate confidence
   - 61-80 = High confidence
   - 81-100 = Very high confidence, very certain about the evaluation

**Important Instructions:**
- Look carefully at all visual elements: text, diagrams, charts, tables, images
- Consider both the visual content and any provided text extraction
- Be strict in your evaluation - only mark as relevant if the page genuinely contains information to answer the question
- Only mark as correct if you can verify the answer from the page content
- Provide a confidence score that reflects how certain you are about your relevancy and correctness judgments

Respond with a valid JSON object in this exact format:
{
    "relevancy": 0 or 1,
    "correctness": 0 or 1,
    "confidence": 0-100,
    "reasoning": "Brief explanation of your evaluation"
}'''
        
        return prompt
    
    def parse_vlm_response(self, response_text: str) -> Dict:
        """Parse VLM response and extract JSON"""
        try:
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{[^}]*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                # Validate required fields
                if 'relevancy' in result and 'correctness' in result:
                    return {
                        'relevancy': int(result['relevancy']),
                        'correctness': int(result['correctness']),
                        'confidence': int(result.get('confidence', 50)),  # Default to 50 if not provided
                        'reasoning': result.get('reasoning', '')
                    }
            
            # Fallback parsing
            relevancy = 1 if any(word in response_text.lower() for word in ['relevant', 'yes', 'contains']) else 0
            correctness = 1 if any(word in response_text.lower() for word in ['correct', 'accurate', 'right']) else 0
            
            return {
                'relevancy': relevancy,
                'correctness': correctness,
                'reasoning': 'Fallback parsing used'
            }
            
        except Exception as e:
            logger.error(f"Failed to parse VLM response: {e}")
            return {
                'relevancy': 0,
                'correctness': 0,
                'reasoning': f'Parsing failed: {str(e)}'
            }
    
    def annotate_page(self, model_name: str, page_path: Path, qa_pair: QAPair, 
                     page_text: Optional[str] = None) -> AnnotationResult:
        """Annotate a single page with a VLM model"""
        
        page_id = page_path.stem
        start_time = time.time()
        
        try:
            # Create evaluation prompt
            prompt = self.create_evaluation_prompt(qa_pair, page_text)
            
            # Query VLM
            response = self.client.query_vlm(model_name, prompt, page_path)
            processing_time = time.time() - start_time
            
            if 'error' in response:
                return AnnotationResult(
                    page_id=page_id,
                    model_name=model_name,
                    qa_pair=qa_pair,
                    relevancy=qa_pair.ground_truth_relevancy or 0,
                    correctness=qa_pair.ground_truth_correctness or 0,
                    predicted_relevancy=0,
                    predicted_correctness=0,
                    confidence=0,  # Low confidence for error cases
                    reasoning=f"Error: {response['error']}",
                    processing_time=processing_time
                )
            
            # Parse response
            parsed = self.parse_vlm_response(response.get('response', ''))
            
            return AnnotationResult(
                page_id=page_id,
                model_name=model_name,
                qa_pair=qa_pair,
                relevancy=qa_pair.ground_truth_relevancy or 0,
                correctness=qa_pair.ground_truth_correctness or 0,
                predicted_relevancy=parsed['relevancy'],
                predicted_correctness=parsed['correctness'],
                confidence=parsed['confidence'],
                reasoning=parsed['reasoning'],
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Failed to annotate {page_id} with {model_name}: {e}")
            return AnnotationResult(
                page_id=page_id,
                model_name=model_name,
                qa_pair=qa_pair,
                relevancy=qa_pair.ground_truth_relevancy or 0,
                correctness=qa_pair.ground_truth_correctness or 0,
                predicted_relevancy=0,
                predicted_correctness=0,
                confidence=0,  # Low confidence for exception cases
                reasoning=f"Exception: {str(e)}",
                processing_time=processing_time
            )


def load_qa_data(data_file: Path) -> Dict[str, List[QAPair]]:
    """Load QA pairs from label studio data with ground truth labels"""
    qa_data = {}
    
    def convert_label_to_binary(label_str: str) -> int:
        """Convert string labels to binary"""
        if isinstance(label_str, str):
            return 1 if label_str.lower() in ['relevant', 'correct'] else 0
        return 0
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            page_filename = Path(item['image_filename']).name
            page_id = page_filename.replace('.png', '')
            
            qa_pairs = []
            for i, qa in enumerate(item.get('qa_pairs', []), 1):
                # Extract ground truth labels for this QA pair
                relevancy_key = f'relevancy{i}'
                correctness_key = f'correct{i}'
                
                ground_truth_relevancy = convert_label_to_binary(item.get(relevancy_key))
                ground_truth_correctness = convert_label_to_binary(item.get(correctness_key))
                
                qa_pairs.append(QAPair(
                    question=qa['question'],
                    answer=qa['formatted_answer'],
                    hint=qa.get('hint'),
                    ground_truth_relevancy=ground_truth_relevancy,
                    ground_truth_correctness=ground_truth_correctness
                ))
            
            if qa_pairs:
                qa_data[page_id] = qa_pairs
                
    except Exception as e:
        logger.error(f"Failed to load QA data: {e}")
    
    return qa_data


def main():
    parser = argparse.ArgumentParser(description='VLM Page Annotation Script')
    parser.add_argument('--pages_dir', type=Path, default='data/pages', 
                       help='Directory containing page images')
    parser.add_argument('--qa_data', type=Path, default='data/label-studio-data-min.json',
                       help='JSON file with QA pairs')
    parser.add_argument('--pdf_dir', type=Path, help='Directory containing PDF files for text extraction')
    parser.add_argument('--models', nargs='+', default=None,
                       help='List of Ollama VLM models to use')
    parser.add_argument('--output', type=Path, default='auto_annotate/results/vlm_annotations.json',
                       help='Output file for annotations')
    parser.add_argument('--max_workers', type=int, default=2,
                       help='Maximum number of parallel workers')
    parser.add_argument('--max_pages', type=int, help='Maximum number of pages to process')
    parser.add_argument('--parallel_models', action='store_true',
                       help='Process all models in parallel (may cause CUDA OOM with large models)')
    parser.add_argument('--ollama_url', default='http://localhost:11434',
                       help='Ollama server URL')
    
    args = parser.parse_args()
    
    # Setup
    pages_dir = Path(args.pages_dir)
    qa_data_file = Path(args.qa_data)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize client
    client = OllamaVLMClient(args.ollama_url)
    annotator = VLMAnnotator(client, args.pdf_dir)
    
    # Handle empty or None models argument
    if not args.models:
        vlms = ['gemma3:12b', 'gemma3:27b']
        logger.info(f"No models specified in --models, using default models: {vlms}")
        args.models = vlms

    # Max pages limit
    args.max_pages = 1
    
    models_to_use = args.models
    logger.info(f"Using models: {models_to_use}")
    
    # Load QA data
    logger.info("Loading QA data...")
    qa_data = load_qa_data(qa_data_file)
    logger.info(f"Loaded QA data for {len(qa_data)} pages")
    
    # Apply max_pages limit to annotated pages, not all image files
    if args.max_pages:
        qa_data_keys = list(qa_data.keys())[:args.max_pages]
        qa_data = {k: qa_data[k] for k in qa_data_keys}
        logger.info(f"Limited to first {len(qa_data)} annotated pages (--max_pages {args.max_pages})")
    
    # Get only the page files that have QA data
    page_files = []
    for page_id in qa_data.keys():
        page_file = pages_dir / f"{page_id}.png"
        if page_file.exists():
            page_files.append(page_file)
        else:
            logger.warning(f"Image file not found for page: {page_id}")
    
    logger.info(f"Found {len(page_files)} page images with QA data")
    
    # Create annotation tasks
    tasks = []
    for page_file in page_files:
        page_id = page_file.stem
        if page_id in qa_data:  # This should always be true now
            for qa_pair in qa_data[page_id]:
                for model in models_to_use:
                    # Extract text from specific page if PDF directory provided
                    page_text = annotator.extract_page_text(page_id) if args.pdf_dir else None
                    
                    tasks.append((model, page_file, qa_pair, page_text))
    
    logger.info(f"Created {len(tasks)} annotation tasks")
    
    # Process annotations
    results = []
    
    if args.parallel_models:
        # Process all models in parallel (may cause CUDA OOM)
        logger.info("Processing all models in parallel")
        
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_task = {
                executor.submit(annotator.annotate_page, model, page_file, qa_pair, page_text): (model, page_file, qa_pair)
                for model, page_file, qa_pair, page_text in tasks
            }
            
            for future in as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Completed: {result.page_id} - {result.model_name} - GT_R:{result.relevancy} GT_C:{result.correctness} P_R:{result.predicted_relevancy} P_C:{result.predicted_correctness}")
                except Exception as e:
                    model, page_file, qa_pair = future_to_task[future]
                    logger.error(f"Task failed: {page_file.stem} - {model}: {e}")
            
    else:
        # Default: Process models sequentially to avoid CUDA OOM
        logger.info("Processing models sequentially to avoid CUDA memory issues")
        
        for model in models_to_use:
            logger.info(f"Processing model: {model}")
            
            # Get tasks for this model only
            model_tasks = [(m, pf, qa, pt) for m, pf, qa, pt in tasks if m == model]
            logger.info(f"Processing {len(model_tasks)} tasks for {model}")
            
            # Process this model's tasks in parallel
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                future_to_task = {
                    executor.submit(annotator.annotate_page, model, page_file, qa_pair, page_text): (model, page_file, qa_pair)
                    for model, page_file, qa_pair, page_text in model_tasks
                }
                
                for future in as_completed(future_to_task):
                    try:
                        result = future.result()
                        results.append(result)
                        logger.info(f"Completed: {result.page_id} - {result.model_name} - GT_R:{result.relevancy} GT_C:{result.correctness} P_R:{result.predicted_relevancy} P_C:{result.predicted_correctness}")
                    except Exception as e:
                        model, page_file, qa_pair = future_to_task[future]
                        logger.error(f"Task failed: {page_file.stem} - {model}: {e}")
            
            logger.info(f"Completed all tasks for {model}")
            
            # Force garbage collection between models
            import gc
            gc.collect()
    
    # Save results
    logger.info(f"Saving {len(results)} results to {output_file}")
    results_data = []
    for result in results:
        results_data.append({
            'page_id': result.page_id,
            'model_name': result.model_name,
            'question': result.qa_pair.question,
            'answer': result.qa_pair.answer,
            'hint': result.qa_pair.hint,
            'relevancy': result.relevancy,  # Ground truth
            'correctness': result.correctness,  # Ground truth
            'predicted_relevancy': result.predicted_relevancy,  # VLM prediction
            'predicted_correctness': result.predicted_correctness,  # VLM prediction
            'confidence': result.confidence,  # VLM confidence score (0-100)
            'reasoning': result.reasoning,
            'processing_time': result.processing_time
        })
    
    # Save combined results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    # Save separate results for each model
    output_dir = output_file.parent
    for model in models_to_use:
        model_results = [r for r in results_data if r['model_name'] == model]
        if model_results:
            # Clean model name for filename (replace : and other special chars)
            safe_model_name = model.replace(':', '_').replace('/', '_')
            model_output_file = output_dir / f"vlm_annotations_{safe_model_name}.json"
            
            with open(model_output_file, 'w', encoding='utf-8') as f:
                json.dump(model_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(model_results)} results for {model} to {model_output_file}")
    
    # Print enhanced summary with per-model breakdown
    total_results = len(results)
    if total_results > 0:
        # Overall aggregated statistics
        avg_gt_relevancy = sum(r.relevancy for r in results) / total_results
        avg_gt_correctness = sum(r.correctness for r in results) / total_results
        avg_pred_relevancy = sum(r.predicted_relevancy for r in results) / total_results
        avg_pred_correctness = sum(r.predicted_correctness for r in results) / total_results
        avg_confidence = sum(r.confidence for r in results) / total_results
        relevancy_accuracy = sum(1 for r in results if r.relevancy == r.predicted_relevancy) / total_results
        correctness_accuracy = sum(1 for r in results if r.correctness == r.predicted_correctness) / total_results
        avg_time = sum(r.processing_time for r in results) / total_results
        
        # Per-model statistics
        model_stats = {}
        for model in models_to_use:
            model_results = [r for r in results if r.model_name == model]
            if model_results:
                count = len(model_results)
                model_stats[model] = {
                    'count': count,
                    'relevancy_accuracy': sum(1 for r in model_results if r.relevancy == r.predicted_relevancy) / count,
                    'correctness_accuracy': sum(1 for r in model_results if r.correctness == r.predicted_correctness) / count,
                    'avg_confidence': sum(r.confidence for r in model_results) / count,
                    'avg_time': sum(r.processing_time for r in model_results) / count,
                    'avg_pred_relevancy': sum(r.predicted_relevancy for r in model_results) / count,
                    'avg_pred_correctness': sum(r.predicted_correctness for r in model_results) / count
                }
        
        # Find best performing model
        best_relevancy_model = max(model_stats.keys(), key=lambda m: model_stats[m]['relevancy_accuracy']) if model_stats else None
        best_correctness_model = max(model_stats.keys(), key=lambda m: model_stats[m]['correctness_accuracy']) if model_stats else None
        highest_confidence_model = max(model_stats.keys(), key=lambda m: model_stats[m]['avg_confidence']) if model_stats else None
        fastest_model = min(model_stats.keys(), key=lambda m: model_stats[m]['avg_time']) if model_stats else None
        
        # Generate summary report
        summary = f"""
=== VLM ANNOTATION SUMMARY ===
Total annotations: {total_results} (across {len(models_to_use)} model{'s' if len(models_to_use) > 1 else ''})

OVERALL PERFORMANCE (aggregated across all models):
- Ground Truth Baseline - Relevancy: {avg_gt_relevancy:.2f}, Correctness: {avg_gt_correctness:.2f}
- Predicted Averages - Relevancy: {avg_pred_relevancy:.2f}, Correctness: {avg_pred_correctness:.2f}
- Overall Accuracy - Relevancy: {relevancy_accuracy:.2f}, Correctness: {correctness_accuracy:.2f}
- Average Confidence: {avg_confidence:.1f}/100
- Average Processing Time: {avg_time:.2f}s
"""
        
        if len(models_to_use) > 1:
            summary += "\nPER-MODEL BREAKDOWN:\n"
            for model in models_to_use:
                if model in model_stats:
                    stats = model_stats[model]
                    summary += f"  {model}:\n"
                    summary += f"    - Annotations: {stats['count']}\n"
                    summary += f"    - Accuracy: Relevancy {stats['relevancy_accuracy']:.2f}, Correctness {stats['correctness_accuracy']:.2f}\n"
                    summary += f"    - Avg Confidence: {stats['avg_confidence']:.1f}/100\n"
                    summary += f"    - Avg Time: {stats['avg_time']:.2f}s\n"
            
            summary += f"\nMODEL COMPARISON:\n"
            if best_relevancy_model:
                summary += f"  - Best Relevancy Accuracy: {best_relevancy_model} ({model_stats[best_relevancy_model]['relevancy_accuracy']:.2f})\n"
            if best_correctness_model:
                summary += f"  - Best Correctness Accuracy: {best_correctness_model} ({model_stats[best_correctness_model]['correctness_accuracy']:.2f})\n"
            if highest_confidence_model:
                summary += f"  - Highest Confidence: {highest_confidence_model} ({model_stats[highest_confidence_model]['avg_confidence']:.1f})\n"
            if fastest_model:
                summary += f"  - Fastest Processing: {fastest_model} ({model_stats[fastest_model]['avg_time']:.2f}s avg)\n"
        
        summary += f"\nResults saved to: {output_file}\n"
        summary += "=" * 50
        
        logger.info(summary)
        
        # Save summary statistics as text file
        summary_file = output_dir / "vlm_annotation_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
        logger.info(f"Summary statistics saved to: {summary_file}")


if __name__ == "__main__":
    main()