#!/usr/bin/env python3
import json
import time
import argparse
import os
import ollama
import openai


SYSTEM_PROMPT = """
You are an expert VQA grader.
Given a question, reference answer, and proposed answer, decide if the proposed
answer is fully correct.
Respond ONLY with the single word "yes" (fully correct) or "no" (anything
less than fully correct). No other text.
""".strip()


class LLMJudge:
    def __init__(self, judge_model="qwen2.5:14b", use_openai=False, openai_model="gpt-4"):
        self.judge_model = judge_model
        self.use_openai = use_openai
        self.openai_model = openai_model
        
    def evaluate_answer(self, question, reference_answer, predicted_answer):
        start_time = time.time()
        prompt = f"""Question: {question}

Reference Answer: {reference_answer}

Proposed Answer: {predicted_answer}"""

        try:
            if self.use_openai:
                response = openai.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    max_tokens=10
                )
                judgment = response.choices[0].message.content.strip().lower()
            else:
                response = ollama.chat(
                    model=self.judge_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    options={"temperature": 0.0}
                )
                judgment = response['message']['content'].strip().lower()

            is_correct = judgment == "yes"
            return {
                'success': True,
                'is_correct': is_correct,
                'judgment': judgment,
                'time': time.time() - start_time
            }
        except Exception as e:
            return {
                'success': False,
                'is_correct': None,
                'judgment': None,
                'time': time.time() - start_time,
                'error': str(e)
            }


class ResultsEvaluator:
    def evaluate_existing_answers(self, judge, results_dir, output_dir="src/generators/eval", limit=None, models=None):
        print(f"Judge model: {judge.judge_model if not judge.use_openai else judge.openai_model}")

        norm_models = None
        if models:
            norm_models = [m.replace("/", "-").replace(".", "-").replace("_", "-").replace(":", "-") for m in models]

        result_files = []
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                if file.endswith('.json'):
                    result_files.append(os.path.join(root, file))
        print(f"Found {len(result_files)} result files to evaluate in {results_dir}")

        os.makedirs(output_dir, exist_ok=True)
        all_evaluations = {}

        for result_file in result_files:
            filename = os.path.basename(result_file)
            model_name = filename.split("_")[-1].replace('.json', '')
            if norm_models and model_name not in norm_models:
                continue
            print(f"Processing model: {model_name}")

            print(f"\nEvaluating: {result_file}")
            with open(result_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)

            if limit:
                existing_results = existing_results[:limit]

            evaluated_results = []
            for result in existing_results:
                reference_answer = result.get('ground_truth')
                predicted = result.get('predicted')
                question = result.get('question')
                
                if result.get('success') and predicted and reference_answer:
                    print(f"  Evaluating: {question[:30]}...")
                    judgment = judge.evaluate_answer(question, reference_answer, predicted)
                    evaluated_results.append({
                        **result,
                        'judgment_success': judgment['success'],
                        'is_correct': judgment['is_correct'],
                        'judgment': judgment['judgment'],
                        'judgment_time': judgment['time'],
                        'judgment_error': judgment.get('error')
                    })
                else:
                    evaluated_results.append({
                        **result,
                        'reference_answer': reference_answer,
                        'judgment_success': False,
                        'is_correct': None,
                        'judgment': None,
                        'judgment_time': None,
                        'judgment_error': 'Generation failed or missing ground truth'
                    })

            output_file = os.path.join(output_dir, f'{model_name}.json')
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(evaluated_results, f, indent=2, ensure_ascii=False)
            print(f"  Saved: {output_file}")

            # Create a more meaningful key for evaluation summary
            retriever_name = os.path.basename(os.path.dirname(result_file))
            # Handle baseline case where retriever_name would be "results"
            if retriever_name == "results":
                retriever_name = "baselines"
            key = f"{retriever_name}/{model_name}"
            all_evaluations[key] = evaluated_results

        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        for key, results in all_evaluations.items():
            if '/' in key:
                retriever, model = key.split('/', 1)  # Use maxsplit=1 to handle model names with slashes
            else:
                retriever, model = "unknown", key
            total = len(results)
            generation_successful = sum(1 for r in results if r['success'])
            judgment_successful = sum(1 for r in results if r['judgment_success'])
            correct_answers = sum(1 for r in results if r['is_correct'] is True)
            accuracy = correct_answers / judgment_successful if judgment_successful > 0 else 0
            print(f"\n{retriever} / {model}:")
            print(f"  Total entries: {total}")
            print(f"  Generation successful: {generation_successful}")
            print(f"  Judgment successful: {judgment_successful}")
            print(f"  Correct answers: {correct_answers}")
            print(f"  Accuracy: {accuracy:.3f}")

        return all_evaluations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', help='Models to test')
    parser.add_argument('--evaluate_baselines', action='store_true', help='Evaluate baseline models')
    parser.add_argument('--retrievers', nargs='+', help='Retrievers to test')
    parser.add_argument('--top_k', nargs='+', type=int, help='List of top-k values to evaluate (e.g. --top_k 1 3 5)')
    parser.add_argument('--output_dir', default='src/generators/eval')
    parser.add_argument('--is_test', action='store_true', help='Use test results and output to test directory')
    parser.add_argument('--baseline_results_dir', default='src/generators/results/baselines', help='Directory containing baseline results')
    parser.add_argument('--pipeline_results_dir', default='src/generators/results/retrieval_pipeline', help='Directory containing end-to-end retrieval + generation pipeline results')
    parser.add_argument('--limit', type=int, help='Limit entries')
    parser.add_argument('--judge_model', default='qwen2.5:14b', help='Judge model for evaluation')
    parser.add_argument('--use_openai_judge', action='store_true', help='Use OpenAI model as judge')
    parser.add_argument('--openai_judge_model', default='gpt-4', help='OpenAI judge model')
    args = parser.parse_args()

    args.is_test = True  # Set to True for testing purposes
    if not args.models:
        # args.models = ['gemma3:4b-it-q4_K_M', 'qwen2.5vl:3b',
        #                'qwen2.5vl:7b', 'gemma3:12b-it-q4_K_M',
        #                'OpenGVLab/InternVL3-8B-hf', 'OpenGVLab/InternVL3-2B-hf']
        args.models = ['Qwen/Qwen2.5-VL-7B-Instruct']
    if not args.retrievers:
        args.retrievers = ['nomic-ai/colnomic-embed-multimodal-3b', 'nomic-ai/colnomic-embed-multimodal-7b']
    if not args.limit:
        args.limit = None
    if not args.top_k:
        args.top_k = [1, 3]

    # Adjust paths based on is_test flag
    if args.is_test:
        args.output_dir = os.path.join(args.output_dir, 'test')
        args.baseline_results_dir = 'src/generators/results/test/baselines'
        args.pipeline_results_dir = 'src/generators/results/test/retrieval_pipeline'

    try:
        ollama.list()
        print("✓ Ollama connected")
    except Exception as e:
        print(f"✗ Ollama error: {e}")
        return

    if args.use_openai_judge:
        try:
            openai.models.list()
            print("✓ OpenAI connected")
        except Exception as e:
            print(f"✗ OpenAI error: {e}")
            return

    judge = LLMJudge(
        judge_model=args.judge_model,
        use_openai=args.use_openai_judge,
        openai_model=args.openai_judge_model
    )
    evaluator = ResultsEvaluator()

    if args.evaluate_baselines:
        # Baseline: evaluate each VLM (model) separately, ignore retrievers
        results_dir = args.baseline_results_dir
        output_dir = os.path.join(args.output_dir, 'baselines')
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n=== Evaluating baselines ===")
        evaluator.evaluate_existing_answers(
            judge,
            results_dir,
            output_dir,
            args.limit,
            args.models
        )
    else:
        # Standard evaluation: iterate over top_k values and retrievers
        for top_k in args.top_k:
            print(f"\n=== Evaluating for top_k: {top_k} ===")
            for retriever in args.retrievers:
                print(f"\n=== Evaluating for retriever: {retriever} ===")
                results_dir = os.path.join(args.pipeline_results_dir, f'top_k_{top_k}', retriever.replace("/", "_"))
                output_subdir = os.path.join(args.output_dir, f'top_k_{top_k}', retriever.replace("/", "_"))
                
                if not os.path.exists(results_dir):
                    print(f"Results directory not found: {results_dir}")
                    continue
                    
                evaluator.evaluate_existing_answers(
                    judge,
                    results_dir,
                    output_subdir,
                    args.limit,
                    args.models
                )

if __name__ == "__main__":
    main()
