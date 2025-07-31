#!/usr/bin/env python3
import json
import time
import argparse
import os
from pathlib import Path
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
    def evaluate_existing_answers(self, judge, reference_data_file, results_dir, output_dir="src/generators/results", limit=None, models=None, evaluate_baselines=False):
        print(f"Judge model: {judge.judge_model if not judge.use_openai else judge.openai_model}")

        if not evaluate_baselines:
            with open(reference_data_file, 'r', encoding='utf-8') as f:
                reference_data = json.load(f)
            reference_map = {}
            q_counter = 1
            for entry in reference_data:
                for qa in entry.get('qa_pairs', []):
                    q_key = f"q{q_counter}"
                    reference_map[q_key] = qa.get('formatted_answer', qa.get('original_answer', ''))
                    q_counter += 1
            print(f"Loaded {len(reference_data)} reference entries")
            print(f"Created reference mapping for {len(reference_map)} questions")
        else:
            reference_map = {}
            print("Baseline evaluation mode: using ground_truth from files")

        norm_models = None
        if models:
            norm_models = [m.replace("/", "-").replace(".", "-").replace("_", "-").replace(":", "-") for m in models]

        result_files = []
        for root, dirs, files in os.walk(results_dir):
            for file in files:
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
                if evaluate_baselines:
                    reference_answer = result.get('ground_truth')
                    predicted = result.get('predicted')
                    question = result.get('question')
                    if result.get('success') and predicted:
                        print(f"  Evaluating baseline: {question[:30]}...")
                        judgment = judge.evaluate_answer(question, reference_answer, predicted)
                        evaluated_results.append({
                            **result,
                            'reference_answer': reference_answer,
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
                            'judgment_error': 'Generation failed'
                        })
                else:
                    q_id = result['q_id']
                    question = result['question']
                    predicted = result['predicted']
                    reference_answer = reference_map.get(q_id)
                    if not reference_answer:
                        print(f"  No reference answer for {q_id}, skipping")
                        continue
                    if result['success'] and predicted:
                        print(f"  Evaluating {q_id}: {question[:30]}...")
                        judgment = judge.evaluate_answer(question, reference_answer, predicted)
                        evaluated_results.append({
                            **result,
                            'reference_answer': reference_answer,
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
                            'judgment_error': 'Generation failed'
                        })

            output_file = os.path.join(output_dir, f'{model_name}.json')
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(evaluated_results, f, indent=2, ensure_ascii=False)
            print(f"  Saved: {output_file}")

            retriever_name = os.path.basename(os.path.dirname(result_file))
            key = f"{retriever_name}/{model_name}"
            all_evaluations[key] = evaluated_results

        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        for key, results in all_evaluations.items():
            retriever, model = key.split('/')
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
    parser.add_argument('--top_k', default='1', help='Number of top-k pages to provide to each model')
    parser.add_argument('--reference_data_file', default='data/label-studio-data-min.json', help='File with reference answers')
    parser.add_argument('--output_dir', default='src/generators/results')
    parser.add_argument('--limit', type=int, help='Limit entries')
    parser.add_argument('--judge_model', default='qwen2.5:14b', help='Judge model for evaluation')
    parser.add_argument('--use_openai_judge', action='store_true', help='Use OpenAI model as judge')
    parser.add_argument('--openai_judge_model', default='gpt-4', help='OpenAI judge model')
    args = parser.parse_args()

    if not args.models:
        args.models = ['gemma3:4b-it-q4_K_M', 'qwen2.5vl:3b',
                       'qwen2.5vl:7b', 'gemma3:12b-it-q4_K_M',
                       'OpenGVLab/InternVL3-8B-hf', 'OpenGVLab/InternVL3-2B-hf']
    if not args.retrievers:
        args.retrievers = ['nomic-ai/colnomic-embed-multimodal-3b', 'nomic-ai/colnomic-embed-multimodal-7b']
    if not args.limit:
        args.limit = None
    args.top_k = 3
    args.evaluate_baselines = True

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
        results_dir = 'src/generators/baselines/results'
        output_dir = os.path.join(args.output_dir, 'judged', 'baselines')
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        for model in args.models:
            # Normalize model name to match file naming
            norm_model = model.replace("/", "-").replace(".", "-").replace("_", "-").replace(":", "-")
            # Adjust to locate files with prefixes like 'hf_baselines_' or 'ollama_baselines_'
            possible_prefixes = ["hf_baselines_", "ollama_baselines_"]
            result_file = None
            for prefix in possible_prefixes:
                candidate_file = os.path.join(results_dir, f'{prefix}{norm_model}.json')
                if os.path.exists(candidate_file):
                    result_file = candidate_file
                    break

            if not result_file:
                print(f"Baseline result file not found for model: {model}")
                continue
            print(f"\n=== Evaluating baseline for model: {model} ===")
            # Evaluate and save output in output_dir/norm_model.json
            evaluator.evaluate_existing_answers(
                judge,
                args.reference_data_file,
                results_dir,
                output_dir,
                args.limit,
                [model],
                args.evaluate_baselines
            )
    else:
        # Standard evaluation: iterate over retrievers
        for retriever in args.retrievers:
            print(f"\n=== Evaluating for retriever: {retriever} ===")
            results_dir = os.path.join(args.output_dir, f'top_k_{args.top_k}', retriever.replace("/", "_"))
            output_subdir = os.path.join(args.output_dir, 'judged', f'top_k_{args.top_k}', retriever.replace("/", "_"))
            os.makedirs(results_dir, exist_ok=True)
            if not os.path.exists(results_dir):
                print(f"Results directory not found: {results_dir}")
                continue
            evaluator.evaluate_existing_answers(
                judge,
                args.reference_data_file,
                results_dir,
                output_subdir,
                args.limit,
                args.models,
                args.evaluate_baselines
            )

if __name__ == "__main__":
    main()
