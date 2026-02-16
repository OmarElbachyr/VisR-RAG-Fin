#!/usr/bin/env python3
import json
import os
import argparse
from pathlib import Path


def load_judged_results(results_dir):
    """Load all judged results from directory"""
    results = {}
    
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                path_parts = file_path.split(os.sep)
                # Expect: .../judged/top_k_X/retriever/model.json
                if len(path_parts) < 4:
                    continue
                retriever = path_parts[-2]
                top_k_dir = path_parts[-3]
                model = file.replace('.json', '')
                top_k = top_k_dir.replace('top_k_', '') if 'top_k_' in top_k_dir else top_k_dir
                with open(file_path, 'r') as f:
                    data = json.load(f)
                key = f"{top_k}/{retriever}/{model}"
                results[key] = data
    
    return results


def compute_metrics(results):
    """Compute accuracy and other metrics for each model/retriever combination"""
    metrics = {}
    
    for key, data in results.items():
        top_k, retriever, model = key.split('/')
        
        total = len(data)
        generation_successful = sum(1 for r in data if r['success'])
        judgment_successful = sum(1 for r in data if r['judgment_success'])
        correct_answers = sum(1 for r in data if r['is_correct'] is True)
        incorrect_answers = sum(1 for r in data if r['is_correct'] is False)
        
        # Calculate metrics
        generation_rate = generation_successful / total if total > 0 else 0
        judgment_rate = judgment_successful / total if total > 0 else 0
        accuracy = correct_answers / judgment_successful if judgment_successful > 0 else 0
        
        # Average judgment time
        judgment_times = [r['judgment_time'] for r in data if r['judgment_time'] is not None]
        avg_judgment_time = sum(judgment_times) / len(judgment_times) if judgment_times else 0
        
        metrics[key] = {
            'top_k': top_k,
            'retriever': retriever,
            'model': model,
            'total': total,
            'generation_successful': generation_successful,
            'judgment_successful': judgment_successful,
            'correct_answers': correct_answers,
            'incorrect_answers': incorrect_answers,
            'generation_rate': generation_rate,
            'judgment_rate': judgment_rate,
            'accuracy': accuracy,
            'avg_judgment_time': avg_judgment_time
        }
    
    return metrics


def print_summary_table(metrics):
    """Print a summary table of all metrics"""
    print(f"\n{'='*130}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*130}")
    
    # Header
    print(f"{'Top-K':<6} {'Retriever':<35} {'Model':<25} {'Total':<6} {'Gen%':<6} {'Acc%':<6} {'Correct':<8} {'Time(s)':<8}")
    print(f"{'-'*130}")
    
    # Sort by top_k, then by accuracy descending
    sorted_metrics = sorted(metrics.items(), key=lambda x: (x[1]['top_k'], -x[1]['accuracy']))
    
    for key, m in sorted_metrics:
        print(f"{m['top_k']:<6} {m['retriever']:<35} {m['model']:<25} {m['total']:<6} "
              f"{m['generation_rate']*100:<5.1f}% {m['accuracy']*100:<5.1f}% "
              f"{m['correct_answers']:<8} {m['avg_judgment_time']:<7.2f}")


def print_topk_comparison(metrics):
    """Compare different top_k values across models and retrievers"""
    print(f"\n{'='*80}")
    print("TOP-K COMPARISON (Average across models and retrievers)")
    print(f"{'='*80}")
    
    # Group by top_k
    topk_stats = {}
    for key, m in metrics.items():
        top_k = m['top_k']
        if top_k not in topk_stats:
            topk_stats[top_k] = []
        topk_stats[top_k].append(m)
    
    # Calculate averages
    topk_averages = {}
    for top_k, stats_list in topk_stats.items():
        total_entries = sum(s['total'] for s in stats_list)
        total_correct = sum(s['correct_answers'] for s in stats_list)
        total_judged = sum(s['judgment_successful'] for s in stats_list)
        
        avg_accuracy = total_correct / total_judged if total_judged > 0 else 0
        avg_gen_rate = sum(s['generation_rate'] for s in stats_list) / len(stats_list)
        avg_time = sum(s['avg_judgment_time'] for s in stats_list) / len(stats_list)
        
        topk_averages[top_k] = {
            'accuracy': avg_accuracy,
            'generation_rate': avg_gen_rate,
            'total_entries': total_entries,
            'total_correct': total_correct,
            'avg_time': avg_time,
            'num_combinations': len(stats_list)
        }
    
    # Print sorted by top_k value
    print(f"{'Top-K':<8} {'Combinations':<12} {'Avg Acc%':<10} {'Gen Rate%':<10} {'Total Correct':<13} {'Avg Time(s)':<12}")
    print(f"{'-'*80}")
    
    sorted_topks = sorted(topk_averages.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999)
    for top_k, stats in sorted_topks:
        print(f"{top_k:<8} {stats['num_combinations']:<12} {stats['accuracy']*100:<9.1f}% {stats['generation_rate']*100:<9.1f}% "
              f"{stats['total_correct']:<13} {stats['avg_time']:<11.2f}")


def print_model_comparison(metrics):
    """Compare models across retrievers"""
    print(f"\n{'='*80}")
    print("MODEL COMPARISON (Average across retrievers)")
    print(f"{'='*80}")
    
    # Group by model
    model_stats = {}
    for key, m in metrics.items():
        model = m['model']
        if model not in model_stats:
            model_stats[model] = []
        model_stats[model].append(m)
    
    # Calculate averages
    model_averages = {}
    for model, stats_list in model_stats.items():
        total_entries = sum(s['total'] for s in stats_list)
        total_correct = sum(s['correct_answers'] for s in stats_list)
        total_judged = sum(s['judgment_successful'] for s in stats_list)
        
        avg_accuracy = total_correct / total_judged if total_judged > 0 else 0
        avg_gen_rate = sum(s['generation_rate'] for s in stats_list) / len(stats_list)
        avg_time = sum(s['avg_judgment_time'] for s in stats_list) / len(stats_list)
        
        model_averages[model] = {
            'accuracy': avg_accuracy,
            'generation_rate': avg_gen_rate,
            'total_entries': total_entries,
            'total_correct': total_correct,
            'avg_time': avg_time
        }
    
    # Print sorted by accuracy
    print(f"{'Model':<25} {'Avg Acc%':<10} {'Gen Rate%':<10} {'Total Correct':<13} {'Avg Time(s)':<12}")
    print(f"{'-'*80}")
    
    sorted_models = sorted(model_averages.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for model, stats in sorted_models:
        print(f"{model:<25} {stats['accuracy']*100:<9.1f}% {stats['generation_rate']*100:<9.1f}% "
              f"{stats['total_correct']:<13} {stats['avg_time']:<11.2f}")


def print_retriever_comparison(metrics):
    """Compare retrievers across models"""
    print(f"\n{'='*80}")
    print("RETRIEVER COMPARISON (Average across models)")
    print(f"{'='*80}")
    
    # Group by retriever
    retriever_stats = {}
    for key, m in metrics.items():
        retriever = m['retriever']
        if retriever not in retriever_stats:
            retriever_stats[retriever] = []
        retriever_stats[retriever].append(m)
    
    # Calculate averages
    retriever_averages = {}
    for retriever, stats_list in retriever_stats.items():
        total_entries = sum(s['total'] for s in stats_list)
        total_correct = sum(s['correct_answers'] for s in stats_list)
        total_judged = sum(s['judgment_successful'] for s in stats_list)
        
        avg_accuracy = total_correct / total_judged if total_judged > 0 else 0
        avg_gen_rate = sum(s['generation_rate'] for s in stats_list) / len(stats_list)
        avg_time = sum(s['avg_judgment_time'] for s in stats_list) / len(stats_list)
        
        retriever_averages[retriever] = {
            'accuracy': avg_accuracy,
            'generation_rate': avg_gen_rate,
            'total_entries': total_entries,
            'total_correct': total_correct,
            'avg_time': avg_time
        }
    
    # Print sorted by accuracy
    print(f"{'Retriever':<40} {'Avg Acc%':<10} {'Gen Rate%':<10} {'Total Correct':<13} {'Avg Time(s)':<12}")
    print(f"{'-'*80}")
    
    sorted_retrievers = sorted(retriever_averages.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for retriever, stats in sorted_retrievers:
        print(f"{retriever:<40} {stats['accuracy']*100:<9.1f}% {stats['generation_rate']*100:<9.1f}% "
              f"{stats['total_correct']:<13} {stats['avg_time']:<11.2f}")


def save_metrics_json(metrics, output_file):
    """Save metrics to JSON file"""
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {output_file}")


def main():
    results_dir = 'src/results/judged' # Directory containing judged results
    output_file = 'src/results/generation_metrics.json' # Output file for metrics JSON
    no_save = False # Don't save metrics to file

    # Load results
    print(f"Loading results from: {results_dir}")
    results = load_judged_results(results_dir)
    
    if not results:
        print("No judged results found!")
        return
    
    print(f"Found {len(results)} result sets")
    
    # Compute metrics
    metrics = compute_metrics(results)
    
    # Print comparisons
    print_summary_table(metrics)
    print_topk_comparison(metrics)
    print_model_comparison(metrics)
    print_retriever_comparison(metrics)
    
    # Save metrics
    if not no_save:
        save_metrics_json(metrics, output_file)


if __name__ == "__main__":
    main()
