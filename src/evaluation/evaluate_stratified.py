"""
Evaluate retrieval results stratified by question_type or question_category.

This script loads a chunks CSV file and a JSON file containing retrieval results,
then evaluates the results with standard IR metrics broken down by the specified field.
"""

import json
import argparse
import os
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple

from evaluation.classes.document_provider import DocumentProvider
from evaluation.classes.query_qrel_builder import QueryQrelsBuilder
from evaluation.classes.ir_evaluator import IrMeasuresEvaluator


def load_retrieval_results(json_path: str) -> Dict:
    """Load retrieval results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def stratify_queries_and_qrels(chunks_path: str, stratify_by: str) -> Dict[str, Tuple[Dict, Dict]]:
    """
    Stratify queries and qrels by the specified field.
    
    Args:
        chunks_path: Path to the chunks CSV file
        stratify_by: Field to stratify by ('question_type' or 'question_category')
    
    Returns:
        Dictionary mapping stratum value to (queries, qrels) tuple
    """
    # Load chunks to get question metadata
    df = pd.read_csv(chunks_path)
    
    # Build full queries and qrels
    builder = QueryQrelsBuilder(chunks_path)
    full_queries, full_qrels = builder.build()
    
    # Group by stratification field
    stratified = defaultdict(lambda: {'queries': {}, 'qrels': {}})
    
    # For each unique question_id, find its stratum value
    question_metadata = df[['question_id', stratify_by]].drop_duplicates('question_id')
    
    for _, row in question_metadata.iterrows():
        qid = row['question_id']
        stratum = row[stratify_by]
        
        # Skip if missing data
        if pd.isna(stratum) or qid not in full_queries:
            continue
        
        # Add query and qrels for this question
        stratified[stratum]['queries'][qid] = full_queries[qid]
        if qid in full_qrels:
            stratified[stratum]['qrels'][qid] = full_qrels[qid]
    
    return {stratum: (data['queries'], data['qrels']) for stratum, data in stratified.items()}


def filter_run_by_queries(run: Dict, queries: Dict) -> Dict:
    """Filter a run to only include specified queries."""
    return {qid: scores for qid, scores in run.items() if qid in queries}


def evaluate_metrics(run: Dict, qrels: Dict, k_values: List[int] = [1, 3, 5, 10], verbose: bool = False) -> Dict:
    """
    Evaluate retrieval results using IrMeasuresEvaluator (same as retrievers).
    
    Args:
        run: Dictionary mapping query_id to {doc_id: score}
        qrels: Dictionary mapping query_id to {doc_id: relevance}
        k_values: List of k values for cut-off metrics
        verbose: Whether to print evaluation output
    
    Returns:
        Dictionary with metrics organized by k and global metrics
    """
    evaluator = IrMeasuresEvaluator()
    return evaluator.evaluate(run, qrels, k_values, verbose=verbose)


def format_metrics_table(stratified_results: Dict[str, Dict], overall_results: Dict) -> str:
    """Format metrics as a readable table matching IrMeasuresEvaluator format."""
    lines = []

    # Helper for pretty row
    def pretty_row(k, ndcg, prec, rec):
        return f"K={k:<2}  NDCG:{ndcg:.4f}  P:{prec:.4f}  R:{rec:.4f}"

    # Overall results
    lines.append("")
    lines.append("=== Evaluation Results ===")
    for k in [1, 3, 5, 10]:
        if k in overall_results:
            m = overall_results[k]
            lines.append(pretty_row(k, m['ndcg'], m['precision'], m['recall']))
    if 'global' in overall_results:
        g = overall_results['global']
        lines.append(f"GLOBAL MRR:{g['mrr']:.4f}  Rprec:{g['rprec']:.4f}")

    # Stratified results
    lines.append("")
    lines.append("=== Stratified Results ===")
    for stratum, metrics in sorted(stratified_results.items()):
        lines.append("")
        lines.append(f"[{stratum}]")
        for k in [1, 3, 5, 10]:
            if k in metrics:
                m = metrics[k]
                lines.append(pretty_row(k, m['ndcg'], m['precision'], m['recall']))
        if 'global' in metrics:
            g = metrics['global']
            lines.append(f"GLOBAL MRR:{g['mrr']:.4f}  Rprec:{g['rprec']:.4f}")
    lines.append("")
    return "\n".join(lines)


def main(args):
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.results_json)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate output filenames
    results_basename = os.path.splitext(os.path.basename(args.results_json))[0]
    output_json = os.path.join(
        args.output_dir, 
        f"{results_basename}_stratified_by_{args.stratify_by}.json"
    )
    output_txt = os.path.join(
        args.output_dir,
        f"{results_basename}_stratified_by_{args.stratify_by}.txt"
    )
    
    print(f"\nLoading retrieval results from: {args.results_json}")
    print(f"Loading chunks from: {args.chunks_path}")
    print(f"Stratifying by: {args.stratify_by}")
    
    # Load retrieval results
    retrieval_results = load_retrieval_results(args.results_json)
    
    # Convert retrieval results format (with query text) to simple run format
    run = {}
    for qid, data in retrieval_results.items():
        if isinstance(data, dict) and 'results' in data:
            run[qid] = data['results']
        else:
            run[qid] = data
    
    # Stratify queries and qrels
    print("\nStratifying queries and qrels...")
    stratified_data = stratify_queries_and_qrels(args.chunks_path, args.stratify_by)
    print(f"Found {len(stratified_data)} strata: {list(stratified_data.keys())}")

    # Special handling for evidence: print three tables (text, table, chart/diagram/image)
    if args.stratify_by == "evidence":
        # Build mapping from qid to evidence types (split on ; or , or |)
        import re
        df = pd.read_csv(args.chunks_path)
        qid_to_evidence = dict(df[['question_id', 'evidence']].drop_duplicates('question_id').values)
        def parse_evidence(ev):
            if pd.isna(ev):
                return set()
            return set(re.split(r'[;,|]', str(ev).lower().strip()))
        # Build sets of qids for each group
        text_qids = set()
        table_qids = set()
        visual_qids = set()
        other_qids = set()
        for qid, ev in qid_to_evidence.items():
            ev_set = parse_evidence(ev)
            if 'text' in ev_set:
                text_qids.add(qid)
            if 'table' in ev_set:
                table_qids.add(qid)
            if 'chart' in ev_set or 'diagram' in ev_set or 'image' in ev_set:
                visual_qids.add(qid)
            # Add to other if it doesn't match any of the above
            if not (('text' in ev_set) or ('table' in ev_set) or ('chart' in ev_set or 'diagram' in ev_set or 'image' in ev_set)):
                other_qids.add(qid)
        
        # Merge other_qids into visual_qids
        visual_qids.update(other_qids)
        
        # Get qrels and queries for each group
        builder = QueryQrelsBuilder(args.chunks_path)
        all_queries, all_qrels = builder.build()
        
        # Print unique QAs count at the beginning
        unique_qas = len(all_qrels)
        print(f"Total unique QAs: {unique_qas}\n")
        
        groups = [
            ("Evidence: text", text_qids),
            ("Evidence: table", table_qids),
            ("Evidence: [chart, diagram, image, other]", visual_qids),
        ]
        output_lines = []
        total_qas_in_tables = 0
        
        for group_name, qid_set in groups:
            group_queries = {qid: all_queries[qid] for qid in qid_set if qid in all_queries}
            group_qrels = {qid: all_qrels[qid] for qid in qid_set if qid in all_qrels}
            group_run = filter_run_by_queries(run, group_queries)
            metrics = evaluate_metrics(group_run, group_qrels, verbose=False)
            # Count QAs and unique pages (actual QAs used in evaluation)
            num_qas = len(group_qrels)
            total_qas_in_tables += num_qas
            unique_pages = set()
            for qrels in group_qrels.values():
                unique_pages.update(qrels.keys())
            num_pages = len(unique_pages)
            output_lines.append(f"\n{'='*40}\n{group_name}\nQAs: {num_qas}  Pages: {num_pages}\n{'='*40}")
            output_lines.append(format_metrics_table({}, metrics))
        
        # Add summary at the end
        output_lines.append(f"\n{'='*40}\nSummary\n{'='*40}")
        output_lines.append(f"Total QAs across all tables: {total_qas_in_tables}")
        
        results_table = "\n".join(output_lines)
        print(results_table)
        with open(output_txt, 'w') as f:
            f.write(results_table)
        print(f"Results saved to: {output_txt}")
        return

    # Default: original stratified output
    # Evaluate each stratum
    stratified_results = {}
    all_results = {'overall': {}, 'by_stratum': {}}

    # Calculate overall metrics
    print("\nEvaluating overall results...")
    builder = QueryQrelsBuilder(args.chunks_path)
    full_queries, full_qrels = builder.build()
    overall_metrics = evaluate_metrics(run, full_qrels, verbose=False)
    all_results['overall'] = overall_metrics

    # Calculate stratified metrics
    print("\nEvaluating by stratum...")
    for stratum, (queries, qrels) in sorted(stratified_data.items()):
        print(f"  {stratum}: {len(queries)} queries")
        stratum_run = filter_run_by_queries(run, queries)
        metrics = evaluate_metrics(stratum_run, qrels, verbose=False)
        stratified_results[stratum] = metrics
        all_results['by_stratum'][stratum] = metrics

    # Format and display results
    results_table = format_metrics_table(stratified_results, overall_metrics)
    print(results_table)
    
    # Also add evidence-based tables at the end
    print("\n" + "="*80)
    print("EVIDENCE-BASED EVALUATION")
    print("="*80)
    
    # Build mapping from qid to evidence types (split on ; or , or |)
    import re
    df = pd.read_csv(args.chunks_path)
    qid_to_evidence = dict(df[['question_id', 'evidence']].drop_duplicates('question_id').values)
    def parse_evidence(ev):
        if pd.isna(ev):
            return set()
        return set(re.split(r'[;,|]', str(ev).lower().strip()))
    
    # Build sets of qids for each group
    text_qids = set()
    table_qids = set()
    visual_qids = set()
    other_qids = set()
    for qid, ev in qid_to_evidence.items():
        ev_set = parse_evidence(ev)
        if 'text' in ev_set:
            text_qids.add(qid)
        if 'table' in ev_set:
            table_qids.add(qid)
        if 'chart' in ev_set or 'diagram' in ev_set or 'image' in ev_set:
            visual_qids.add(qid)
        # Add to other if it doesn't match any of the above
        if not (('text' in ev_set) or ('table' in ev_set) or ('chart' in ev_set or 'diagram' in ev_set or 'image' in ev_set)):
            other_qids.add(qid)
    
    # Merge other_qids into visual_qids
    visual_qids.update(other_qids)
    
    # Generate evidence-based tables
    evidence_groups = [
        ("Evidence: text", text_qids),
        ("Evidence: table", table_qids),
        ("Evidence: [chart, diagram, image, other]", visual_qids),
    ]
    evidence_lines = []
    total_evidence_qas = 0
    
    for group_name, qid_set in evidence_groups:
        group_queries = {qid: full_queries[qid] for qid in qid_set if qid in full_queries}
        group_qrels = {qid: full_qrels[qid] for qid in qid_set if qid in full_qrels}
        group_run = filter_run_by_queries(run, group_queries)
        metrics = evaluate_metrics(group_run, group_qrels, verbose=False)
        # Count QAs and unique pages (actual QAs used in evaluation)
        num_qas = len(group_qrels)
        total_evidence_qas += num_qas
        unique_pages = set()
        for qrels in group_qrels.values():
            unique_pages.update(qrels.keys())
        num_pages = len(unique_pages)
        evidence_lines.append(f"\n{'='*40}\n{group_name}\nQAs: {num_qas}  Pages: {num_pages}\n{'='*40}")
        evidence_lines.append(format_metrics_table({}, metrics))
    
    # Add summary
    evidence_lines.append(f"\n{'='*40}\nSummary\n{'='*40}")
    evidence_lines.append(f"Total QAs across all evidence tables: {total_evidence_qas}")
    
    evidence_table = "\n".join(evidence_lines)
    print(evidence_table)
    
    # Save combined results
    combined_table = results_table + "\n" + evidence_table
    with open(output_txt, 'w') as f:
        f.write(combined_table)
    print(f"Results saved to: {output_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval results stratified by question_type or question_category")
    parser.add_argument("--chunks_path", type=str, help="Path to the chunks CSV file")
    parser.add_argument("--results_json", type=str, help="Path to the JSON file containing retrieval results (sorted_run.json)")
    parser.add_argument("--stratify_by", type=str, choices=['question_type', 'question_category'], default='question_category', help="Field to stratify results by (default: question_category)")
    parser.add_argument("--output_dir", type=str, help="Directory to save output files (default: same as results_json)")
    
    args = parser.parse_args()

    args.chunks_path = "src/dataset/chunks/second_pass/chunked_pages_second_pass.csv"
    args.output_dir = "src/evaluation/stratified"

    args.results_json = "src/retrievers/results/chunked_pages_second_pass_rq1/0.5/retrieved_pages/nomic-ai_colnomic-embed-multimodal-7b_sorted_run.json"
    # "question_type" | "evidence"
    args.stratify_by = "question_type"

    main(args)
