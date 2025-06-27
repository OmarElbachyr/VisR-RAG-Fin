from typing import Dict, List
import numpy as np
import ir_measures
from ir_measures import *  # nDCG@k, P@k, ..

class Evaluator_ir:
    """Light wrapper around *ir_measures* that prints and returns cut‑off metrics, MRR, and R‑Precision."""

    def __init__(
        self,
    ) -> None:
        pass
        
    def evaluate(
        self,
        run: Dict[str, Dict[str, float]],
        qrels: Dict[str, Dict[str, int]],
        k_values: List[int],
        verbose: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate retrieval *run* against *qrels* using *ir_measures*."""
        print("Evaluating run with ir_measures...")
        
        measures = []
        for k in k_values:
            measures += [nDCG@k, P@k, Recall@k, AP@k]
        # global metrics
        measures += [RR, Rprec]
       
        results = ir_measures.calc_aggregate(measures, qrels, run)

        # Organize by k and global
        metrics_by_k: Dict[str, Dict[str, float]] = {}
        for k in k_values:
            metrics_by_k[k] = {
                'ndcg': results.get(nDCG@k, 0.0),
                'precision': results.get(P@k, 0.0),
                'recall': results.get(Recall@k, 0.0)
            }
        metrics_by_k['global'] = {
            'mrr': results.get(RR, 0.0),
            'rprec': results.get(Rprec, 0.0)
        }

        # Pretty print
        if verbose:
            print("\n=== Evaluation Results ===")
            for k in k_values:
                m = metrics_by_k[k]
                print(
                    f"K={k:<2}  NDCG:{m['ndcg']:.4f}  P:{m['precision']:.4f}  R:{m['recall']:.4f}"
                )
            g = metrics_by_k['global']
            print(f"GLOBAL MRR:{g['mrr']:.4f}  Rprec:{g['rprec']:.4f}")

        return metrics_by_k
