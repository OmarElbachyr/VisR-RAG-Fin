from typing import Dict, Set, List
import numpy as np
import pytrec_eval

class Evaluator:
    def __init__(self, k_values: List[int] = [1, 3, 5, 10],
                 metrics: Set[str] = {"ndcg_cut", "map_cut", "recall", "P"}):
        self.k_values = k_values
        self.metrics = metrics

    def evaluate(self, run: Dict[str, Dict[str, float]], 
                qrels: Dict[str, Dict[str, int]], 
                verbose: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Evaluate retrieval results using pytrec_eval
        Args:
            run: Dict[qid -> Dict[doc_id -> score]]
            qrels: Dict[qid -> Dict[doc_id -> relevance]]
            verbose: Whether to print results
        Returns:
            Dict of evaluation metrics
        """
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, self.metrics)
        results = evaluator.evaluate(run)
        
        # Aggregate metrics
        metrics_by_k = {}
        for k in self.k_values:
            metrics_by_k[k] = {
                'ndcg': np.mean([res.get(f"ndcg_cut_{k}", 0.0) for res in results.values()]),
                'map': np.mean([res.get(f"map_cut_{k}", 0.0) for res in results.values()]),
                'recall': np.mean([res.get(f"recall_{k}", 0.0) for res in results.values()]),
                'precision': np.mean([res.get(f"P_{k}", 0.0) for res in results.values()])
            }
        
        if verbose:
            print("\n=== Evaluation Results ===")
            for k in self.k_values:
                metrics = metrics_by_k[k]
                print(f"K={k:<2}  NDCG:{metrics['ndcg']:.4f}  "
                      f"MAP:{metrics['map']:.4f}  "
                      f"R:{metrics['recall']:.4f}  "
                      f"P:{metrics['precision']:.4f}")
        
        return metrics_by_k