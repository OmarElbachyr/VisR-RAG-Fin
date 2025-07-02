from typing import Dict, Set, List
import numpy as np
import pytrec_eval

class PyTrecEvaluator:
    """Light wrapper around *pytrec_eval* that prints and returns cut‑off metrics, MRR, and R‑Precision."""

    def __init__(
        self,
        metrics: Set[str] = {
            "ndcg_cut",
            "map_cut",
            "recall",
            "P",
            # global metrics -----------------
            "recip_rank",  # MRR
            "Rprec",       # R‑Precision
        },
    ) -> None:
        self.measures = metrics

    # ------------------------------------------------------------------
    def evaluate(
        self,
        run: Dict[str, Dict[str, float]],
        qrels: Dict[str, Dict[str, int]],
        k_values: List[int],
        verbose: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate retrieval *run* against *qrels* using *pytrec_eval*.

        Parameters
        ----------
        run : dict
            Mapping ``qid → {doc_id → score}`` (higher is better)
        qrels : dict
            Mapping ``qid → {doc_id → relevance}`` (int ≥ 0)
        verbose : bool, default ``True``
            Print tabular summary to stdout.
        """
        print("Evaluating run with pytrec_eval...")
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, self.measures)
        results = evaluator.evaluate(run)

        # Aggregate cut‑off metrics
        metrics_by_k: Dict[str | int, Dict[str, float]] = {}
        for k in k_values:
            metrics_by_k[k] = {
                "ndcg": np.mean([res.get(f"ndcg_cut_{k}", 0.0) for res in results.values()]),
                "map": np.mean([res.get(f"map_cut_{k}", 0.0) for res in results.values()]),
                "recall": np.mean([res.get(f"recall_{k}", 0.0) for res in results.values()]),
                "precision": np.mean([res.get(f"P_{k}", 0.0) for res in results.values()]),
            }

        # Global (non‑cut) metrics: MRR & R‑Precision
        metrics_by_k["global"] = {
            "mrr": np.mean([res.get("recip_rank", 0.0) for res in results.values()]),
            "rprec": np.mean([res.get("Rprec", 0.0) for res in results.values()]),
        }

        # Pretty print
        if verbose:
            print("\n=== Evaluation Results ===")
            for k in k_values:
                m = metrics_by_k[k]
                print(
                    f"K={k:<2}  NDCG:{m['ndcg']:.4f}  P:{m['precision']:.4f}  R:{m['recall']:.4f}  MAP:{m['map']:.4f}"
                )
            g = metrics_by_k["global"]
            print(f"GLOBAL MRR:{g['mrr']:.4f}  Rprec:{g['rprec']:.4f}")

        return metrics_by_k