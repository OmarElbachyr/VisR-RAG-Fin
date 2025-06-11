from typing import Dict, Literal
import numpy as np

from retrievers.base import BaseRetriever
from retrievers.colbert_retriever import ColBERTRetriever
from retrievers.splade_retriever import SpladeRetriever
from evaluation.document_provider import DocumentProvider


class CoLiPLaIRetriever(BaseRetriever):
    """
    Hybrid retriever combining ColBERT and SPLADE via weighted interpolation.
    """

    def __init__(
        self,
        provider: DocumentProvider,
        splade_weight: float = 0.5,
        colbert_weight: float = 0.5,
        **kwargs
    ):
        super().__init__()
        assert np.isclose(splade_weight + colbert_weight, 1.0), "Weights must sum to 1."
        self.splade = SpladeRetriever(provider, **kwargs)
        self.colbert = ColBERTRetriever(provider, **kwargs)
        self.splade_weight = splade_weight
        self.colbert_weight = colbert_weight

    def search(
        self,
        queries: Dict[str, str],
        k: int = -1,
        agg: Literal["max", "mean", "sum"] = "max"
    ) -> Dict[str, Dict[str, float]]:

        splade_scores = self.splade.search(queries, agg=agg)
        colbert_scores = self.colbert.search(queries, k=k, agg=agg)

        run: Dict[str, Dict[str, float]] = {}

        for qid in queries:
            combined_scores = {}
            splade_docs = splade_scores.get(qid, {})
            colbert_docs = colbert_scores.get(qid, {})
            all_pages = set(splade_docs) | set(colbert_docs)

            for page in all_pages:
                s_score = splade_docs.get(page, 0.0)
                c_score = colbert_docs.get(page, 0.0)
                combined_score = self.splade_weight * s_score + self.colbert_weight * c_score
                combined_scores[page] = combined_score

            run[qid] = combined_scores

        return run
