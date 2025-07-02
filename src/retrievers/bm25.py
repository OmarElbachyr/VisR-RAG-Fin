from __future__ import annotations

from typing import Dict, Literal
import numpy as np
from rank_bm25 import BM25Okapi

from evaluation.document_provider import DocumentProvider
from retrievers.base import BaseRetriever


class BM25Retriever(BaseRetriever):
    """Whitespace‐token BM25 with page‐level aggregation."""

    def __init__(self, provider: DocumentProvider) -> None:
        super().__init__()
        self.doc_ids = provider.ids
        self.bm25 = BM25Okapi(provider.tokens)
        self.chunk_to_page = provider.chunk_to_page

    def _aggregate_scores(cls, vals: list[float], agg: Literal["max", "mean", "sum"]) -> float:
        if agg == "max":
            return float(max(vals))
        elif agg == "sum":
            return float(sum(vals))
        elif agg == "mean":
            return float(np.mean(vals))
        else:
            raise ValueError("Unsupported aggregation method.")

    def search(self, queries: Dict[str, str], agg: Literal["max", "mean", "sum"] = "max") -> Dict[str, Dict[str, float]]:
        run: Dict[str, Dict[str, float]] = {}
        for qid, qtext in queries.items():
            chunk_scores = self.bm25.get_scores(qtext.split())
            page_scores: Dict[str, list[float]] = {}
            for doc_id, score in zip(self.doc_ids, chunk_scores):
                page = self.chunk_to_page.get(doc_id).split('.')[0] # remove .png extension
                if page is None:
                    continue
                page_scores.setdefault(str(page), []).append(score)
            run[qid] = {p: self._aggregate_scores(vals, agg) for p, vals in page_scores.items()}
        return run