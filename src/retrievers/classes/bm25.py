from __future__ import annotations

from typing import Dict, Literal, Optional
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi

from evaluation.classes.document_provider import DocumentProvider
from retrievers.classes.base import BaseRetriever


class BM25Retriever(BaseRetriever):
    """Whitespace‐token BM25 with page‐level aggregation."""

    def __init__(self, provider: DocumentProvider, hard_negatives_chunks_path: Optional[str] = None) -> None:
        super().__init__()
        
        # Load positive chunks
        self.doc_ids = provider.ids
        all_tokens = list(provider.tokens)
        self.chunk_to_page = dict(provider.chunk_to_page)
        num_positive_chunks = len(self.doc_ids)
        print(f"  → Positive chunks: {num_positive_chunks}")
        
        # Load hard negatives if provided
        num_hard_negatives = 0
        num_skipped = 0
        if hard_negatives_chunks_path is not None:
            hard_negatives_chunks_path = Path(hard_negatives_chunks_path)
            if hard_negatives_chunks_path.exists():
                hn_provider = DocumentProvider(str(hard_negatives_chunks_path), use_nltk_preprocessor=True)
                existing_ids = set(self.doc_ids)
                
                # Add hard negative chunks that aren't already in positive set
                for chunk_id, tokens in zip(hn_provider.ids, hn_provider.tokens):
                    if chunk_id in existing_ids:
                        num_skipped += 1
                        continue
                    self.doc_ids.append(chunk_id)
                    all_tokens.append(tokens)
                    self.chunk_to_page[chunk_id] = hn_provider.chunk_to_page[chunk_id]
                    num_hard_negatives += 1
                
                print(f"  → Hard negative chunks: {num_hard_negatives}")
                if num_skipped > 0:
                    print(f"  → Skipped {num_skipped} duplicates")
        
        print(f"  → Total chunks indexed: {len(self.doc_ids)}")
        
        # Store index statistics
        self.index_stats = {
            "positive_chunks": num_positive_chunks,
            "hard_negatives": num_hard_negatives,
            "skipped_duplicates": num_skipped,
            "total_indexed": len(self.doc_ids),
        }
        
        # Build BM25 index with all tokens
        self.bm25 = BM25Okapi(all_tokens)

    @staticmethod
    def _aggregate_scores(vals: list[float], agg: Literal["max", "mean", "sum"]) -> float:
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
            run[qid] = {p: BM25Retriever._aggregate_scores(vals, agg) for p, vals in page_scores.items()}
        return run