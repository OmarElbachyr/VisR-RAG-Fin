from __future__ import annotations

from typing import Dict, List, Literal
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from evaluation.document_provider import DocumentProvider
from retrievers.base import BaseRetriever


class SentenceTransformerRetriever(BaseRetriever):
    """Dense retriever using SentenceTransformer + FAISS (full‑corpus scoring)."""

    def __init__(
        self,
        provider: DocumentProvider,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        super().__init__()
        self.model = SentenceTransformer(model_name, device=device)

        ids, embeds = provider.get("dense", encode_fn=self._encode)
        self.doc_ids: List[str] = ids
        self.chunk_to_page = provider.chunk_to_page  # map chunk_id → page_number

        self.doc_embeddings = np.asarray(embeds, dtype="float32")
        faiss.normalize_L2(self.doc_embeddings)
        dim = self.doc_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.doc_embeddings)

    # ------------------------------------------------------------------
    def _encode(self, texts: List[str]) -> List[List[float]]:  # used by provider cache
        return (
            self.model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
            .astype("float32")
            .tolist()
        )
    # ------------------------------------------------------------------

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
        """Score every chunk, then use max score per page."""
        run: Dict[str, Dict[str, float]] = {}
        docs = self.doc_embeddings

        for qid, qtext in queries.items():
            q_emb = self.model.encode([qtext], normalize_embeddings=True, show_progress_bar=False).astype("float32")
            scores = (docs @ q_emb[0]).tolist()

            page_scores: Dict[str, List[float]] = {}
            for doc_id, sc in zip(self.doc_ids, scores):
                pg = self.chunk_to_page.get(doc_id).split('.')[0] # remove .png extension
                if pg is None:
                    continue
                page_scores.setdefault(str(pg), []).append(sc)

            run[qid] = {p: self._aggregate_scores(vals, agg) for p, vals in page_scores.items()}
        return run
    