from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal
from pylate import models, indexes, retrieve
import numpy as np

from evaluation.classes.document_provider import DocumentProvider
from retrievers.base import BaseRetriever
import torch


class ColBERTRetriever(BaseRetriever):
    def __init__(
        self,
        provider: DocumentProvider,
        model_name: str = "lightonai/GTE-ModernColBERT-v1",
        device_map: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 16,
        index_folder: str | Path = "indexes/pylate-index",
        index_name: str = "index",
        override: bool = True,
    ) -> None:
        super().__init__()
        self.model = models.ColBERT(model_name_or_path=model_name, device=device_map)
        self.device = next(self.model.parameters()).device
        print(f"Using device: {self.device}")

        self.index = indexes.Voyager(
            index_folder=str(index_folder),
            index_name=index_name,
            override=override,
        )
        ids, texts = provider.get("text")
        self.doc_ids: List[str] = ids
        self.chunk_to_page = provider.chunk_to_page
        embeddings = self.model.encode(
            texts,
            device=self.device,
            batch_size=batch_size,
            is_query=False,
            show_progress_bar=True,
        )
        self.index.add_documents(
            documents_ids=ids,
            documents_embeddings=embeddings,
        )
        self.searcher = retrieve.ColBERT(index=self.index)

    def _aggregate_scores(cls, vals: list[float], agg: Literal["max", "mean", "sum"]) -> float:
        if agg == "max":
            return float(max(vals))
        elif agg == "sum":
            return float(sum(vals))
        elif agg == "mean":
            return float(np.mean(vals))
        else:
            raise ValueError("Unsupported aggregation method.")

    def search(self, queries: Dict[str, str], k: int = -1, agg: Literal["max", "mean", "sum"] = "max", batch_size=32) -> Dict[str, Dict[str, float]]:
        qids = list(queries.keys())
        qtexts = list(queries.values())
        qembs = self.model.encode(
            qtexts,
            device=self.device,
            batch_size=batch_size,
            is_query=True,
            show_progress_bar=True,
        )
        results = self.searcher.retrieve(
            queries_embeddings=qembs,
            k=k,  # k=-1 to score all documents
        )
        run: Dict[str, Dict[str, float]] = {}
        for qid, hits in zip(qids, results):
            page_scores: Dict[str, List[float]] = {}
            for hit in hits:
                doc_id = hit["id"]
                score = hit["score"]
                pg = self.chunk_to_page.get(doc_id).split('.')[0] # remove .png extension
                if pg is None:
                    continue
                page_scores.setdefault(str(pg), []).append(score)
            run[qid] = {p: self._aggregate_scores(vals, agg) for p, vals in page_scores.items()}
        return run
    