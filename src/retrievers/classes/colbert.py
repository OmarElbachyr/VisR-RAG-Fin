from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional
from pylate import models, indexes, retrieve
import numpy as np

from evaluation.classes.document_provider import DocumentProvider
from retrievers.classes.base import BaseRetriever
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
        hard_negatives_chunks_path: Optional[str] = None,
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
        
        # Load positive chunks
        ids, texts = provider.get("text")
        self.doc_ids: List[str] = list(ids)
        self.chunk_to_page = dict(provider.chunk_to_page)
        all_texts = list(texts)
        num_positive_chunks = len(self.doc_ids)
        print(f"  → Positive chunks: {num_positive_chunks}")
        
        # Load hard negatives if provided
        num_hard_negatives = 0
        num_skipped = 0
        if hard_negatives_chunks_path is not None:
            hard_negatives_chunks_path = Path(hard_negatives_chunks_path)
            if hard_negatives_chunks_path.exists():
                hn_provider = DocumentProvider(str(hard_negatives_chunks_path), use_nltk_preprocessor=True)
                hn_ids, hn_texts = hn_provider.get("text")
                existing_ids = set(self.doc_ids)
                
                # Add hard negative chunks that aren't already in positive set
                for chunk_id, text in zip(hn_ids, hn_texts):
                    if chunk_id in existing_ids:
                        num_skipped += 1
                        continue
                    self.doc_ids.append(chunk_id)
                    all_texts.append(text)
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
        
        # Encode and index all texts
        embeddings = self.model.encode(
            all_texts,
            device=self.device,
            batch_size=batch_size,
            is_query=False,
            show_progress_bar=True,
        )
        self.index.add_documents(
            documents_ids=self.doc_ids,
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
    