from __future__ import annotations

from typing import Dict, List, Literal, Optional
from pathlib import Path
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from evaluation.classes.document_provider import DocumentProvider
from retrievers.classes.base import BaseRetriever

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'
    
class SentenceTransformerRetriever(BaseRetriever):
    """Dense retriever using SentenceTransformer + FAISS (full‑corpus scoring)."""

    def __init__(
        self,
        provider: DocumentProvider,
        model_name: str = "BAAI/bge-m3",
        device_map: str = "cuda" if torch.cuda.is_available() else "cpu",
        is_instruct: bool = False,
        task_description: str = "Given a user query, retrieve the most relevant passages from the document corpus",
        batch_size: int = 32,
        hard_negatives_chunks_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.model = SentenceTransformer(model_name, device=device_map)
        self.is_instruct = is_instruct
        self.task_description = task_description
        self.batch_size = batch_size

        # Load positive chunks
        ids, embeds = provider.get("dense", encode_fn=self._encode)
        self.doc_ids: List[str] = list(ids)
        self.chunk_to_page = dict(provider.chunk_to_page)
        all_embeddings = list(embeds)
        num_positive_chunks = len(self.doc_ids)
        print(f"  → Positive chunks: {num_positive_chunks}")
        
        # Load hard negatives if provided
        num_hard_negatives = 0
        num_skipped = 0
        if hard_negatives_chunks_path is not None:
            hard_negatives_chunks_path = Path(hard_negatives_chunks_path)
            if hard_negatives_chunks_path.exists():
                hn_provider = DocumentProvider(str(hard_negatives_chunks_path), use_nltk_preprocessor=True)
                hn_ids, hn_embeds = hn_provider.get("dense", encode_fn=self._encode)
                existing_ids = set(self.doc_ids)
                
                # Add hard negative chunks that aren't already in positive set
                for chunk_id, embed in zip(hn_ids, hn_embeds):
                    if chunk_id in existing_ids:
                        num_skipped += 1
                        continue
                    self.doc_ids.append(chunk_id)
                    all_embeddings.append(embed)
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

        # Build FAISS index with all embeddings
        self.doc_embeddings = np.asarray(all_embeddings, dtype="float32")
        faiss.normalize_L2(self.doc_embeddings)
        dim = self.doc_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.doc_embeddings)

    # ------------------------------------------------------------------
    def _encode(self, texts: List[str]) -> List[List[float]]:  # used by provider cache
        return (
            self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=False, normalize_embeddings=True)
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
            if self.is_instruct:
                q_input = get_detailed_instruct(self.task_description, qtext)
            else:
                q_input = qtext

            q_emb = self.model.encode([q_input], normalize_embeddings=True, show_progress_bar=False).astype("float32")
            scores = (docs @ q_emb[0]).tolist()

            page_scores: Dict[str, List[float]] = {}
            for doc_id, sc in zip(self.doc_ids, scores):
                pg = self.chunk_to_page.get(doc_id).split('.')[0]  # remove .png extension
                if pg is None:
                    continue
                page_scores.setdefault(str(pg), []).append(sc)

            run[qid] = {p: self._aggregate_scores(vals, agg) for p, vals in page_scores.items()}
        return run