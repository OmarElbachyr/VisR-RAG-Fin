from __future__ import annotations

from typing import Dict, Literal, Optional
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sparsembed import model as splade_model_module, retrieve as splade_retrieve

from evaluation.classes.document_provider import DocumentProvider
from retrievers.classes.base import BaseRetriever


class SpladeRetriever(BaseRetriever):
    """Sparse‐Max SPLADE with page‐level aggregation."""

    def __init__(
        self,
        provider: DocumentProvider,
        model_name: str = "naver/splade-v3",
        device_map: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 16,
        k_tokens_index: int = 256,
        hard_negatives_chunks_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        
        # Load tokenizer + MLM
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        mlm = AutoModelForMaskedLM.from_pretrained(model_name).to(device_map)

        # wrap in SPLADE model
        splade_model = splade_model_module.Splade(
            model=mlm,
            tokenizer=tokenizer,
            device=device_map
        )

        # Load positive chunks
        ids, texts = provider.get(kind="text")
        documents = [{"id": doc_id, "text": txt} for doc_id, txt in zip(ids, texts)]
        self.chunk_to_page = dict(provider.chunk_to_page)
        num_positive_chunks = len(documents)
        print(f"  → Positive chunks: {num_positive_chunks}")
        
        # Load hard negatives if provided
        num_hard_negatives = 0
        num_skipped = 0
        if hard_negatives_chunks_path is not None:
            hard_negatives_chunks_path = Path(hard_negatives_chunks_path)
            if hard_negatives_chunks_path.exists():
                hn_provider = DocumentProvider(str(hard_negatives_chunks_path), use_nltk_preprocessor=True)
                hn_ids, hn_texts = hn_provider.get(kind="text")
                existing_ids = set(ids)
                
                # Add hard negative chunks that aren't already in positive set
                for chunk_id, text in zip(hn_ids, hn_texts):
                    if chunk_id in existing_ids:
                        num_skipped += 1
                        continue
                    documents.append({"id": chunk_id, "text": text})
                    self.chunk_to_page[chunk_id] = hn_provider.chunk_to_page[chunk_id]
                    num_hard_negatives += 1
                
                print(f"  → Hard negative chunks: {num_hard_negatives}")
                if num_skipped > 0:
                    print(f"  → Skipped {num_skipped} duplicates")
        
        print(f"  → Total chunks indexed: {len(documents)}")
        
        # Store index statistics
        self.index_stats = {
            "positive_chunks": num_positive_chunks,
            "hard_negatives": num_hard_negatives,
            "skipped_duplicates": num_skipped,
            "total_indexed": len(documents),
        }

        # init retriever and index all chunks
        retr = splade_retrieve.SpladeRetriever(
            key="id",
            on=["text"],
            model=splade_model
        )
        retr = retr.add(
            documents=documents,
            batch_size=batch_size,
            k_tokens=k_tokens_index
        )
        self._retriever = retr

        # store defaults
        self.batch_size = batch_size
        self.default_k_tokens = k_tokens_index

    @staticmethod
    def _aggregate_scores(vals: list[float], agg: Literal["max", "mean", "sum"]) -> float:
        if agg == "max":
            return float(max(vals))
        elif agg == "sum":
            return float(sum(vals))
        elif agg == "mean":
            return float(np.mean(vals))
        else:
            raise ValueError(f"Unsupported aggregation method: {agg!r}")

    def search(
        self,
        queries: Dict[str, str],
        agg: Literal["max", "mean", "sum"] = "max",
        *,
        k_tokens: int | None = None,
        k: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """
        :param queries: mapping qid→text
        :param agg: how to roll up chunk scores to page scores
        :param k_tokens: max activated tokens per query (defaults to indexing value)
        :param k: how many chunk‐hits to fetch before aggregation
        :returns: mapping qid→{ page_id: aggregated_score }
        """
        k_tokens = k_tokens or self.default_k_tokens

        # run SPLADE; one list of hits per query
        raw = self._retriever(
            list(queries.values()),
            k_tokens=k_tokens,
            k=k,
            batch_size=self.batch_size
        )

        run: Dict[str, Dict[str, float]] = {}
        for qid, hits in zip(queries.keys(), raw):
            page_scores: Dict[str, list[float]] = defaultdict(list)
            for hit in hits:
                doc_id = hit["id"]
                score = hit["similarity"]
                page = self.chunk_to_page.get(doc_id)
                if page is None:
                    continue
                # strip extension if any
                page = str(page).split(".", 1)[0]
                page_scores[page].append(score)

            run[qid] = {
                page: self._aggregate_scores(vals, agg)
                for page, vals in page_scores.items()
            }

        return run