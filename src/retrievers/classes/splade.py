from __future__ import annotations

from typing import Dict, Literal
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
    ) -> None:
        super().__init__()
        # preserve chunk→page mapping
        self.chunk_to_page = provider.chunk_to_page

        # load tokenizer + MLM
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        mlm = AutoModelForMaskedLM.from_pretrained(model_name).to(device_map)

        # wrap in SPLADE model
        splade_model = splade_model_module.Splade(
            model=mlm,
            tokenizer=tokenizer,
            device=device_map
        )

        # build simple docs list from provider
        ids, texts = provider.get(kind="text")
        documents = [{"id": doc_id, "text": txt} for doc_id, txt in zip(ids, texts)]

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