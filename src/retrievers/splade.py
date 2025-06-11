import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import Dict, Literal

from retrievers.base import BaseRetriever
from evaluation.document_provider import DocumentProvider


class SpladeRetriever(BaseRetriever):
    """Sparse SPLADE retriever with page-level aggregation."""

    def __init__(self, provider: DocumentProvider, model_name: str = "naver/splade-cocondenser-ensembledistil") -> None:
        super().__init__()
        self.doc_ids = provider.ids
        self.documents = provider.texts
        self.chunk_to_page = provider.chunk_to_page
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device).eval()
        self.doc_embs = self._encode_texts(self.documents)

    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            sparse_rep = torch.log1p(F.relu(logits))
            sparse_rep = torch.max(sparse_rep, dim=1).values
        return sparse_rep.cpu()

    def _aggregate_scores(self, vals: list[float], agg: Literal["max", "mean", "sum"]) -> float:
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
            query_emb = self._encode_texts([qtext])[0]
            scores = torch.matmul(self.doc_embs, query_emb).numpy()
            page_scores: Dict[str, list[float]] = {}
            for doc_id, score in zip(self.doc_ids, scores):
                page = self.chunk_to_page.get(doc_id)
                if page is None:
                    continue
                page = page.split('.')[0]  # remove .png or file extension
                page_scores.setdefault(page, []).append(score)
            run[qid] = {p: self._aggregate_scores(vals, agg) for p, vals in page_scores.items()}
        return run