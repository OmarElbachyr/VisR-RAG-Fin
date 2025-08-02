# retrievers/biqwen2_5.py

from typing import Dict, Union
from pathlib import Path
from collections import OrderedDict

import torch
from PIL import Image

from colpali_engine.models import BiQwen2_5, BiQwen2_5_Processor
from evaluation.classes.document_provider import DocumentProvider
from retrievers.classes.base import BaseRetriever

class BiQwen2_5Retriever(BaseRetriever):
    """Page‐level multimodal retrieval using BiQwen2_5 embeddings."""
    def __init__(
        self,
        provider: DocumentProvider,
        image_dir: Union[str, Path],
        model_name: str = "nomic-ai/nomic-embed-multimodal-3b",
        device_map: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 16,
    ) -> None:
        super().__init__()
        # Unique page filenames in order
        filenames = list(OrderedDict.fromkeys(provider.chunk_to_page.values()))
        self.page_ids = [Path(fn).stem for fn in filenames]
        image_dir = Path(image_dir)

        # Load the BiQwen2_5 model & processor
        self.model = BiQwen2_5.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation="eager",
        ).eval()
        self.processor = BiQwen2_5_Processor.from_pretrained(model_name)
        self.device = next(self.model.parameters()).device

        # Embed all page images
        images = [Image.open(image_dir / fn).convert("RGB") for fn in filenames]
        self.page_embeddings = self._embed_images(images, batch_size)

    def _embed_images(self, images: list[Image.Image], batch_size: int) -> torch.Tensor:
        embs = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = self.processor.process_images(batch)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = self.model(**inputs)
            embs.append(emb)
        return torch.cat(embs, dim=0)

    def _embed_queries(self, queries: list[str], batch_size: int) -> torch.Tensor:
        embs = []
        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            inputs = self.processor.process_queries(batch)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = self.model(**inputs)
            embs.append(emb)
        return torch.cat(embs, dim=0)

    def search(
        self,
        queries: Dict[str, str],
        batch_size: int = 8,
    ) -> Dict[str, Dict[str, float]]:
        """
        :param queries: dict mapping query-ID → query text
        :param batch_size: how many queries to embed at once
        :returns: dict mapping query-ID → { page_id: similarity_score }
        """
        q_ids = list(queries.keys())
        q_texts = list(queries.values())
        # Embed queries in batches
        query_embeddings = self._embed_queries(q_texts, batch_size)
        # Score against all page embeddings
        scores = self.processor.score(
            [qe for qe in torch.unbind(query_embeddings)],
            [pe for pe in torch.unbind(self.page_embeddings)]
        )
        scores = scores.cpu().numpy()
        # Build and return result dict
        return {
            qid: {
                pid: float(scores[i, j]) for j, pid in enumerate(self.page_ids)
            }
            for i, qid in enumerate(q_ids)
        }
