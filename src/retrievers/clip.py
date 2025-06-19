# retrievers/clip.py

from typing import Dict, Union
from pathlib import Path
from collections import OrderedDict

import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel 

from evaluation.document_provider import DocumentProvider
from retrievers.base import BaseRetriever


class ClipRetriever(BaseRetriever):
    """Page-level image–text matching using clip."""

    def __init__(
        self,
        provider: DocumentProvider,
        image_dir: Union[str, Path],
        model_name: str = "openai/clip-vit-base-patch32",  
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 16,
    ) -> None:
        super().__init__()
        filenames = list(OrderedDict.fromkeys(provider.chunk_to_page.values()))
        self.page_ids = [Path(fn).stem for fn in filenames]
        image_dir = Path(image_dir)

        # Load CLIP model & processor
        self.model = CLIPModel.from_pretrained(model_name).eval().to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = device

        # Load and embed all images
        images = [Image.open(image_dir / fn).convert("RGB") for fn in filenames]
        self.page_embeddings = self._embed_images(images, batch_size)

    def _embed_images(self, images: list[Image.Image], batch_size: int) -> torch.Tensor:
        embs = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_emb = self.model.get_image_features(**inputs)
                image_emb = image_emb / image_emb.norm(p=2, dim=-1, keepdim=True)
            embs.append(image_emb)
        return torch.cat(embs, dim=0)

    def search(
        self,
        queries: Dict[str, str],
        batch_size: int = 8,
    ) -> Dict[str, Dict[str, float]]:
        q_ids = list(queries.keys())
        q_texts = list(queries.values())
        run: Dict[str, Dict[str, float]] = {}

        for i in range(0, len(q_texts), batch_size):
            batch_qids = q_ids[i : i + batch_size]
            batch_texts = q_texts[i : i + batch_size]

            # Embed text queries
            inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                text_emb = self.model.get_text_features(**inputs)
                text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)

            # Compute cosine similarity
            scores = text_emb @ self.page_embeddings.T  # shape [B, P]
            scores = scores.cpu().numpy()

            for bi, qid in enumerate(batch_qids):
                run[qid] = {
                    pid: float(scores[bi, pi])
                    for pi, pid in enumerate(self.page_ids)
                }

        return run
