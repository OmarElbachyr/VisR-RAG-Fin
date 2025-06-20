from typing import Dict, Union
from pathlib import Path
from collections import OrderedDict

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel, BitsAndBytesConfig

from evaluation.document_provider import DocumentProvider
from retrievers.base import BaseRetriever


class SigLIPRetriever(BaseRetriever):
    """Page-level image–text matching using SigLIP."""

    def __init__(
        self,
        provider: DocumentProvider,
        image_dir: Union[str, Path],
        model_name: str = "google/siglip-base-patch16-224",
        device_map: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 16,
    ) -> None:
        super().__init__()
        filenames = list(OrderedDict.fromkeys(provider.chunk_to_page.values()))
        self.page_ids = [Path(fn).stem for fn in filenames]
        image_dir = Path(image_dir)

        # Load SigLIP model & processor with quantization config
        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float32
        )   
        self.model = AutoModel.from_pretrained(
            model_name,
            device_map=device_map,
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = next(self.model.parameters()).device
        print(f"Using device: {self.device}")
        
        # Load and embed all images
        images = [Image.open(image_dir / fn).convert("RGB") for fn in filenames]
        self.page_embeddings = self._embed_images(images, batch_size)

    def _embed_images(self, images: list[Image.Image], batch_size: int) -> torch.Tensor:
        embs = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")

            # Cast pixel_values to model dtype (usually float32 with 4-bit quantization)
            pixel_values = inputs["pixel_values"].to(self.device, dtype=torch.float32)

            with torch.no_grad():
                image_emb = self.model.get_image_features(pixel_values=pixel_values)
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
            batch_texts = [f"This is a photo of {t}" for t in q_texts[i : i + batch_size]]

            inputs = self.processor(text=batch_texts, return_tensors="pt", padding="max_length").to(self.device)
            with torch.no_grad():
                text_emb = self.model.get_text_features(**inputs)
                text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)

            scores = text_emb @ self.page_embeddings.T  # cosine similarity
            scores = scores.cpu().numpy()

            for bi, qid in enumerate(batch_qids):
                run[qid] = {
                    pid: float(scores[bi, pi])
                    for pi, pid in enumerate(self.page_ids)
                }

        return run
