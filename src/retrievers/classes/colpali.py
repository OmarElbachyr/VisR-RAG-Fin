# retrievers/colpali.py

from typing import Dict, Union, List
from pathlib import Path
from collections import OrderedDict

import torch
import numpy as np
from PIL import Image

from colpali_engine.models import ColPali, ColPaliProcessor
from evaluation.classes.document_provider import DocumentProvider
from retrievers.classes.base import BaseRetriever


class ColPaliRetriever(BaseRetriever):
    """Direct page‐level image–text matching with ColPali (no chunking or aggregation)."""

    def __init__(
        self,
        provider: DocumentProvider,
        image_dirs: Union[str, Path, List[Union[str, Path]]] = "data/pages",
        model_name: str = "vidore/colpali-v1.3",
        device_map: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 16,
    ) -> None:
        super().__init__()
        # Gather unique page filenames in order
        filenames = list(OrderedDict.fromkeys(provider.chunk_to_page.values()))
        self.page_ids = [Path(fn).stem for fn in filenames]
        
        # Convert image_dirs to list if single path (backward compatible)
        if isinstance(image_dirs, (str, Path)):
            image_dirs = [image_dirs]
        else:
            image_dirs = list(image_dirs)
        image_dirs = [Path(d) for d in image_dirs]

        # Load ColPali model & processor
        self.model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map
        ).eval()
        self.processor = ColPaliProcessor.from_pretrained(model_name)
        self.device = next(self.model.parameters()).device
        print(f"Using device: {self.device}")

        # Load and embed each page image (search in multiple directories)
        images = self._load_images(filenames, image_dirs)
        self.page_embeddings = self._embed_images(images, batch_size)

    def _load_images(
        self,
        filenames: List[str],
        image_dirs: List[Path],
    ) -> List[Image.Image]:
        """Load images from multiple directories, searching in order."""
        images = []
        for fn in filenames:
            img = None
            for img_dir in image_dirs:
                img_path = img_dir / fn
                if img_path.exists():
                    try:
                        img = Image.open(img_path).convert("RGB")
                        break
                    except Exception as e:
                        print(f"⚠️  Failed to load {img_path}: {e}")
                        continue
            
            if img is None:
                raise FileNotFoundError(
                    f"Image not found in any directory: {fn}\n"
                    f"Searched in: {[str(d) for d in image_dirs]}"
                )
            images.append(img)
        
        print(f"✅ Loaded {len(images)} images from {len(image_dirs)} directory(ies)")
        return images

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

    def search(
        self,
        queries: Dict[str, str],
        batch_size: int = 8,
    ) -> Dict[str, Dict[str, float]]:
        """
        :param queries: dict mapping query-ID → query text
        :param batch_size: how many queries to embed/score at once
        :returns: dict mapping query-ID → { page_id: similarity_score }
        """
        q_ids = list(queries.keys())
        q_texts = list(queries.values())
        run: Dict[str, Dict[str, float]] = {}

        # process in batches: embed + score immediately
        for i in range(0, len(q_texts), batch_size):
            batch_qids = q_ids[i : i + batch_size]
            batch_texts = q_texts[i : i + batch_size]

            # embed this batch of queries
            inputs = self.processor.process_queries(batch_texts)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                q_emb = self.model(**inputs)  # shape [B, seq_len, D]

            # score against all pages: returns [B, P]
            scores_tensor = self.processor.score_multi_vector(
                q_emb, self.page_embeddings
            )
            scores = scores_tensor.cpu().numpy()

            # record each query’s scores
            for bi, qid in enumerate(batch_qids):
                run[qid] = {
                    pid: float(scores[bi, pi])
                    for pi, pid in enumerate(self.page_ids)
                }

        return run
