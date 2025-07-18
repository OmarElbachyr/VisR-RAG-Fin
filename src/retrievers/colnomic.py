from typing import Dict, Union
from pathlib import Path
from collections import OrderedDict

import torch
from PIL import Image

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from evaluation.document_provider import DocumentProvider
from retrievers.base import BaseRetriever


class ColNomicRetriever(BaseRetriever):
    """Page-level image–text matching using nomic-ai/colnomic-embed-multimodal-3b."""

    def __init__(
        self,
        provider: DocumentProvider,
        image_dir: Union[str, Path],
        model_name: str = "nomic-ai/colnomic-embed-multimodal-3b",
        device_map: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 8,
    ) -> None:
        super().__init__()

        # Gather unique page filenames
        filenames = list(OrderedDict.fromkeys(provider.chunk_to_page.values()))
        self.page_ids = [Path(fn).stem for fn in filenames]
        image_dir = Path(image_dir)

        # Load model and processor
        self.model = ColQwen2_5.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        ).eval()
        self.processor = ColQwen2_5_Processor.from_pretrained(model_name)
        self.device = next(self.model.parameters()).device
        print(f"Using device: {self.device}")

        # Load and embed all page images
        images = [Image.open(image_dir / fn).convert("RGB") for fn in filenames]
        self.page_embeddings = self._embed_images(images, batch_size)  # [P, D]

    def _embed_images(self, images: list[Image.Image], batch_size: int) -> torch.Tensor:
        chunks = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
    
            # Resize images to consistent shape
            resized_batch = [img.resize((224, 224)) for img in batch]  # or (256, 256), depending on model expectation
    
            inputs = self.processor.process_images(resized_batch).to(self.device)
            with torch.no_grad():
                image_emb = self.model(**inputs)  # [B, D]
            chunks.append(image_emb)
    
        return torch.cat(chunks, dim=0)  # [P, D]
    

    def search(self, queries: Dict[str, str], batch_size: int = 8) -> Dict[str, Dict[str, float]]:
        qids = list(queries.keys())
        texts = list(queries.values())
        run: Dict[str, Dict[str, float]] = {}

        for i in range(0, len(texts), batch_size):
            qbatch = qids[i : i + batch_size]
            tbatch = texts[i : i + batch_size]

            inputs = self.processor.process_queries(tbatch).to(self.device)
            with torch.no_grad():
                query_emb = self.model(**inputs)  # [B, D]

            # Use the official processor scoring method
            scores_tensor = self.processor.score_multi_vector(query_emb, self.page_embeddings)
            scores = scores_tensor.cpu().numpy()

            for bi, qid in enumerate(qbatch):
                run[qid] = {
                    pid: float(scores[bi, pi])
                    for pi, pid in enumerate(self.page_ids)
                }

        return run
