from typing import Dict, Union
from pathlib import Path
from collections import OrderedDict

import torch
from PIL import Image
from transformers import AutoModel
from evaluation.document_provider import DocumentProvider
from retrievers.base import BaseRetriever

class GmeQwen2VL7BRetriever(BaseRetriever):
    """
    Page-level multimodal retrieval using GME-Qwen2-VL-7B-Instruct embeddings.
    """
    def __init__(
        self,
        provider: DocumentProvider,
        image_dir: Union[str, Path],
        model_name: str = "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct",
        device_map: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 16,
        instruction: str = "Given a user query, retrieve the most relevant passages from the document corpus",
    ) -> None:
        super().__init__()
        # Store the instruction for text embeddings
        self.instruction = instruction

        # 1. Collect unique page filenames & their IDs
        filenames = list(OrderedDict.fromkeys(provider.chunk_to_page.values()))
        self.page_ids = [Path(fn).stem for fn in filenames]
        image_dir = Path(image_dir)

        # 2. Load GME model with trust_remote_code for custom methods
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        ).eval()  # type: ignore
        self.device = next(self.model.parameters()).device

        # 3. Precompute image embeddings
        images = [Image.open(image_dir / fn).convert("RGB") for fn in filenames]
        self.page_embeddings = self._embed_images(images, batch_size)

    def _embed_images(self, images: list[Image.Image], batch_size: int) -> torch.Tensor:
        embs = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            with torch.no_grad():
                # Returns already normalized embeddings
                batch_emb = self.model.get_image_embeddings(images=batch, is_query=False)
            embs.append(batch_emb.to(self.device))
        return torch.cat(embs, dim=0)

    def _embed_queries(self, texts: list[str], batch_size: int) -> torch.Tensor:
        embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            with torch.no_grad():
                # Pass the instruction for retrieval to the model
                batch_emb = self.model.get_text_embeddings(
                    texts=batch,
                    instruction=self.instruction
                )
            embs.append(batch_emb.to(self.device))
        return torch.cat(embs, dim=0)

    def search(
        self,
        queries: Dict[str, str],
        batch_size: int = 8,
    ) -> Dict[str, Dict[str, float]]:
        """
        :param queries: dict mapping a query ID → query text
        :param batch_size: how many queries to embed at once
        :returns: dict mapping query ID → (page_id → similarity score)
        """
        q_ids = list(queries.keys())
        q_texts = list(queries.values())

        # 1. Embed queries with instruction context
        q_embs = self._embed_queries(q_texts, batch_size)  # [Q, D]

        # 2. Compute cosine similarities (dot of normalized vectors)
        scores = q_embs @ self.page_embeddings.T           # [Q, P]
        scores = scores.cpu().numpy()

        # 3. Assemble results
        return {
            qid: {
                pid: float(scores[i, j])
                for j, pid in enumerate(self.page_ids)
            }
            for i, qid in enumerate(q_ids)
        }
