from typing import Dict, List, Union
from pathlib import Path
from collections import OrderedDict

import torch
from PIL import Image

from colpali_engine.models import ColIdefics3, ColIdefics3Processor

from evaluation.classes.document_provider import DocumentProvider
from retrievers.classes.base import BaseRetriever

def pad_embeddings(emb_list: List[torch.Tensor], max_chunks: int) -> torch.Tensor:
    """
    Pads/truncates a list of [seq_len, dim] tensors to shape [B, max_chunks, dim].
    """
    if not emb_list:
        raise ValueError("emb_list is empty")

    dim = emb_list[0].shape[-1]
    batch_size = len(emb_list)
    padded = emb_list[0].new_zeros(batch_size, max_chunks, dim)

    for i, e in enumerate(emb_list):
        seq_len = min(e.size(0), max_chunks)
        padded[i, :seq_len] = e[:seq_len]

    return padded

class ColSmol(BaseRetriever):
    """Direct page‐level image–text matching with ColIdefics3 (no chunking or aggregation)."""

    def __init__(
        self,
        provider: DocumentProvider,
        image_dir: Union[str, Path],
        model_name: str = "vidore/colSmol-256M",
        device_map: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 16,
    ) -> None:
        super().__init__()
        # Gather unique page filenames in order
        filenames = list(OrderedDict.fromkeys(provider.chunk_to_page.values()))
        self.page_ids = [Path(fn).stem for fn in filenames]
        image_dir = Path(image_dir)

        # Load ColIdefics3 model & processor
        self.model = ColIdefics3.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation='eager',
        ).eval()
        self.processor = ColIdefics3Processor.from_pretrained(model_name)
        self.device = next(self.model.parameters()).device
        print(f"Using device: {self.device}")

        # Load and embed each page image
        images = [Image.open(image_dir / fn).convert("RGB") for fn in filenames]
        self.page_embeddings = self._embed_images(images, batch_size)

    def _embed_images(
        self,
        images: List[Image.Image],
        batch_size: int,
    ) -> torch.Tensor:
        embs: List[torch.Tensor] = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = self.processor.process_images(batch)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = self.model(**inputs)
            embs.extend(emb)

        max_chunks = max(e.size(0) for e in embs) 
        embs_padded =  pad_embeddings(embs, max_chunks) 

        # for i, e in enumerate(embs_padded):
        #     print(f"Embedding {i}: shape {e.shape}")
        # print('----------') 

        return embs_padded

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

        for i in range(0, len(q_texts), batch_size):
            batch_qids = q_ids[i : i + batch_size]
            batch_texts = q_texts[i : i + batch_size]

            # embed this batch of queries
            inputs = self.processor.process_queries(batch_texts)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                q_emb = self.model(**inputs)

            # score against all pages: returns [B, P]
            scores_tensor = self.processor.score_multi_vector(
                q_emb, self.page_embeddings
            )
            scores = scores_tensor.cpu().numpy()

            for bi, qid in enumerate(batch_qids):
                run[qid] = {
                    pid: float(scores[bi, pi])
                    for pi, pid in enumerate(self.page_ids)
                }

        return run
