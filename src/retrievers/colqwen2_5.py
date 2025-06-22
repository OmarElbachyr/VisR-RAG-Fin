from typing import Dict, List, Union
from pathlib import Path
from collections import OrderedDict

import torch
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from evaluation.document_provider import DocumentProvider
from retrievers.base import BaseRetriever

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
class ColQwen2_5Retriever(BaseRetriever):
    """Direct page‐level image–text matching with ColQwen2.5 (no chunking or aggregation)."""

    def __init__(
        self,
        provider: DocumentProvider,
        image_dir: Union[str, Path],
        model_name: str = "vidore/colqwen2.5-v0.2",
        device_map: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 16,
    ) -> None:
        super().__init__()
        # Gather unique page filenames in order
        filenames = list(OrderedDict.fromkeys(provider.chunk_to_page.values()))
        self.page_ids = [Path(fn).stem for fn in filenames]
        image_dir = Path(image_dir)

        # Choose attention implementation if available
        attn_impl = (
            "flash_attention_2"
            if is_flash_attn_2_available()
            else None
        )

        # Load ColQwen2.5 model & processor
        self.model = ColQwen2_5.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation=attn_impl,
        ).eval()
        self.processor = ColQwen2_5_Processor.from_pretrained(model_name)
        self.device = next(self.model.parameters()).device
        print(f"Using device: {self.device}")

        # Load and embed each page image
        images = [Image.open(image_dir / fn).convert("RGB") for fn in filenames]
        self.page_embeddings = self._embed_images(images, batch_size)

    def _embed_images(
        self,
        images: list[Image.Image],
        batch_size: int,
    ) -> torch.Tensor:
        embs = []
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

        # process in batches: embed + score immediately
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

            # record each query’s scores
            for bi, qid in enumerate(batch_qids):
                run[qid] = {
                    pid: float(scores[bi, pi])
                    for pi, pid in enumerate(self.page_ids)
                }

        return run
