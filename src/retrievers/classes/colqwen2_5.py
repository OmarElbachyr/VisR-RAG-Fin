from typing import Dict, List, Optional, Union
from pathlib import Path
from collections import OrderedDict

import torch
from PIL import Image

from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

# Minimum dimension required by Qwen2 VL image processor (smart_resize factor)
MIN_IMAGE_DIMENSION = 28
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
class ColQwen2_5Retriever(BaseRetriever):
    """Direct page‐level image–text matching with ColQwen2.5 (no chunking or aggregation)."""

    def __init__(
        self,
        provider: DocumentProvider,
        image_dir: Union[str, Path] = "data/pages",
        hard_negatives_dir: Optional[Union[str, Path]] = None,
        model_name: str = "vidore/colqwen2.5-v0.2",
        device_map: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 16,
        resize_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        # Gather unique page filenames in order
        filenames = list(OrderedDict.fromkeys(provider.chunk_to_page.values()))
        self.page_ids = [Path(fn).stem for fn in filenames]
        
        # Store resize parameter
        self.resize_ratio = resize_ratio
        
        # Convert to Path
        image_dir = Path(image_dir)

        # Load ColQwen2.5 model & processor
        self.model = ColQwen2_5.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation='eager',
        ).eval()
        self.processor = ColQwen2_5_Processor.from_pretrained(model_name)
        self.device = next(self.model.parameters()).device
        print(f"Using device: {self.device}")

        # Load and embed each page image
        images, valid_filenames, num_skipped_positive = self._load_images(filenames, image_dir)
        # Update page_ids to only include valid images
        self.page_ids = [Path(fn).stem for fn in valid_filenames]
        num_positive_pages = len(images)
        print(f"  → Positive pages: {num_positive_pages}")
        
        # Load hard negatives if provided
        num_hard_negatives = 0
        num_skipped = 0
        if hard_negatives_dir is not None:
            hard_negatives_dir = Path(hard_negatives_dir)
            hard_neg_images, hard_neg_ids, skipped = self._load_hard_negatives(hard_negatives_dir)
            images.extend(hard_neg_images)
            self.page_ids.extend(hard_neg_ids)
            num_hard_negatives = len(hard_neg_images)
            num_skipped = skipped
            print(f"  → Hard negative pages: {num_hard_negatives}")
        
        print(f"  → Total pages to index: {len(images)}")
        
        # Store index statistics
        self.index_stats = {
            "positive_pages": num_positive_pages,
            "hard_negatives": num_hard_negatives,
            "skipped_duplicates": num_skipped,
            "total_indexed": len(images),
        }
        
        self.page_embeddings = self._embed_images(images, batch_size)

    def _resize_image(self, img: Image.Image) -> Image.Image:
        """Resize image by ratio (1.0 = original, 0.5 = 50%, etc.)"""
        if self.resize_ratio != 1.0:
            w, h = img.size
            new_w = int(w * self.resize_ratio)
            new_h = int(h * self.resize_ratio)
            img = img.resize((new_w, new_h), resample=Image.BILINEAR)
        
        return img

    def _is_valid_image_size(self, img: Image.Image, img_path: Path) -> bool:
        """Check if image meets minimum dimension requirements for Qwen2 VL."""
        w, h = img.size
        if w < MIN_IMAGE_DIMENSION or h < MIN_IMAGE_DIMENSION:
            print(f"⚠️  Skipping {img_path}: dimensions {w}x{h} below minimum {MIN_IMAGE_DIMENSION}px")
            return False
        return True

    def _load_images(
        self,
        filenames: List[str],
        image_dir: Path,
    ) -> tuple[List[Image.Image], List[str], int]:
        """Load images from a directory.
        
        Returns:
            images: List of loaded images
            valid_filenames: List of filenames that passed validation
            skipped: Number of skipped images due to invalid dimensions
        """
        images = []
        valid_filenames = []
        skipped = 0
        
        for fn in filenames:
            img_path = image_dir / fn
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
            
            try:
                img = Image.open(img_path).convert("RGB")
                img = self._resize_image(img)
                
                # Validate image dimensions
                if not self._is_valid_image_size(img, img_path):
                    skipped += 1
                    continue
                
                images.append(img)
                valid_filenames.append(fn)
            except Exception as e:
                raise RuntimeError(f"Failed to load {img_path}: {e}")
        
        print(f"✅ Loaded {len(images)} images from {image_dir}")
        if skipped > 0:
            print(f"  → Skipped {skipped} images with invalid dimensions")
        return images, valid_filenames, skipped

    def _load_hard_negatives(
        self,
        hard_negatives_dir: Path,
    ) -> tuple[List[Image.Image], List[str], int]:
        """
        Load all images from hard negatives directory.
        Skips images that are already in page_ids (positive pages).
        
        Returns:
            images: List of loaded images
            page_ids: List of page IDs (filename stems)
            skipped: Number of skipped duplicates
        """
        images = []
        page_ids = []
        existing_ids = set(self.page_ids)
        skipped = 0
        
        # Get all image files
        image_files = sorted(hard_negatives_dir.glob("*.png"))
        
        for img_path in image_files:
            page_id = img_path.stem
            
            # Skip if already a positive page
            if page_id in existing_ids:
                skipped += 1
                continue
            
            try:
                img = Image.open(img_path).convert("RGB")
                img = self._resize_image(img)
                
                # Validate image dimensions
                if not self._is_valid_image_size(img, img_path):
                    skipped += 1
                    continue
                
                images.append(img)
                page_ids.append(page_id)
            except Exception as e:
                print(f"⚠️  Failed to load hard negative {img_path}: {e}")
                continue
        
        print(f"✅ Loaded {len(images)} hard negatives from {hard_negatives_dir}")
        if skipped > 0:
            print(f"  → Skipped {skipped} (duplicates or invalid dimensions)")
        
        return images, page_ids, skipped

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
