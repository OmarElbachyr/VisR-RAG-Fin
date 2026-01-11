from typing import Dict, List, Optional, Union
from pathlib import Path
from collections import OrderedDict

import torch
from PIL import Image
from transformers import AutoModel
from evaluation.classes.document_provider import DocumentProvider
from retrievers.classes.base import BaseRetriever

# Minimum dimension to avoid processing issues with very small images
MIN_IMAGE_DIMENSION = 28

class GmeQwen2VL7BRetriever(BaseRetriever):
    """
    Page-level multimodal retrieval using GME-Qwen2-VL-7B-Instruct embeddings.
    """
    def __init__(
        self,
        provider: DocumentProvider,
        image_dir: Union[str, Path],
        hard_negatives_dir: Optional[Union[str, Path]] = None,
        model_name: str = "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct",
        device_map: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 16,
        resize_ratio: float = 1.0,
        instruction: str = "Given a user query, retrieve the most relevant passages from the document corpus",
    ) -> None:
        super().__init__()
        # Store the instruction for text embeddings
        self.instruction = instruction

        # 1. Collect unique page filenames & their IDs
        filenames = list(OrderedDict.fromkeys(provider.chunk_to_page.values()))
        self.page_ids = [Path(fn).stem for fn in filenames]
        
        # Store resize parameter
        self.resize_ratio = resize_ratio
        
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
        """Check if image meets minimum dimension requirements."""
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
        """Load all images from hard negatives directory.
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
