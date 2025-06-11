from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Any, Tuple
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class DocumentProvider:
    """Load chunk corpus and expose views and helpers."""

    def __init__(self, 
                 csv_path: str | Path, 
                 deduplicate: bool = True, 
                 use_nltk_preprocessor: bool = True
    ) -> None:
        df = pd.read_csv(csv_path, usecols=["chunk_id", "text_description", "image_filename", "query"])
        if deduplicate:
            df = df.drop_duplicates(subset=["chunk_id"])
        self._ids: List[str] = df["chunk_id"].tolist()
        self._texts: List[str] = df["text_description"].fillna("").astype(str).tolist()
        self._queries: List[str] = df["query"].fillna("").astype(str).tolist()
        self._chunk_to_page: Dict[str, int] = dict(zip(df["chunk_id"], df["image_filename"]))
        self._tokens: list[list[str]] | None = None
        self._embed_cache: Dict[Any, List[Any]] = {}

        # Flag to switch between simple .split() and your NLTK-based preprocessing
        self._use_nltk_preprocessor = use_nltk_preprocessor

    # basic access -----------------------------------------------------------
    @property
    def ids(self) -> List[str]:
        return self._ids

    @property
    def tokens(self) -> List[List[str]]:
        return self._token_view
    
    @property
    def texts(self) -> List[str]:
        return self._texts

    @property
    def chunk_to_page(self) -> Dict[str, int]:
        return self._chunk_to_page

    @property
    def stats(self) -> Dict[str, int]:
        """Return statistics about the document collection:
        - unique_pages: number of unique pages (image filenames)
        - total_chunks: total number of chunks
        - unique_queries: number of unique queries
        """
        return {
            "unique_pages": len(set(self._chunk_to_page.values())),
            "total_chunks": len(self._ids),
            "unique_queries": len(set(self._queries))
        }

    # internal views ---------------------------------------------------------
    @property
    def _token_view(self) -> List[List[str]]:
        if self._tokens is None:
            if self._use_nltk_preprocessor:
                # wrap list of texts into a dict for preprocess_text
                passages = {i: txt for i, txt in enumerate(self._texts)}
                self._tokens = self.preprocess_text(passages)
            else:
                self._tokens = [t.split() for t in self._texts]
        return self._tokens
    
    def preprocess_text(self, passages: Dict[int, str]) -> List[List[str]]:
        """
        Preprocess text by performing the following steps:
            - Remove stopwords
            - Strip punctuation (retain only alphanumeric tokens)
            - Convert all words to lowercase
        """
        stop_words = set(stopwords.words("english"))
        tokenized_list = [
            [
                word.lower()
                for word in word_tokenize(sentence)
                if word.isalnum() and word.lower() not in stop_words
            ]
            for sentence in passages.values()
        ]
        return tokenized_list

    def _embedding_view(self, encode_fn: Callable[[List[str]], List[Any]]) -> List[Any]:
        if encode_fn not in self._embed_cache:
            self._embed_cache[encode_fn] = encode_fn(self._texts)
        return self._embed_cache[encode_fn]

    # main getter ------------------------------------------------------------
    def get(
        self,
        kind: str = "text",
        *,
        encode_fn: Callable[[List[str]], List[Any]] | None = None,
    ) -> Tuple[List[str], List[Any]]:
        if kind in {"text", "raw"}:
            return self._ids, self._texts
        if kind in {"tokens", "bm25"}:
            return self._ids, self._token_view
        if kind == "dense":
            if encode_fn is None:
                raise ValueError("encode_fn required for dense view")
            return self._ids, self._embedding_view(encode_fn)
        raise ValueError("unknown kind")
