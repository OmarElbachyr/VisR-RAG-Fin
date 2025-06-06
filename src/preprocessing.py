from pathlib import Path
from haystack.utils import convert_files_to_documents
from haystack.nodes import PreProcessor

def ingest_corpus(raw_dir: Path, chunk_length: int = 200, overlap: int = 20):
    """Yield Haystack Documents with enriched metadata."""
    docs = convert_files_to_documents(raw_dir, split_paragraphs=False)
    splitter = PreProcessor(
        split_by="word",
        split_length=chunk_length,
        split_overlap=overlap,
        respect_sentence_boundary=True
    )
    for d in splitter.process(docs):
        # add / normalise metadata keys here
        d.meta.setdefault("source_id", d.id)   # fallback if missing
        yield d
