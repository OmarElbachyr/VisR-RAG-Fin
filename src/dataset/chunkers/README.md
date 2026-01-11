# Chunkers - Scripts overview

This folder contains scripts for chunking annotated and noise pages into CSV files for downstream retriever testing.

- ``chunk_annotated_pages_edited.py``
  - Purpose: Chunk annotated pages (pages with QAs) into a CSV.

- ``chunk_noise_pages_final.py``
  - Purpose: Sample noise/hard-negative pages in various ways and chunk them. Final version used to create hard negatives stored in ``data/hard_negative_pages_text``, output to CSV.

**Note:** The generated CSV files are used for testing retrievers in `src/retrievers/test`.