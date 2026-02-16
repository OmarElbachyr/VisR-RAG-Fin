# Dataset - Data Processing and Chunking

This folder contains scripts and utilities for processing dataset annotations and generating chunks for retriever evaluation.

## Directory Structure

- **`chunkers/`** - Scripts for chunking annotated and noise pages into CSV format
  - `chunk_annotated_pages_final.py` - Chunks annotated pages (pages with QAs) into CSV
  - `chunk_noise_pages_final.py` - Samples and chunks hard-negative/noise pages into CSV
  
- **`chunks/`** - Generated CSV files from chunking scripts
  - `final_chunks/` - Processed annotated pages
  - `noise_pages_chunks/` - Hard negative page samples
  - `rq_3/` - Research question 3 specific chunks
  - `second_pass/` - Secondary processing pass chunks

## Usage

Generated CSV files from the `chunks/` directory are used for testing retrievers in `src/retrievers/test`.
