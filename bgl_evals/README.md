# BGL Preprocessing and Evaluation Guide

Requirements: see `requirements.txt` for dependencies.

This document provides step-by-step instructions for preparing data and evaluating retrievers using the BGL preprocessing scripts.

## Preprocessing

Before running any evaluation, you need to prepare the following:

### 1. Extract Pages as PNG

- Extract pages from your PDFs at **200 DPI** (preferably) and save them as PNG files in:
  
  ```plaintext
  bgl_evals/pages 
  ```

- Use the filename format:
  
  ```plaintext
  <pdf_name>_<page_number>.png
  ```
  
  For example:
  ```plaintext
  3M_2018_10K_60.png
  ```

- You can refer to the existing examples in the `bgl_evals/pages` folder.

### 2. Create the QA Pairs JSON

- Prepare a JSON file linking each document to its related question–answer pairs.
- Name this file:
  
  ```plaintext
  bgl_evals/document_qa_pairs.json
  ```

- The structure should look like:
  ```json
  [
    {
      "doc_name": "3M_2018_10K",
      "qa_pairs": [
        {
          "question": "What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement.",
          "answer": "$1577.00",
          "evidence_pages": [60]
        }
        // ... more QA entries ...
      ]
    }
    // ... more documents ...
  ]
  ```

- If there are multiple evidence pages, list them all:
  ```json
  "evidence_pages": [22, 26]
  ```

### 3. Chunk Pages into CSV

- Run the chunking script to parse and split pages into chunks:
  
  ```bash
  python bgl_evals/chunk_pages_bgl.py
  ```

- This will generate `chunked_pages.csv` in the `bgl_evals/` folder.
- Move or copy this CSV to:
  
  ```plaintext
  src/dataset/chunked_pages.csv
  ```

- The CSV file is formatted for downstream evaluation.

## Evaluation

Once preprocessing is complete, you can evaluate the retrievers:

1. Ensure `chunked_pages.csv` is located in `src/dataset/`.
2. Run individual retriever tests in `src/test/retrievers_test/`, or use the global test script:
   ```bash
   python src/test/global_retrievers_test.py
   ```
3. Evaluation results will be saved to:
   ```plaintext
   src/results/retriever_results.json
   ```
