# Evaluation - Information Retrieval and Answer Correctness Assessment

This folder contains scripts and utilities for evaluating retriever performance, computing IR metrics, and assessing answer correctness using LLM judges.

## Directory Structure

- **`classes/`** - Core evaluation classes and utilities
  - `document_provider.py` - Loads chunk corpus from CSV files and provides views with preprocessing (tokenization, stopword removal, embeddings)
  - `ir_evaluator.py` - IR evaluation using ir_measures library; computes nDCG, MRR... etc
  - `query_qrel_builder.py` - Builds query and relevance judgments (qrels) from chunk CSV files for IR evaluation

- **`stratified/`** - Stratified evaluation results by question type
  - Contains sorted run files stratified by question type from different retriever models

## Main Scripts

- **`compute_metrics.py`** - Computes accuracy and other metrics (generation rate, judgment rate, success rates) for judged QA results; aggregates metrics by model, retriever, and top-k values
- **`evaluate_stratified.py`** - Performs stratified IR evaluation by question type or category; evaluates retrieval results against relevance judgments with metrics broken down by stratification field
- **`llm_judge_evaluator.py`** - Uses LLM judges (Ollama or OpenAI) to evaluate answer correctness - the LLM judge assigns "yes" (fully correct) or "no" (incorrect) verdicts based on questions, reference answers, and generated answers

