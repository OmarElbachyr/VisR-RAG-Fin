# Generators Module

This module contains generator implementations for answering visual questions (VQA) using different model providers and retrieval strategies.

## Overview

The generators module provides two main approaches for answer generation:

1. **Baselines**: Generate answers directly from the image/question without retrieval context
2. **Top-K**: Generate answers using the top-k retrieved pages/chunks as context

## Supported Models

The module supports multiple model providers:

- **HuggingFace**: `hf_models_baselines.py`, `hf_models_topk.py`
  - Uses HuggingFace transformers library
  - Supports vision-language models like Qwen, InternVL, Gemma, etc.
  
- **OpenAI**: `openai_models_baselines.py`, `openai_models_topk.py`
  - Uses OpenAI API (GPT models)
  - Requires `OPENAI_API_KEY` environment variable

- **Gemini**: `gemini_models_baselines.py`, `gemini_models_topk.py`
  - Uses Google Gemini API
  - Requires `GEMINI_API_KEY` environment variable

- **Ollama**: `ollama_models_baselines.py`, `ollama_models_topk.py`
  - Uses locally running Ollama models

## Directory Structure

```
generators/
├── {provider}_models_baselines.py    # Generator scripts using baseline approach
├── {provider}_models_topk.py         # Generator scripts using top-k retrieval
├── prompt_utils.py                   # Utility functions for loading prompts
├── prompts/                          # Prompt templates
└── rq_5/                             # Research question 5 specific implementations
    ├── top_k_hybrid_generator.py     # Hybrid retrieval approach
    ├── top_k_text_generator.py       # Text-only generation
    └── pull_ollama_models.sh         # Script to pull required Ollama models
```

## Prompt Templates

Prompts are loaded using `prompt_utils.load_prompt()` function.

## Usage

### Running Generators

Each generator script can be executed independently with command-line arguments:

```bash
python hf_models_baselines.py --model <model_name> --data <data_file> --annotations <annotations_file>
python openai_models_topk.py --model gpt-4o-mini --data <data_file> --top-k 3
```

Common arguments:
- `--model`: Model identifier or name
- `--data`: Path to input data file (JSON format)
- `--annotations`: Path to annotations file
- `--top-k`: Number of retrieved pages to use (default: 1 or 3)

### Model Importing

```python
from hf_models_topk import HuggingFaceTopKGenerator
from openai_models_baselines import OpenAIBaselineGenerator

# Initialize generator
generator = HuggingFaceTopKGenerator(
    data_file="data.json",
    annotations_file="annotations.json",
    models=["Qwen/Qwen2.5-VL-3B-Instruct"],
    top_k=3
)

# Generate answers
results = generator.generate()
```

## Environment Setup

Required environment variables:

- `OPENAI_API_KEY`: For OpenAI models
- `GEMINI_API_KEY`: For Gemini models
- `HF_TOKEN`: For HuggingFace gated models (optional)

Set these in a `.env` file.

## Research Question 5 (rq_5/)

Contains specialized implementations for hybrid and text-based generation approaches with corresponding output directories, for research question number 5 of the paper.

## Output Format

Generated results are typically saved as JSON files with the following structure:

```json
{
  "question_id": "...",
  "model": "model_name",
  "generated_answer": "...",
  "top_k": 1,
  "context": [...]
}
```
