# VisR-RAG-Fin: Benchmarking Multimodal RAG for Financial Documents

Experimental evaluation of multimodal Retrieval-Augmented Generation systems on visually rich financial documents. This repository implements and evaluates page-level question answering using different retrieval strategies (text-based, vision-based, hybrid) and generation models across financial document collections.

## VisR-RAG-Fin Benchmark

The main contribution is an open-source benchmark for evaluating multimodal RAG on visually rich documents:

- **Annotations** (LabelStudio format with all original annotation values preserved):
  - [merged_annotations.json](data/annotations/final_annotations/merged_annotations/merged_annotations.json) - Raw QA pairs sourced from LabelStudio
  - [merged_annotations_filtered.json](data/annotations/final_annotations/merged_annotations/merged_annotations_filtered.json) - Cleaned dataset with invalid or irrelevant QAs removed
  - [second_pass_classified_qa_category_A.json](data/annotations/final_annotations/filtered_annotations/by_category/second_pass_classified_qa_category_A.json) - **Benchmark dataset used in experiments** (post Query Rewriting step)

- **Chunks**: [src/dataset/chunks/](src/dataset/chunks/) contains processed page chunks for retrieval evaluation (specific chunk set depends on experiment)

> **Benchmark Construction**: The benchmark construction materials are available in the [vqa-benchmark](https://github.com/OmarElbachyr/vqa-benchmark) repository.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Test retrievers (includes IR evaluation)
python src/retrievers/test/global_text_retrievers.py

# Generate answers (baseline)
python src/generators/hf_models_baselines.py --model <model_name> --data <data_file>

# Generate answers (with retrieval)
python src/generators/hf_models_topk.py --model <model_name> --data <data_file> --top-k 3

# Evaluate answer correctness
python src/evaluation/llm_judge_evaluator.py
```

## Project Structure

```
├── requirements.txt        # Python dependencies
├── setup.py               # Package setup configuration
├── data/                  # Datasets and annotations → See data/README.md
└── src/
    ├── dataset/           # Data processing and chunking → See src/dataset/README.md
    ├── retrievers/        # Page retrieval implementations → See src/retrievers/README.md
    ├── generators/        # Answer generation with VLMs → See src/generators/README.md
    └── evaluation/        # IR metrics and answer evaluation → See src/evaluation/README.md
```

## Environment Variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Required variables:

- `OPENAI_API_KEY` - For OpenAI models
- `GEMINI_API_KEY` - For Gemini models
- `HF_TOKEN` - For gated HuggingFace models (optional)

## Detailed Documentation

- **[Data & Annotations](data/README.md)** - Dataset structure, annotations, and pages
- **[Dataset Processing](src/dataset/README.md)** - Chunking and data preparation
- **[Retrievers](src/retrievers/README.md)** - Text, vision, and hybrid retrieval methods
- **[Generators](src/generators/README.md)** - VLM-based answer generation
- **[Evaluation](src/evaluation/README.md)** - IR metrics and LLM judge evaluation
