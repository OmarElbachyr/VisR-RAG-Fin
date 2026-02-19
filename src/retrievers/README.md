# Retrievers Module

Page retrieval implementations for VQA-IR-QA tasks.

## Usage

`src/retrievers/test/retrievers_test/` contains individual tests for each retriever.

Use scripts in `src/retrievers/test/` to run retrievers:
- `global_text_retrievers.py` - Test all text-based retrievers
- `global_vision_retrievers.py` - Test all vision-based retrievers  
- `global_colvision_retrievers.py` - Test vision-language retrievers
- `run_all_global_tests.sh` - Run all tests sequentially

## Directory Structure

```
retrievers/
├── classes/         # Retriever implementations
├── test/            # Example scripts & unit tests
│   ├── retrievers_test/  # Individual retriever tests
│   └── global_*.py       # Global test scripts
└── results/         # Evaluation results
```

## Testing

```bash
# Test text-based retrievers
python src/retrievers/test/global_text_retrievers.py

# Test vision-based retrievers
python src/retrievers/test/global_vision_retrievers.py

# Test vision-language retrievers
python src/retrievers/test/global_colvision_retrievers.py

# Run all tests
bash src/retrievers/test/run_all_global_tests.sh
```