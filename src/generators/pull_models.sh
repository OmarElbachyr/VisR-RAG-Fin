#!/bin/bash

# List of P14 Ollama models to pull
MODELS=(
  "qwen2.5vl:3b-fp16"
  "qwen2.5vl:7b-fp16"
  "gemma3:4b-it-fp16"
  "gemma3:12b-it-fp16"
)

for MODEL in "${MODELS[@]}"; do
  echo "Pulling $MODEL..."
  ollama pull "$MODEL"
done

echo "All P14 models pulled."