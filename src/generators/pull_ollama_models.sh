#!/bin/bash

# Usage: ./pull_ollama_models.sh small
#        ./pull_ollama_models.sh large
#        ./pull_ollama_models.sh all

SMALL_MODELS=(
  "qwen2.5vl:3b"
  # gemma3n models
  "gemma3n:e2b-it-q4_K_M" # instruction-tuned
  "gemma3n:e4b-it-q4_K_M" # instruction-tuned
  # gemma models
  "gemma3:1b-it-q4_K_M" 
  "gemma3:4b-it-q4_K_M"
)

LARGE_MODELS=(
  "qwen2.5vl:7b"
  "llama3.2-vision:11b"
  "gemma3:12b-it-q4_K_M"
  "gemma3:27b-it-q4_K_M"
)

if [ "$1" == "small" ]; then
  MODELS=("${SMALL_MODELS[@]}")
elif [ "$1" == "large" ]; then
  MODELS=("${LARGE_MODELS[@]}")
elif [ "$1" == "all" ]; then
  MODELS=("${SMALL_MODELS[@]}" "${LARGE_MODELS[@]}")
else
  echo "Usage: $0 [small|large|all]"
  exit 1
fi

for model in "${MODELS[@]}"; do
  echo "Pulling $model..."
  ollama pull "$model"
done
