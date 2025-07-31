#!/bin/bash
# filepath: run_internvl_models.sh

python3 src/generators/hf_models_topk.py --models OpenGVLab/InternVL3-8B-hf
python3 src/generators/hf_models_topk.py --models OpenGVLab/InternVL3-2B-hf