#!/bin/bash

# Run: nohup bash src/test/run_all_global_tests.sh > logs/global_tests_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -e

# Run all global test scripts in sequence

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 -u "$SCRIPT_DIR/global_text_retrievers.py"
echo "global_text_retrievers.py finished."

python3 -u "$SCRIPT_DIR/global_vision_retrievers.py"
echo "global_vision_retrievers.py finished."

python3 -u "$SCRIPT_DIR/global_colvision_retrievers.py"
echo "global_colvision_retrievers.py finished."

echo "All global test scripts completed."
