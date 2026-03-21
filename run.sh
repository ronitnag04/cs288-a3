#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ $# -ne 2 ]; then
    echo "Error: Expected exactly 2 arguments (questions file and predictions output file)"
    echo "Usage: $0 <questions_file> <predictions_output_file>"
    exit 1
fi

QUESTIONS_FILE="$1"
PREDICTIONS_FILE="$2"

if [ ! -f "$QUESTIONS_FILE" ]; then
    echo "Error: Questions file '$QUESTIONS_FILE' does not exist"
    exit 1
fi

PREDICTIONS_DIR=$(dirname "$PREDICTIONS_FILE")
if [ ! -d "$PREDICTIONS_DIR" ]; then
    echo "Creating predictions directory '$PREDICTIONS_DIR'"
    mkdir -p "$PREDICTIONS_DIR"
fi

python3 main.py "$QUESTIONS_FILE" "$PREDICTIONS_FILE" --skip_cache_key_validation
