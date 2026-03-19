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
    echo "Error: Predictions directory '$PREDICTIONS_DIR' does not exist"
    exit 1
fi

MODEL_NAME="${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
python3 -c "
from sentence_transformers import SentenceTransformer
try:
    SentenceTransformer('${MODEL_NAME}', local_files_only=True)
    print('Embedding model ${MODEL_NAME} found in local cache')
except:
    # If not found locally, download it
    print('Embedding model ${MODEL_NAME} not found locally, downloading...')
    SentenceTransformer('${MODEL_NAME}')
    print('Embedding model ${MODEL_NAME} downloaded')
"

python3 main.py "$QUESTIONS_FILE" "$PREDICTIONS_FILE"
