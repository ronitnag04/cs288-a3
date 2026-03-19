#!/usr/bin/env bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

DOWNLOAD_FLAG="--download-embedding-model"

if [[ "${1:-}" == "$DOWNLOAD_FLAG" ]]; then
  # Downloads the embedding model weights into the local HF cache so you can run
  # RETRIEVAL_MODE=dense offline later (e.g., on Gradescope-like no-network runs).
  MODEL_NAME="${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
  echo "Downloading embedding model: ${MODEL_NAME}"
  python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${MODEL_NAME}')"
  echo "Done."
  exit 0
fi

echo "Usage:"
echo "  bash run.sh --download-embedding-model"
