#!/usr/bin/env bash
set -e

rm submission.zip

zip submission.zip \
    main.py \
    embeddings.py \
    llm.py \
    run.sh \
    embeddings_cache/*

echo "Submission size: $(du -m submission.zip | cut -f1)MB"