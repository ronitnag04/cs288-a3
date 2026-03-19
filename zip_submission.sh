#!/usr/bin/env bash
set -e

zip submission.zip \
    main.py \
    embeddings.py \
    llm.py \
    run.sh \
    embeddings_cache/*